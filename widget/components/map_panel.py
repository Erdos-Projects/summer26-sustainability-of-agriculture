"""Map panel: owns the Leaflet map and all region-of-interest selection.

Selection can be a single point (click) or an area, drawn either as a
rectangle or a free-form polygon. Whichever mode is active, the result is
normalized to a GeoJSON geometry (Point or Polygon) and written to the
shared `region-geom` store — that store is the only thing the info and
forecast panels need to read.

This module also owns the IWQIS site markers and, via `selected-site`,
which site (if any) is currently selected for the info panel's timeseries
display.

The map's `dl.LayerGroup` ids below act as render "slots": this module owns
their layout placement, but the info and forecast panels populate
`mapunit-layer` and `forecast-layer` respectively.
"""

import base64
import json
from pathlib import Path

import geopandas as gpd
import plotly.graph_objects as go
import dash_leaflet as dl
from dash import Input, Output, State, html, dcc, no_update, ALL, ctx
from shapely.geometry import shape, Point

from data import water, map_overlays

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

# Inverted mask: world polygon with Iowa bounding box (+ padding) cut as a hole.
# Dims everything outside the box at 50% opacity while leaving data layers on top unaffected.
_IOWA_MASK = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]],  # world
            [[-99.0, 39.0], [-87.5, 39.0], [-87.5, 45.0], [-99.0, 45.0], [-99.0, 39.0]],  # Iowa box
        ],
    },
    "properties": {},
}

_NHD_MIN_ORDER = 5
_NHD_SIMPLIFY_TOLERANCE = 0.005  # degrees; ~50% vertex reduction at state zoom

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_FLOWLINES_ASSET = _ASSETS_DIR / "iowa_flowlines.geojson"
_WATERBODIES_ASSET = _ASSETS_DIR / "iowa_waterbodies.geojson"


def _ensure_nhd_assets():
    """Write simplified NHD GeoJSON to assets/ at startup if not already present."""
    if _FLOWLINES_ASSET.exists() and _WATERBODIES_ASSET.exists():
        return
    _ASSETS_DIR.mkdir(exist_ok=True)
    if not _FLOWLINES_ASSET.exists():
        fl = map_overlays.get_flowlines()
        fl = fl[fl["StreamOrde"] >= _NHD_MIN_ORDER][["geometry"]]
        fl["geometry"] = fl.geometry.simplify(_NHD_SIMPLIFY_TOLERANCE)
        _FLOWLINES_ASSET.write_text(fl.to_json())
    if not _WATERBODIES_ASSET.exists():
        wb = map_overlays.get_waterbodies()[["geometry"]]
        wb["geometry"] = wb.geometry.simplify(_NHD_SIMPLIFY_TOLERANCE)
        _WATERBODIES_ASSET.write_text(wb.to_json())


_ensure_nhd_assets()


TILE_URLS = {
    "street": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "humanitarian": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    "watercolor": "https://tiles.stadiamaps.com/tiles/stamen_watercolor/{z}/{x}/{y}.jpg",
}

# dl.EditControl's `draw` prop is only read once, at mount (dash-leaflet does
# not react to later changes to it), so the set of enabled draw tools is fixed
# here. Both rectangle and polygon are always available as separate buttons in
# the Leaflet draw toolbar (bottom-left of the map).
DRAW_TOOLS = {
    "rectangle": True,
    "polygon": True,
    "polyline": False,
    "circle": False,
    "circlemarker": False,
    "marker": False,
}


def _svg_img(inner):
    svg = '<svg viewBox="0 0 16 16" width="16" height="16"' ' xmlns="http://www.w3.org/2000/svg">' + inner + "</svg>"
    src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return html.Img(src=src, width=16, height=16, style={"display": "block"})


def _draw_btn(icon, btn_id, title):
    return html.Button(
        icon,
        id=btn_id,
        n_clicks=0,
        title=title,
        style={
            "background": "white",
            "border": "1px solid #bbb",
            "borderRadius": "3px",
            "padding": "5px",
            "cursor": "pointer",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "width": "30px",
            "height": "30px",
            "boxSizing": "border-box",
        },
    )


def _rect_icon():
    return _svg_img('<rect x="1" y="4" width="14" height="8"' ' fill="none" stroke="#555" stroke-width="1.5"/>')


def _poly_icon():
    return _svg_img('<path d="M8,1 L15,6 L12,14 L4,14 L1,6 Z"' ' fill="none" stroke="#555" stroke-width="1.5"/>')


def _trash_icon():
    return _svg_img(
        '<line x1="1" y1="4" x2="15" y2="4" stroke="#555" stroke-width="1.5"/>'
        '<path d="M6 4V2H10V4" fill="none" stroke="#555" stroke-width="1.5"/>'
        '<path d="M3 4L4 14H12L13 4" fill="none" stroke="#555" stroke-width="1.5"/>'
        '<line x1="6" y1="7" x2="6" y2="11" stroke="#555" stroke-width="1.5"/>'
        '<line x1="10" y1="7" x2="10" y2="11" stroke="#555" stroke-width="1.5"/>'
    )


_SECTION_LABEL = {
    "fontSize": "11px",
    "fontWeight": "bold",
    "color": "#888",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "marginBottom": "6px",
}
_SECTION_LABEL_SUMMARY = {**_SECTION_LABEL, "cursor": "pointer", "userSelect": "none"}

# Tools panel style constants — kept in sync with the toggle callback below.
_PANEL_BASE = {
    "position": "absolute",
    "top": "0",
    "right": "0",
    "height": "100%",
    "width": "26%",
    "minWidth": "240px",
    "background": "rgba(255,255,255,0.97)",
    "borderLeft": "1px solid rgba(0,0,0,0.15)",
    "boxShadow": "-3px 0 10px rgba(0,0,0,0.12)",
    "zIndex": 1200,
    "overflowY": "auto",
    "padding": "44px 12px 12px 12px",
    "boxSizing": "border-box",
}
_PANEL_OPEN = _PANEL_BASE
_PANEL_CLOSED = {**_PANEL_BASE, "display": "none"}

_GRAPH_OVERLAY_BASE = {
    "position": "absolute",
    "top": "10px",
    "left": "10px",
    "width": "38%",
    "height": "35vh",
    "background": "rgba(255,255,255,0.95)",
    "borderRadius": "6px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
    "zIndex": 700,
    "boxSizing": "border-box",
}
_GRAPH_OVERLAY_HIDDEN = {**_GRAPH_OVERLAY_BASE, "display": "none"}
_GRAPH_OVERLAY_VISIBLE = {**_GRAPH_OVERLAY_BASE, "display": "block"}


def load_iowa_geojson():
    states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip")
    return states[states["NAME"] == "Iowa"].__geo_interface__


iowa_geojson = load_iowa_geojson()
IWQIS_SITES = water.get_site_metadata()[["site_uid", "latitude", "longitude"]]


def _sites_in_polygon(geojson_geom):
    """Return site_uids whose location falls inside a GeoJSON geometry."""
    poly = shape(geojson_geom)
    return [
        row.site_uid
        for row in IWQIS_SITES.itertuples(index=False)
        if poly.contains(Point(row.longitude, row.latitude))
    ]


def make_iwqis_markers(selected_uids=None):
    """Build small clickable circle markers for the IWQIS sites.

    Each marker uses bubblingMouseEvents=False so clicking it does not also
    trigger the map's click handler. Each marker has a pattern-matching id
    so on_iwqis_marker_click can tell which site was clicked.
    """
    selected = set(selected_uids or [])
    sites = list(IWQIS_SITES[["site_uid", "latitude", "longitude"]].itertuples(index=False, name=None))
    return [
        dl.CircleMarker(
            id={"type": "iwqis-marker", "index": site_uid},
            center=[lat, lon],
            radius=7 if site_uid in selected else 5,
            color="darkred" if site_uid in selected else ("darkblue" if site_uid.startswith("USGS") else "darkgreen"),
            fillColor="red" if site_uid in selected else ("dodgerblue" if site_uid.startswith("USGS") else "limegreen"),
            fillOpacity=0.8,
            weight=1,
            bubblingMouseEvents=False,
        )
        for (site_uid, lat, lon) in sites
    ]


def layout():
    """Return the full-width map with an expandable tools panel on the right."""
    return html.Div(
        style={"position": "relative", "width": "100%"},
        children=[
            dcc.Store(id="tools-open", data=False),
            dl.Map(
                id="map",
                center=IOWA_CENTER,
                zoom=IOWA_ZOOM,
                zoomControl=False,
                children=[
                    dl.TileLayer(id="tile-layer"),
                    dl.GeoJSON(
                        data=_IOWA_MASK,
                        options={
                            "style": {
                                "fillColor": "black",
                                "fillOpacity": 0.2,
                                "weight": 0,
                                "interactive": False,
                            }
                        },
                    ),
                    dl.ZoomControl(position="bottomleft"),
                    dl.GeoJSON(
                        data=iowa_geojson,
                        options={
                            "style": {
                                "color": "steelblue",
                                "weight": 2.5,
                                "fillOpacity": 0,
                            }
                        },
                    ),
                    dl.LayerGroup(id="hydro-layer"),
                    dl.LayerGroup(id="mapunit-layer"),
                    dl.LayerGroup(id="upstream-layer"),
                    dl.LayerGroup(id="iwqis-layer"),
                    dl.LayerGroup(id="marker-layer"),
                    dl.FeatureGroup(
                        [
                            dl.EditControl(
                                id="edit-control",
                                position="bottomleft",
                                draw=DRAW_TOOLS,
                                edit={"edit": False, "remove": True},
                            )
                        ]
                    ),
                    dl.LayerGroup(id="forecast-layer"),
                ],
                style={"height": "85vh", "minHeight": "500px", "width": "100%"},
            ),
            # ── tools panel ────────────────────────────────────────────────
            html.Div(
                id="tools-panel",
                style=_PANEL_CLOSED,
                children=[
                    html.Details(
                        [
                            html.Summary("Selection", style=_SECTION_LABEL_SUMMARY),
                            html.Div(
                                style={"display": "flex", "gap": "12px", "marginTop": "8px"},
                                children=[
                                    html.Div(
                                        style={"flex": "0 0 auto"},
                                        children=[
                                            html.Div("Tool select", style={"fontSize": "11px", "fontWeight": "600", "color": "#888", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                                            dcc.RadioItems(
                                                id="selection-mode",
                                                options=[
                                                    {"label": " Point", "value": "point"},
                                                    {"label": " Area", "value": "area"},
                                                ],
                                                value="point",
                                                style={"fontSize": "13px", "marginTop": "4px"},
                                            ),
                                            html.Div(
                                                id="area-tool-container",
                                                style={"marginTop": "8px", "marginLeft": "16px", "display": "none"},
                                                children=[
                                                    html.Div(
                                                        [
                                                            _draw_btn(_rect_icon(), "draw-rect-btn", "Draw rectangle"),
                                                            _draw_btn(_poly_icon(), "draw-poly-btn", "Draw polygon"),
                                                            _draw_btn(_trash_icon(), "draw-delete-btn", "Clear drawn area"),
                                                        ],
                                                        style={"display": "flex", "gap": "4px"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"flex": "1 1 auto", "borderLeft": "1px solid #eee", "paddingLeft": "12px"},
                                        children=[
                                            html.Div("Sites Selected", style={"fontSize": "11px", "fontWeight": "600", "color": "#888", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                                            html.Div(id="sites-selected-list", style={"fontSize": "13px", "marginTop": "4px", "color": "#555"}),
                                            html.Span(
                                                "clear selection",
                                                id="clear-selection-btn",
                                                n_clicks=0,
                                                style={"display": "none"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                        open=True,
                        style={"marginBottom": "10px"},
                    ),
                    html.Hr(style={"margin": "10px 0", "borderColor": "#eee"}),
                    html.Details(
                        [
                            html.Summary("Display", style=_SECTION_LABEL_SUMMARY),
                            dcc.Checklist(
                                id="iwqis-toggle",
                                options=[{"label": " Show water sites", "value": "show"}],
                                value=["show"],
                                style={"fontSize": "13px", "marginTop": "6px"},
                            ),
                            dcc.Checklist(
                                id="graph-toggle",
                                options=[{"label": " Show nitrate graph", "value": "show"}],
                                value=["show"],
                                style={"fontSize": "13px", "marginTop": "4px"},
                            ),
                            dcc.Checklist(
                                id="hydro-toggle",
                                options=[{"label": " Show rivers & lakes", "value": "show"}],
                                value=[],
                                style={"fontSize": "13px", "marginTop": "4px"},
                            ),
                            dcc.Checklist(
                                id="upstream-toggle",
                                options=[
                                    {"label": " Show all basins", "value": "show-all"},
                                    {"label": " Show site basin on select", "value": "show-site"},
                                ],
                                value=[],
                                style={"fontSize": "13px", "marginTop": "4px"},
                            ),
                        ],
                        open=True,
                        style={"marginBottom": "10px"},
                    ),
                    html.Hr(style={"margin": "10px 0", "borderColor": "#eee"}),
                    html.Details(
                        [
                            html.Summary("Forecast", style=_SECTION_LABEL_SUMMARY),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "N surplus (kg/ha):", style={"fontSize": "12px", "marginRight": "6px"}
                                            ),
                                            dcc.Input(
                                                id="surplus-input",
                                                type="number",
                                                value=0,
                                                style={"width": "80px", "fontSize": "12px"},
                                            ),
                                        ],
                                        style={"marginBottom": "8px", "marginTop": "6px"},
                                    ),
                                    html.Button(
                                        "Run forecast", id="run-forecast-button", n_clicks=0, style={"fontSize": "12px"}
                                    ),
                                    html.Div(id="forecast-results", style={"marginTop": "8px", "fontSize": "12px"}),
                                ]
                            ),
                        ],
                        open=True,
                        style={"marginBottom": "10px"},
                    ),
                    html.Hr(style={"margin": "10px 0", "borderColor": "#eee"}),
                    html.Details(
                        [
                            html.Summary("Map Layers", style=_SECTION_LABEL_SUMMARY),
                            dcc.RadioItems(
                                id="tile-selector",
                                options=[
                                    {"label": " Street", "value": "street"},
                                    {"label": " Satellite", "value": "satellite"},
                                    {"label": " Humanitarian", "value": "humanitarian"},
                                    {"label": " Watercolor", "value": "watercolor"},
                                ],
                                value="street",
                                style={"fontSize": "13px", "marginTop": "6px"},
                            ),
                        ],
                        open=True,
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            # ── hamburger button (always on top, always visible) ────────────
            html.Button(
                "☰",
                id="hamburger-btn",
                n_clicks=0,
                style={
                    "position": "absolute",
                    "top": "10px",
                    "right": "10px",
                    "zIndex": 1500,
                    "background": "white",
                    "border": "1px solid rgba(0,0,0,0.2)",
                    "borderRadius": "4px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
                    "padding": "4px 8px",
                    "cursor": "pointer",
                    "fontSize": "18px",
                    "lineHeight": "1",
                    "color": "#333",
                },
            ),
            # ── timeseries graph overlay (upper-left, shown on site select) ──
            html.Div(
                id="map-graph-overlay",
                style=_GRAPH_OVERLAY_HIDDEN,
                children=[
                    html.Button(
                        "×",
                        id="close-graph-btn",
                        n_clicks=0,
                        style={
                            "position": "absolute",
                            "top": "4px",
                            "right": "6px",
                            "background": "none",
                            "border": "none",
                            "fontSize": "18px",
                            "lineHeight": "1",
                            "cursor": "pointer",
                            "color": "#666",
                            "padding": "2px 6px",
                            "zIndex": 1,
                        },
                    ),
                    dcc.Graph(
                        id="timeseries-graph",
                        style={"width": "100%", "height": "100%"},
                        config={"responsive": True, "scrollZoom": True},
                        figure=go.Figure(),
                    ),
                ],
            ),
        ],
    )


def register_callbacks(app):
    _CLEAR_BTN_VISIBLE = {
        "fontSize": "11px",
        "color": "#aaa",
        "cursor": "pointer",
        "textDecoration": "underline",
        "marginTop": "6px",
        "display": "block",
    }
    _CLEAR_BTN_HIDDEN = {**_CLEAR_BTN_VISIBLE, "display": "none"}

    @app.callback(
        Output("sites-selected-list", "children"),
        Output("clear-selection-btn", "style"),
        Input("selected-site", "data"),
        Input("active-graph-site", "data"),
    )
    def update_sites_selected(selected_uids, active_uid):
        selected_uids = selected_uids or []
        if not selected_uids:
            return html.Span("None", style={"color": "#aaa", "fontStyle": "italic"}), _CLEAR_BTN_HIDDEN
        items = [
            html.Div(
                [
                    html.Span(
                        uid,
                        id={"type": "graph-site-btn", "index": uid},
                        n_clicks=0,
                        style={
                            "flex": "1",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap",
                            "cursor": "pointer",
                            "fontWeight": "bold" if uid == active_uid else "normal",
                        },
                    ),
                    html.Span(
                        "×",
                        id={"type": "remove-site-btn", "index": uid},
                        n_clicks=0,
                        style={"cursor": "pointer", "color": "#bbb", "marginLeft": "6px", "fontWeight": "bold", "flexShrink": "0"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "padding": "1px 0"},
            )
            for uid in selected_uids
        ]
        return html.Div(items), _CLEAR_BTN_VISIBLE

    @app.callback(
        Output("selected-site", "data"),
        Input({"type": "iwqis-marker", "index": ALL}, "n_clicks"),
        Input({"type": "remove-site-btn", "index": ALL}, "n_clicks"),
        Input("clear-selection-btn", "n_clicks"),
        Input("edit-control", "geojson"),
        State("selected-site", "data"),
        State("selection-mode", "value"),
        prevent_initial_call=True,
    )
    def update_selected_sites(marker_clicks, remove_clicks, _clear, edit_geojson, current, mode):
        current = current or []
        triggered = ctx.triggered_id

        if triggered == "clear-selection-btn":
            return []

        if isinstance(triggered, dict) and triggered.get("type") == "iwqis-marker":
            if not any(marker_clicks):
                return no_update
            uid = triggered["index"]
            return current if uid in current else current + [uid]

        if isinstance(triggered, dict) and triggered.get("type") == "remove-site-btn":
            if not any(remove_clicks):
                return no_update
            uid = triggered["index"]
            return [s for s in current if s != uid]

        if triggered == "edit-control":
            if mode != "area" or not edit_geojson or not edit_geojson.get("features"):
                return no_update
            geom = edit_geojson["features"][-1]["geometry"]
            to_add = [s for s in _sites_in_polygon(geom) if s not in current]
            return current + to_add if to_add else no_update

        return no_update

    @app.callback(
        Output("active-graph-site", "data"),
        Input({"type": "graph-site-btn", "index": ALL}, "n_clicks"),
        Input("selected-site", "data"),
        State("active-graph-site", "data"),
        prevent_initial_call=True,
    )
    def update_active_graph_site(btn_clicks, selected_uids, current_active):
        selected_uids = selected_uids or []
        triggered = ctx.triggered_id

        if isinstance(triggered, dict) and triggered.get("type") == "graph-site-btn":
            if any(btn_clicks):
                return triggered["index"]

        # selected-site changed: switch to newly added site, or fall back to last
        if current_active not in selected_uids:
            return selected_uids[-1] if selected_uids else None
        # list grew — a new site was appended
        if selected_uids and selected_uids[-1] != current_active:
            return selected_uids[-1]
        return no_update

    @app.callback(
        Output("map-graph-overlay", "style"),
        Input("selected-site", "data"),
        Input("close-graph-btn", "n_clicks"),
        Input("graph-toggle", "value"),
        prevent_initial_call=True,
    )
    def toggle_graph_overlay(selected_uids, _, graph_toggle):
        if ctx.triggered_id == "close-graph-btn":
            return _GRAPH_OVERLAY_HIDDEN
        if "show" not in graph_toggle:
            return _GRAPH_OVERLAY_HIDDEN
        if selected_uids:
            return _GRAPH_OVERLAY_VISIBLE
        return no_update

    @app.callback(
        Output("tile-layer", "url"),
        Input("tile-selector", "value"),
    )
    def switch_tile_layer(value):
        return TILE_URLS[value]

    @app.callback(
        Output("hydro-layer", "children"),
        Input("hydro-toggle", "value"),
    )
    def render_hydro(value):
        if "show" not in value:
            return []
        return [
            dl.GeoJSON(
                url="/assets/iowa_waterbodies.geojson",
                options={
                    "style": {
                        "color": "#2563eb",
                        "weight": 1,
                        "fillColor": "#3b82f6",
                        "fillOpacity": 0.35,
                        "interactive": False,
                    }
                },
            ),
            dl.GeoJSON(
                url="/assets/iowa_flowlines.geojson",
                options={
                    "style": {
                        "color": "#2563eb",
                        "weight": 1.5,
                        "interactive": False,
                    }
                },
            ),
        ]

    _BASIN_STYLE = {
        "color": "purple",
        "weight": 2,
        "fillOpacity": 0.15,
        "fillColor": "purple",
        "interactive": False,
    }

    @app.callback(
        Output("upstream-layer", "children"),
        Input("upstream-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_upstream_area(toggle_values, selected_uids):
        layers = []
        selected_uids = selected_uids or []

        if "show-all" in toggle_values:
            try:
                gdf = water.get_all_basins_union()
                layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN_STYLE}))
            except Exception:
                pass

        if "show-site" in toggle_values:
            for uid in selected_uids:
                try:
                    gdf = water.get_basins(uid)
                    layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN_STYLE}))
                except Exception:
                    pass

        return layers

    @app.callback(
        Output("iwqis-layer", "children"),
        Input("iwqis-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_iwqis_sites(value, selected_uids):
        if "show" in value:
            return make_iwqis_markers(selected_uids or [])
        return []

    @app.callback(
        Output("area-tool-container", "style"),
        Input("selection-mode", "value"),
    )
    def update_selection_tool(mode):
        style = {"marginTop": "8px", "marginLeft": "16px"}
        if mode != "area":
            style["display"] = "none"
        return style

    @app.callback(
        Output("edit-control", "drawToolbar"),
        Output("edit-control", "editToolbar"),
        Input("draw-rect-btn", "n_clicks"),
        Input("draw-poly-btn", "n_clicks"),
        Input("draw-delete-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def trigger_draw_action(*_):
        triggered = ctx.triggered_id
        if triggered == "draw-rect-btn":
            return {"mode": "rectangle"}, no_update
        if triggered == "draw-poly-btn":
            return {"mode": "polygon"}, no_update
        if triggered == "draw-delete-btn":
            # "clear all" auto-clicks the Clear All action button after enabling
            # remove mode, immediately deleting all drawn shapes.
            return no_update, {"mode": "remove", "action": "clear all"}
        return no_update, no_update

    @app.callback(
        Output("tools-panel", "style"),
        Output("tools-open", "data"),
        Input("hamburger-btn", "n_clicks"),
        State("tools-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_tools(n_clicks, is_open):
        new_open = not is_open
        style = dict(_PANEL_OPEN)
        if not new_open:
            style["display"] = "none"
        return style, new_open

    @app.callback(
        Output("region-geom", "data"),
        Output("marker-layer", "children"),
        Input("map", "clickData"),
        Input("edit-control", "geojson"),
        State("selection-mode", "value"),
        prevent_initial_call=True,
    )
    def update_region_geom(click_data, edit_geojson, mode):
        triggered = ctx.triggered_id

        if triggered == "map":
            if mode != "point" or not click_data:
                return no_update, no_update
            latlng = click_data["latlng"]
            lat, lng = latlng["lat"], latlng["lng"]
            geom = {"type": "Point", "coordinates": [lng, lat]}
            marker = dl.Marker(
                position=[lat, lng],
                children=dl.Tooltip(f"{lat:.6f}, {lng:.6f}", permanent=True, direction="top"),
            )
            return geom, [marker]

        if triggered == "edit-control":
            if mode != "area" or not edit_geojson or not edit_geojson.get("features"):
                return no_update, no_update
            geom = edit_geojson["features"][-1]["geometry"]
            return geom, []

        return no_update, no_update
