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
import dash_leaflet as dl
from dash import Input, Output, State, html, dcc, no_update, ALL, ctx

from data import iwqis_utils, basins

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

# ── NHD hydrography paths ─────────────────────────────────────────────────────
# Point these at the files produced by data/download_nhd.py.
_NHD_DIR = Path(__file__).parents[3] / "data" / "map-overlays" / "data"
_NHD_FLOWLINES = _NHD_DIR / "iowa_flowlines.parquet"
_NHD_WATERBODIES = _NHD_DIR / "iowa_waterbodies.parquet"
# Minimum Strahler stream order to display (raise to reduce detail, lower to add it).
_NHD_MIN_ORDER = 5
# ──────────────────────────────────────────────────────────────────────────────

_nhd_cache: dict = {}


def _load_nhd() -> dict:
    """Load and cache NHD GeoJSON on first call; instant thereafter."""
    if not _nhd_cache:
        fl = gpd.read_parquet(_NHD_FLOWLINES, columns=["geometry", "StreamOrde"])
        fl = fl[fl["StreamOrde"] >= _NHD_MIN_ORDER][["geometry"]]
        _nhd_cache["flowlines"] = json.loads(fl.to_json())

        wb = gpd.read_parquet(_NHD_WATERBODIES, columns=["geometry"])
        _nhd_cache["waterbodies"] = json.loads(wb.to_json())
    return _nhd_cache


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
    svg = (
        '<svg viewBox="0 0 16 16" width="16" height="16"'
        ' xmlns="http://www.w3.org/2000/svg">' + inner + '</svg>'
    )
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
    return _svg_img(
        '<rect x="1" y="4" width="14" height="8"'
        ' fill="none" stroke="#555" stroke-width="1.5"/>'
    )


def _poly_icon():
    return _svg_img(
        '<path d="M8,1 L15,6 L12,14 L4,14 L1,6 Z"'
        ' fill="none" stroke="#555" stroke-width="1.5"/>'
    )


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

# Tools panel style constants — kept in sync with the toggle callback below.
_PANEL_BASE = {
    "position": "absolute",
    "top": "0",
    "right": "0",
    "height": "100%",
    "width": "20%",
    "minWidth": "180px",
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


def load_iowa_geojson():
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    )
    return states[states["NAME"] == "Iowa"].__geo_interface__


iowa_geojson = load_iowa_geojson()
IWQIS_SITES = iwqis_utils.get_all_site_locations()

# Hardcoded upstream basin for demonstration; makes two USGS NLDI calls at startup.
_UPSTREAM_LON, _UPSTREAM_LAT = -93.65, 42.03
_UPSTREAM_GEOJSON = json.loads(
    basins.delineate_basin(_UPSTREAM_LAT, _UPSTREAM_LON).to_json()
)


def make_iwqis_markers(selected_uid=None):
    """Build small clickable circle markers for the IWQIS sites.

    Each marker uses bubblingMouseEvents=False so clicking it does not also
    trigger the map's click handler. Each marker has a pattern-matching id
    so on_iwqis_marker_click can tell which site was clicked.
    """
    sites = list(IWQIS_SITES[["uid", "latitude", "longitude"]].itertuples(index=False, name=None))
    return [
        dl.CircleMarker(
            id={"type": "iwqis-marker", "index": uid},
            center=[lat, lon],
            radius=7 if uid == selected_uid else 5,
            color="darkred" if uid == selected_uid else ("darkblue" if uid.startswith("USGS") else "darkgreen"),
            fillColor="red" if uid == selected_uid else ("dodgerblue" if uid.startswith("USGS") else "limegreen"),
            fillOpacity=0.8,
            weight=1,
            bubblingMouseEvents=False,
        )
        for (uid, lat, lon) in sites
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
                children=[
                    dl.TileLayer(id="tile-layer"),
                    dl.GeoJSON(
                        data=iowa_geojson,
                        options={
                            "style": {
                                "color": "steelblue",
                                "weight": 2.5,
                                "fillOpacity": 0.04,
                                "fillColor": "steelblue",
                            }
                        },
                    ),
                    dl.LayerGroup(id="hydro-layer"),
                    dl.LayerGroup(id="mapunit-layer"),
                    dl.LayerGroup(id="upstream-layer"),
                    dl.LayerGroup(id="iwqis-layer"),
                    dl.LayerGroup(id="marker-layer"),
                    dl.FeatureGroup([
                        dl.EditControl(
                            id="edit-control",
                            position="bottomleft",
                            draw=DRAW_TOOLS,
                            edit={"edit": False, "remove": True},
                        )
                    ]),
                    dl.LayerGroup(id="forecast-layer"),
                ],
                style={"height": "70vh", "minHeight": "500px", "width": "100%"},
            ),
            # ── tools panel ────────────────────────────────────────────────
            html.Div(
                id="tools-panel",
                style=_PANEL_CLOSED,
                children=[
                    html.Div("Tools", style=_SECTION_LABEL),
                    dcc.RadioItems(
                        id="selection-mode",
                        options=[
                            {"label": " Point", "value": "point"},
                            {"label": " Area", "value": "area"},
                        ],
                        value="point",
                        style={"fontSize": "13px"},
                    ),
                    # Area draw tools; toggled visible by update_selection_tool.
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
                    html.Hr(style={"margin": "14px 0", "borderColor": "#eee"}),
                    html.Div("Display", style=_SECTION_LABEL),
                    dcc.Checklist(
                        id="iwqis-toggle",
                        options=[{"label": " Show water sites", "value": "show"}],
                        value=[],
                        style={"fontSize": "13px"},
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
                            {"label": " Show site basin", "value": "show-site"},
                            {"label": " Show all site basins", "value": "show-all"},
                        ],
                        value=[],
                        style={"fontSize": "13px", "marginTop": "4px"},
                    ),
                    html.Hr(style={"margin": "14px 0", "borderColor": "#eee"}),
                    html.Div("Map Layers", style=_SECTION_LABEL),
                    dcc.RadioItems(
                        id="tile-selector",
                        options=[
                            {"label": " Street", "value": "street"},
                            {"label": " Satellite", "value": "satellite"},
                            {"label": " Humanitarian", "value": "humanitarian"},
                            {"label": " Watercolor", "value": "watercolor"},
                        ],
                        value="street",
                        style={"fontSize": "13px"},
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
        ],
    )


def register_callbacks(app):
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
        nhd = _load_nhd()
        return [
            dl.GeoJSON(
                data=nhd["waterbodies"],
                options={"style": {
                    "color": "#2563eb",
                    "weight": 1,
                    "fillColor": "#3b82f6",
                    "fillOpacity": 0.35,
                    "interactive": False,
                }},
            ),
            dl.GeoJSON(
                data=nhd["flowlines"],
                options={"style": {
                    "color": "#2563eb",
                    "weight": 1.5,
                    "interactive": False,
                }},
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
    def render_upstream_area(toggle_values, selected_uid):
        layers = []

        if "show-site" in toggle_values and selected_uid:
            try:
                gdf = basins.get_basin_from_file(selected_uid)
                layers.append(dl.GeoJSON(
                    data=json.loads(gdf.to_json()),
                    options={"style": _BASIN_STYLE},
                ))
            except Exception:
                pass

        if "show-all" in toggle_values:
            layers.append(dl.GeoJSON(
                data=_UPSTREAM_GEOJSON,
                options={"style": _BASIN_STYLE},
            ))

        return layers

    @app.callback(
        Output("iwqis-layer", "children"),
        Input("iwqis-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_iwqis_sites(value, selected_uid):
        if "show" in value:
            return make_iwqis_markers(selected_uid)
        return []

    @app.callback(
        Output("selected-site", "data"),
        Input({"type": "iwqis-marker", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_iwqis_marker_click(n_clicks_list):
        if not any(n_clicks_list):
            return no_update
        triggered = ctx.triggered_id
        if not triggered:
            return no_update
        return triggered["index"]

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
