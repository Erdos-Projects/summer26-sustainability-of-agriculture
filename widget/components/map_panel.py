"""Map panel: owns the Leaflet map and all region-of-interest selection.

Selection modes
---------------
Pin Drop : click anywhere on the map to pin a coordinate. Result is written to
           the `region-geom` store as a GeoJSON Point.
Point    : (default) select a monitoring site by clicking its marker. Map clicks
           do not drop a pin in this mode.
Area     : draw a rectangle or free-form polygon to bulk-select monitoring sites.
           Result is written to `region-geom` as a GeoJSON Polygon.

Shared state written by this module
------------------------------------
region-geom       : GeoJSON geometry of the current point or area selection.
active-graph-site : site_uid of the monitoring site whose timeseries is shown
                    in the info panel graph.  Set by clicking a site marker or
                    a row in the Sites Selected table.
selected-sites    : list of site_uids currently highlighted on the map.

Layer slots
-----------
The map contains several named `dl.LayerGroup` elements that act as render
slots.  This module owns their layout placement; other panels populate them:
  mapunit-layer  : reserved for future use
  forecast-layer : populated by forecast_panel
"""

import base64
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import dash_leaflet as dl
from dash import Input, Output, State, html, dcc, no_update, ALL, ctx
from dash_extensions.javascript import assign
from shapely.geometry import shape, Point

from data import water, map_overlays, rain as rain_data, basins, surplus, crops
from geo_utils import delineate_basin_for_pin, delineate_basin_v3_for_pin
from components import basin_editor
import colors

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

# 1×1 transparent PNG used as the placeholder url for hidden ImageOverlays
_TRANSPARENT_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
)
_IOWA_BOUNDS = [[40.3, -96.7], [43.5, -90.1]]

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
_SUBSECTION_LABEL = {
    "fontSize": "10px",
    "fontWeight": "600",
    "color": "#bbb",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "marginTop": "8px",
    "marginBottom": "3px",
    "borderBottom": "1px solid #eee",
    "paddingBottom": "2px",
}
_CHECKBOX_ROW = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "4px"}
_CHECKBOX_STYLE = {"fontSize": "13px"}
_CHECKBOX_LABEL = {"fontSize": "13px", "display": "flex", "alignItems": "center", "cursor": "pointer"}

# Permanent side panel — always visible, 20 % of viewport width
_PANEL_STYLE = {
    "flex": "0 0 32%",
    "width": "32%",
    "height": "100vh",
    "background": "rgba(255,255,255,0.97)",
    "borderLeft": "1px solid rgba(0,0,0,0.15)",
    "boxShadow": "-3px 0 8px rgba(0,0,0,0.10)",
    "overflowY": "auto",
    "padding": "0 12px 12px 12px",
    "boxSizing": "border-box",
}

# Area-select draw button styles
_DRAW_BTN_BASE = {
    "borderRadius": "3px",
    "padding": "5px",
    "cursor": "pointer",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "30px",
    "height": "30px",
    "boxSizing": "border-box",
}
_DRAW_BTN_INACTIVE = {**_DRAW_BTN_BASE, "background": "white", "border": "1px solid #bbb"}
_DRAW_BTN_ACTIVE = {**_DRAW_BTN_BASE, "background": "#dbeafe", "border": "1.5px solid #3b82f6"}

# Menu selector tab styles
_MENU_TAB_ACTIVE = {
    "flex": "1",
    "padding": "8px 0",
    "fontSize": "11px",
    "fontWeight": "700",
    "background": "#1e3a8a",
    "color": "white",
    "border": "none",
    "cursor": "pointer",
    "letterSpacing": "0.4px",
    "textTransform": "uppercase",
}
_MENU_TAB_INACTIVE = {**_MENU_TAB_ACTIVE, "background": "#f1f5f9", "color": "#64748b", "fontWeight": "600"}

_GRAPH_OVERLAY_BASE = {
    "position": "absolute",
    "top": "10px",
    "left": "10px",
    "width": "48%",
    "height": "35vh",
    "background": "rgba(255,255,255,0.95)",
    "borderRadius": "6px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
    "zIndex": 700,
    "boxSizing": "border-box",
}
_GRAPH_OVERLAY_HIDDEN = {**_GRAPH_OVERLAY_BASE, "display": "none"}
_GRAPH_OVERLAY_VISIBLE = {**_GRAPH_OVERLAY_BASE, "display": "block"}

_HELP_BTN_STYLE = {
    "position": "absolute",
    "top": "0",
    "right": "0",
    "background": "none",
    "border": "1.5px solid #aaa",
    "borderRadius": "50%",
    "width": "16px",
    "height": "16px",
    "fontSize": "10px",
    "lineHeight": "14px",
    "cursor": "pointer",
    "color": "#888",
    "padding": "0",
    "fontWeight": "bold",
    "textAlign": "center",
}
_HELP_POPUP_STYLE = {
    "position": "absolute",
    "top": "20px",
    "right": "0",
    "width": "230px",
    "background": "white",
    "border": "1px solid #ddd",
    "borderRadius": "6px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
    "padding": "10px 12px",
    "zIndex": 2000,
    "lineHeight": "1.5",
}
_HP = {"margin": "2px 0 8px 0", "color": "#555", "fontSize": "11px"}


def load_iowa_geojson():
    states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip")
    return states[states["NAME"] == "Iowa"].__geo_interface__


iowa_geojson = load_iowa_geojson()
IWQIS_SITES = water.get_metadata()[["site_uid", "latitude", "longitude"]]


def _sites_in_polygon(geojson_geom):
    """Return site_uids whose location falls inside a GeoJSON geometry."""
    poly = shape(geojson_geom)
    return [
        row.site_uid for row in IWQIS_SITES.itertuples(index=False) if poly.contains(Point(row.longitude, row.latitude))
    ]


def make_iwqis_markers(selected_uids=None, visible_uids=None):
    """Build small clickable circle markers for the IWQIS sites.

    Each marker uses bubblingMouseEvents=False so clicking it does not also
    trigger the map's click handler. Each marker has a pattern-matching id
    so on_iwqis_marker_click can tell which site was clicked.

    visible_uids : set or None
        If provided, only markers whose site_uid is in this set are rendered.
    """
    selected = set(selected_uids or [])
    sites = list(IWQIS_SITES[["site_uid", "latitude", "longitude"]].itertuples(index=False, name=None))
    if visible_uids is not None:
        sites = [(uid, lat, lon) for uid, lat, lon in sites if uid in visible_uids]
    return [
        dl.CircleMarker(
            id={"type": "iwqis-marker", "index": site_uid},
            center=[lat, lon],
            radius=7 if site_uid in selected else 5,
            color=(
                colors.SITE_SELECTED["stroke"]
                if site_uid in selected
                else (colors.SITE_USGS["stroke"] if site_uid.startswith("USGS") else colors.SITE_DEFAULT["stroke"])
            ),
            fillColor=(
                colors.SITE_SELECTED["fill"]
                if site_uid in selected
                else (colors.SITE_USGS["fill"] if site_uid.startswith("USGS") else colors.SITE_DEFAULT["fill"])
            ),
            fillOpacity=0.8,
            weight=1,
            pane="sites-pane",
            bubblingMouseEvents=False,
        )
        for (site_uid, lat, lon) in sites
    ]


def _hr():
    return html.Hr(style={"margin": "10px 0", "borderColor": "#eee"})


def _help_btn(btn_id):
    return html.Button("?", id=btn_id, n_clicks=0, style=_HELP_BTN_STYLE)


def _help_popup(popup_id, close_id, heading, body):
    return html.Div(
        id=popup_id,
        style={"display": "none"},
        children=[
            html.Div(
                [
                    html.Strong(heading, style={"fontSize": "12px"}),
                    html.Button(
                        "×",
                        id=close_id,
                        n_clicks=0,
                        style={
                            "float": "right",
                            "background": "none",
                            "border": "none",
                            "cursor": "pointer",
                            "fontSize": "15px",
                            "color": "#999",
                            "padding": "0",
                            "lineHeight": "1",
                        },
                    ),
                ],
                style={"marginBottom": "6px", "overflow": "hidden"},
            ),
            html.Hr(style={"margin": "0 0 8px 0", "borderColor": "#eee"}),
            html.Div(body),
        ],
    )


def _wrap_with_help(details_el, btn_id, close_id, popup_id, heading, body):
    return html.Div(
        style={"position": "relative", "marginBottom": "10px"},
        children=[
            details_el,
            _help_btn(btn_id),
            _help_popup(popup_id, close_id, heading, body),
        ],
    )


def _build_selection_section():
    details = html.Details(
        [
            html.Summary("Selection", style=_SECTION_LABEL_SUMMARY),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "8px"},
                children=[
                    dcc.RadioItems(
                        id="selection-mode",
                        options=[
                            {"label": " Pin Drop", "value": "pin"},
                            {"label": " Point", "value": "point"},
                            {"label": " Area", "value": "area"},
                        ],
                        value="point",
                        style={"display": "flex", "gap": "10px"},
                        labelStyle={"fontSize": "13px", "display": "flex", "alignItems": "center", "cursor": "pointer"},
                        inputStyle={"marginRight": "3px"},
                    ),
                    html.Div(
                        id="area-tool-container",
                        style={"display": "none"},
                        children=[
                            _draw_btn(_rect_icon(), "draw-rect-btn", "Draw rectangle"),
                            _draw_btn(_poly_icon(), "draw-poly-btn", "Draw polygon"),
                            _draw_btn(_trash_icon(), "draw-delete-btn", "Clear drawn area"),
                        ],
                    ),
                ],
            ),
            html.Hr(style={"margin": "8px 0", "borderColor": "#e8e8e8", "borderStyle": "solid"}),
            html.Div(
                id="sites-selected-list",
                style={"marginTop": "4px"},
            ),
            html.Span(
                "clear selection",
                id="clear-selection-btn",
                n_clicks=0,
                style={"display": "none"},
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="selection-help-btn",
        close_id="selection-help-close-btn",
        popup_id="selection-help-popup",
        heading="Selection",
        body=[
            html.Strong("Tool Select", style={"fontSize": "11px"}),
            html.P("Pin Drop -- click on the map to drop a coordinate pin.", style=_HP),
            html.P("Point -- (default) select a monitoring site by clicking its marker.", style=_HP),
            html.P("Area -- draw a rectangle or polygon to bulk select monitoring sites inside it.", style=_HP),
            html.Strong("Sites Selected", style={"fontSize": "11px"}),
            html.P("List of selected sites. Click the site name to display its timeseries.", style=_HP),
            html.P("Site -- the unique site identifier", style=_HP),
            html.P("Sparsity (%) -- % of rows with non-nan nitrate_concentration", style=_HP),
            html.P("Start -- earliest collection date", style=_HP),
            html.P("End -- last collection date", style=_HP),
            html.P("Lifespan -- difference in years between start and end", style={**_HP, "margin": "2px 0 0 0"}),
        ],
    )


def _build_graph_display_section():
    details = html.Details(
        [
            html.Summary("Graph Display", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="graph-toggle",
                options=[
                    {"label": " Show graph", "value": "show"},
                    {"label": " Display seasons", "value": "seasons"},
                ],
                value=["show"],
                style={**_CHECKBOX_STYLE, **_CHECKBOX_ROW, "marginTop": "6px"},
                labelStyle=_CHECKBOX_LABEL,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Aggregation Interval",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="aggregate-interval",
                                options=[
                                    {"label": "1D", "value": "1D"},
                                    {"label": "3D", "value": "3D"},
                                    {"label": "1W", "value": "1W"},
                                    {"label": "2W", "value": "2W"},
                                    {"label": "1MS", "value": "1MS"},
                                    {"label": "3MS", "value": "3MS"},
                                    {"label": "1YS", "value": "1YS"},
                                ],
                                value="1D",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Agg Method (Water)",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="agg-func-water",
                                options=[
                                    {"label": "sum", "value": "sum"},
                                    {"label": "mean", "value": "mean"},
                                    {"label": "max", "value": "max"},
                                    {"label": "min", "value": "min"},
                                ],
                                value="mean",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Agg Method (Rain)",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="agg-func-rain",
                                options=[
                                    {"label": "sum", "value": "sum"},
                                    {"label": "mean", "value": "mean"},
                                    {"label": "max", "value": "max"},
                                    {"label": "min", "value": "min"},
                                ],
                                value="sum",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginTop": "8px"},
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="graph-help-btn",
        close_id="graph-help-close-btn",
        popup_id="graph-help-popup",
        heading="Graph Display",
        body=[
            html.Strong("Show graph", style={"fontSize": "11px"}),
            html.P(
                "Toggles visibility of the timeseries chart overlay on the map. The chart appears when a monitoring site is selected.",
                style=_HP,
            ),
            html.Strong("Display seasons", style={"fontSize": "11px"}),
            html.P(
                "Overlays thin vertical lines at each solstice and equinox (Mar 21, Jun 21, Sep 21, Dec 21).", style=_HP
            ),
            html.Strong("Aggregate Interval", style={"fontSize": "11px"}),
            html.P(
                [
                    "Resampling period for the timeseries. See ",
                    html.A(
                        "the pandas user guide",
                        href="https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases",
                        target="_blank",
                        style={"color": "#3b82f6"},
                    ),
                    " on offset aliases.",
                ],
                style=_HP,
            ),
            html.Strong("Agg Method (Water / Rain)", style={"fontSize": "11px"}),
            html.P(
                "How values within each interval are combined — chosen separately for water (nitrate) and rain "
                "(precipitation): sum, mean, max, or min.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_forecast_section():
    details = html.Details(
        [
            html.Summary("Forecast", style=_SECTION_LABEL_SUMMARY),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("N surplus (kg/ha):", style={"fontSize": "12px", "marginRight": "6px"}),
                            dcc.Input(
                                id="surplus-input",
                                type="number",
                                value=0,
                                style={"width": "80px", "fontSize": "12px"},
                            ),
                        ],
                        style={"marginBottom": "8px", "marginTop": "6px"},
                    ),
                    html.Button("Run forecast", id="run-forecast-button", n_clicks=0, style={"fontSize": "12px"}),
                    html.Div(id="forecast-results", style={"marginTop": "8px", "fontSize": "12px"}),
                ]
            ),
        ],
        open=False,
    )
    return _wrap_with_help(
        details,
        btn_id="forecast-help-btn",
        close_id="forecast-help-close-btn",
        popup_id="forecast-help-popup",
        heading="Forecast",
        body=[
            html.Strong("N surplus (kg/ha)", style={"fontSize": "11px"}),
            html.P(
                "Estimated nitrogen surplus applied to the land surface, in kg per hectare. Represents excess N beyond crop uptake.",
                style=_HP,
            ),
            html.Strong("Run forecast", style={"fontSize": "11px"}),
            html.P(
                "Runs the nitrate concentration forecast model for the selected region using the N surplus value.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_map_display_section():
    details = html.Details(
        [
            html.Summary("Map Display Options", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="hydro-toggle",
                options=[{"label": " Show rivers & lakes", "value": "show"}],
                value=["show"],
                style={**_CHECKBOX_STYLE, "marginTop": "6px"},
                labelStyle=_CHECKBOX_LABEL,
            ),
            # ── basin display ─────────────────────────────────────────────────
            html.Div("basin display", style=_SUBSECTION_LABEL),
            html.Div(
                [
                    dcc.Checklist(
                        id="basin-preferred-toggle",
                        options=[{"label": " Show basin", "value": "show"}],
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                    dcc.Checklist(
                        id="basin-all-toggle",
                        options=[{"label": " Show all basins", "value": "show"}],
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
                style=_CHECKBOX_ROW,
            ),
            # ── rain ──────────────────────────────────────────────────────────
            html.Div("rain", style=_SUBSECTION_LABEL),
            html.Div(
                [
                    dcc.Checklist(
                        id="rain-grid-toggle",
                        options=[{"label": " Show site rain grid", "value": "show"}],
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
                style=_CHECKBOX_ROW,
            ),
            # ── nitrogen surplus ──────────────────────────────────────────────
            html.Div("nitrogen surplus", style=_SUBSECTION_LABEL),
            html.Div(
                style={"display": "flex", "gap": "10px", "marginTop": "4px"},
                children=[
                    # Left column — checkboxes
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                        children=[
                            dcc.Checklist(
                                id="surplus-heatmap-toggle",
                                options=[{"label": " Show site heatmap", "value": "show"}],
                                value=[],
                                style=_CHECKBOX_STYLE,
                                labelStyle=_CHECKBOX_LABEL,
                            ),
                            dcc.Checklist(
                                id="iowa-surplus-heatmap-toggle",
                                options=[{"label": " Show Iowa heatmap", "value": "show"}],
                                value=[],
                                style=_CHECKBOX_STYLE,
                                labelStyle=_CHECKBOX_LABEL,
                            ),
                        ],
                    ),
                    # Right column — sliders
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                children=[
                                    html.Label(
                                        "Year",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#555",
                                            "whiteSpace": "nowrap",
                                            "width": "40px",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="surplus-year-slider",
                                            min=2000,
                                            max=2017,
                                            step=1,
                                            value=2017,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            updatemode="mouseup",
                                        ),
                                        style={"flex": "1", "marginTop": "-8px", "marginBottom": "-8px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                children=[
                                    html.Label(
                                        "Opacity",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#555",
                                            "whiteSpace": "nowrap",
                                            "width": "40px",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="surplus-opacity-slider",
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=0.8,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            updatemode="mouseup",
                                        ),
                                        style={"flex": "1", "marginTop": "-8px", "marginBottom": "-8px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="map-display-help-btn",
        close_id="map-display-help-close-btn",
        popup_id="map-display-help-popup",
        heading="Map Display Options",
        body=[
            html.Strong("Show rivers & lakes", style={"fontSize": "11px"}),
            html.P("Toggles the NHD hydrography overlay (streams, rivers, and waterbodies).", style=_HP),
            html.Strong("Show basin", style={"fontSize": "11px"}),
            html.P(
                "Displays the preferred drainage basin for each selected site (purple). See Basin Editor in the Debug menu to compare individual basin types.",
                style=_HP,
            ),
            html.Strong("Show all basins", style={"fontSize": "11px"}),
            html.P("Displays the dissolved union of all monitoring site drainage basins.", style=_HP),
            html.Strong("Show site rain grid", style={"fontSize": "11px"}),
            html.P(
                "Displays the IEM precipitation grid cell positions for the active graph site.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_map_layers_section():
    details = html.Details(
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
        open=False,
    )
    return _wrap_with_help(
        details,
        btn_id="map-layers-help-btn",
        close_id="map-layers-help-close-btn",
        popup_id="map-layers-help-popup",
        heading="Map Layers",
        body=[
            html.Strong("Street", style={"fontSize": "11px"}),
            html.P("Standard OpenStreetMap tiles.", style=_HP),
            html.Strong("Satellite", style={"fontSize": "11px"}),
            html.P("ESRI World Imagery satellite tiles.", style=_HP),
            html.Strong("Humanitarian", style={"fontSize": "11px"}),
            html.P("OpenStreetMap Humanitarian style — high contrast, designed for crisis mapping.", style=_HP),
            html.Strong("Watercolor", style={"fontSize": "11px"}),
            html.P("Stadia/Stamen watercolor tiles — artistic style.", style={**_HP, "margin": "2px 0 0 0"}),
        ],
    )


def _build_debugging_section():
    return html.Details(
        [
            html.Summary("Debugging", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="iem-bbox-toggle",
                options=[{"label": " Show IEM footprint", "value": "show"}],
                value=[],
                style={"fontSize": "13px", "marginTop": "6px"},
            ),
        ],
        open=False,
    )


def layout():
    """Return the full-viewport layout: map (80 %) left, tools panel (20 %) right."""
    return html.Div(
        style={"display": "flex", "height": "100vh", "overflow": "hidden"},
        children=[
            dcc.Store(id="active-area-tool", data=None),
            dcc.Store(id="active-menu", data="explore"),
            dcc.Store(id="preferred-basin-version", data=0),
            # ── Map (80 %) ─────────────────────────────────────────────────
            html.Div(
                style={"position": "relative", "flex": "0 0 68%", "width": "68%"},
                children=[
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
                                # interactive:False so this statewide outline's transparent fill
                                # doesn't capture mouse events over all of Iowa (it would block
                                # hover on the rain-grid cells beneath it).
                                options={"style": {"color": "#555", "weight": 2, "fillOpacity": 0, "interactive": False}},
                            ),
                            dl.LayerGroup(id="iem-bbox-layer"),
                            dl.Pane(name="hydro-pane", style={"zIndex": 410}),
                            dl.Pane(name="surplus-pane", style={"zIndex": 413}),
                            dl.Pane(name="rain-grid-pane", style={"zIndex": 415}),
                            dl.Pane(name="basin-pane", style={"zIndex": 420}),
                            dl.Pane(name="sites-pane", style={"zIndex": 430}),
                            dl.LayerGroup(id="mapunit-layer"),
                            dl.LayerGroup(id="hydro-layer"),
                            dl.ImageOverlay(
                                id="surplus-image-overlay",
                                url=_TRANSPARENT_PNG,
                                bounds=_IOWA_BOUNDS,
                                opacity=0,
                                pane="surplus-pane",
                            ),
                            dl.ImageOverlay(
                                id="iowa-surplus-image-overlay",
                                url=_TRANSPARENT_PNG,
                                bounds=_IOWA_BOUNDS,
                                opacity=0,
                                pane="surplus-pane",
                            ),
                            dl.LayerGroup(id="rain-grid-layer"),
                            dl.LayerGroup(id="upstream-layer"),
                            dl.LayerGroup(id="basin1-layer"),
                            dl.LayerGroup(id="basin2-layer"),
                            dl.LayerGroup(id="basin3-layer"),
                            dl.LayerGroup(id="pin-basin-layer"),
                            dl.LayerGroup(id="pin-basin-v3-layer"),
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
                        style={"height": "100vh", "width": "100%"},
                    ),
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
            ),
            # ── Tools panel (20 %) ─────────────────────────────────────────
            html.Div(
                id="tools-panel",
                style=_PANEL_STYLE,
                children=[
                    # Menu selector tabs (full-bleed, no side padding inherited)
                    html.Div(
                        style={"display": "flex", "margin": "0 -12px 12px -12px"},
                        children=[
                            html.Button("Explore", id="menu-tab-explore", n_clicks=0, style=_MENU_TAB_ACTIVE),
                            html.Button("Forecast", id="menu-tab-forecast", n_clicks=0, style=_MENU_TAB_INACTIVE),
                            html.Button("Debug", id="menu-tab-debug", n_clicks=0, style=_MENU_TAB_INACTIVE),
                        ],
                    ),
                    # Explore Menu
                    html.Div(
                        id="explore-menu-content",
                        style={"display": "block"},
                        children=[
                            _build_selection_section(),
                            _hr(),
                            _build_graph_display_section(),
                            _hr(),
                            _build_map_display_section(),
                            _hr(),
                            _build_map_layers_section(),
                        ],
                    ),
                    # Forecast Menu
                    html.Div(
                        id="forecast-menu-content",
                        style={"display": "none"},
                        children=[
                            _build_forecast_section(),
                        ],
                    ),
                    # Debug Menu
                    html.Div(
                        id="debug-menu-content",
                        style={"display": "none"},
                        children=[
                            html.Div(id="debug-sites-table", style={"marginBottom": "8px"}),
                            _hr(),
                            basin_editor.layout(),
                            _hr(),
                            _build_debugging_section(),
                        ],
                    ),
                ],
            ),
        ],
    )


# ── Rain-grid Voronoi cells (rendered as one interactive GeoJSON layer) ────────
# Cells are coloured by surplus (matching the surplus heatmap PNGs), highlight on
# hover, and bind a tooltip with surplus / total-N / crop stats. The style,
# hover-style and tooltip-binding run client-side as JS (one layer scales far
# better than thousands of per-cell components).
_GRID_NODATA_COLOR = "#cccccc"  # cells with no surplus value for the year (e.g. outside Iowa)

_GRID_STYLE_JS = assign(
    """function(feature, context){
        return {color: '#0284c7', weight: 1, fillColor: feature.properties.color, fillOpacity: 0.55};
    }"""
)
_GRID_HOVER_JS = assign(
    """function(feature, context){
        return {weight: 3, color: '#0c4a6e', fillOpacity: 0.8};
    }"""
)
_GRID_ONEACH_JS = assign(
    """function(feature, layer, context){
        if(feature.properties && feature.properties.tooltip){
            // Click opens a popup — reliable even when Leaflet's vector hover
            // hit-testing goes stale after a map pan/zoom (clicks are hit-tested
            // by coordinate, so they always reach the cell).
            layer.bindPopup(feature.properties.tooltip);
            // Hover tooltip too, for the (intermittent) cases where mouseover fires.
            layer.bindTooltip(feature.properties.tooltip, {sticky: true});
            layer.on('mouseover', function(e){ layer.openTooltip(e.latlng); });
            layer.on('mousemove', function(e){ layer.openTooltip(e.latlng); });
            layer.on('mouseout', function(){ layer.closeTooltip(); });
        }
    }"""
)


def _cell_tooltip(row, crop_cols):
    """HTML tooltip string for one Voronoi cell: surplus + total N + crop list."""
    lines = [f"<b>Cell {int(row['node_id'])}</b>"]
    if pd.notna(row.get("surplus_kgha")):
        lines.append(f"Surplus: {row['surplus_kgha']:.0f} kg/ha")
        lines.append(f"Total N: {row['total_kg_N']:,.0f} kg")
    else:
        lines.append("Surplus: n/a (outside data)")
    counts = {c: row[c] for c in crop_cols if pd.notna(row.get(c)) and row[c] > 0}
    total = sum(counts.values())
    if total:
        lines.append("Crops:")
        for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"&nbsp;&nbsp;{k}: {v / total:.0%}")
    return "<br>".join(lines)


def _rain_grid_features(uid, year):
    """GeoJSON (lon/lat) of a site's rain cells, each with a surplus colour and a
    tooltip joining surplus + crop stats for `year`. Raises FileNotFoundError if
    the rain grid hasn't been built."""
    cells = rain_data.get_rain_grid(uid).to_crs("EPSG:4326")[["node_id", "geometry"]]

    try:
        s = surplus.get_surplus_grid(uid)
        s = s[s["year"] == year][["node_id", "surplus_kgha", "total_kg_N", "coverage_frac"]]
    except FileNotFoundError:
        s = pd.DataFrame(columns=["node_id", "surplus_kgha", "total_kg_N", "coverage_frac"])

    try:
        c = crops.get_crops(uid)
        c = c[c["year"] == year].drop(columns=["year"])
    except FileNotFoundError:
        c = pd.DataFrame(columns=["node_id"])
    crop_cols = [col for col in c.columns if col != "node_id"]

    cells = cells.merge(s, on="node_id", how="left").merge(c, on="node_id", how="left")
    cells["color"] = cells["surplus_kgha"].map(
        lambda v: surplus.surplus_to_hex(v) if pd.notna(v) else _GRID_NODATA_COLOR
    )
    cells["tooltip"] = cells.apply(lambda r: _cell_tooltip(r, crop_cols), axis=1)
    return json.loads(cells[["node_id", "color", "tooltip", "geometry"]].to_json())


def _rain_grid_dots(uid):
    """Fallback: rain-cell centroids as dots, for sites without a rain grid yet."""
    try:
        df = rain_data.get_rain(uid)
    except FileNotFoundError:
        return []
    cells = df[["lon", "lat"]].drop_duplicates()
    return [
        dl.CircleMarker(
            center=[row.lat, row.lon],
            radius=4,
            color=colors.RAIN_GRID["stroke"],
            fillColor=colors.RAIN_GRID["fill"],
            fillOpacity=0.5,
            weight=1,
            pane="rain-grid-pane",
            bubblingMouseEvents=False,
        )
        for row in cells.itertuples()
    ]


def register_callbacks(app):
    _HIDDEN = {**_HELP_POPUP_STYLE, "display": "none"}

    for _popup_id, _btn_id, _close_id in [
        ("selection-help-popup", "selection-help-btn", "selection-help-close-btn"),
        ("graph-help-popup", "graph-help-btn", "graph-help-close-btn"),
        ("forecast-help-popup", "forecast-help-btn", "forecast-help-close-btn"),
        ("map-display-help-popup", "map-display-help-btn", "map-display-help-close-btn"),
        ("map-layers-help-popup", "map-layers-help-btn", "map-layers-help-close-btn"),
    ]:

        def _make_cb(close_id=_close_id, hidden=_HIDDEN):
            @app.callback(
                Output(_popup_id, "style"),
                Input(_btn_id, "n_clicks"),
                Input(_close_id, "n_clicks"),
                prevent_initial_call=True,
            )
            def _toggle(_, __, _cid=close_id, _h=hidden):
                return _h if ctx.triggered_id == _cid else _HELP_POPUP_STYLE

        _make_cb()

    _CLEAR_BTN_VISIBLE = {
        "fontSize": "11px",
        "color": "#aaa",
        "cursor": "pointer",
        "textDecoration": "underline",
        "marginTop": "6px",
        "display": "block",
    }
    _CLEAR_BTN_HIDDEN = {**_CLEAR_BTN_VISIBLE, "display": "none"}

    _TH = {
        "fontSize": "11px",
        "fontWeight": "600",
        "color": "#888",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "padding": "2px 4px 5px 4px",
        "borderBottom": "2px dashed #ddd",
        "whiteSpace": "nowrap",
    }
    _TD = {"fontSize": "12px", "padding": "3px 4px", "verticalAlign": "middle"}

    @app.callback(
        Output("sites-selected-list", "children"),
        Output("clear-selection-btn", "style"),
        Output("debug-sites-table", "children"),
        Input("selected-site", "data"),
        Input("active-graph-site", "data"),
    )
    def update_sites_selected(selected_uids, active_uid):
        selected_uids = selected_uids or []

        header = html.Thead(
            html.Tr(
                [
                    html.Th("Site", style={**_TH, "textAlign": "left", "paddingLeft": "0"}),
                    html.Th("Sparsity (%)", style={**_TH, "textAlign": "center"}),
                    html.Th("Start", style={**_TH, "textAlign": "center"}),
                    html.Th("End", style={**_TH, "textAlign": "center"}),
                    html.Th("Lifespan", style={**_TH, "textAlign": "center"}),
                    html.Th("", style={**_TH, "borderBottom": "1px solid #ddd"}),
                ]
            )
        )

        if not selected_uids:
            empty = html.Table([header], style={"width": "100%", "borderCollapse": "collapse"})
            return empty, _CLEAR_BTN_HIDDEN, empty

        rows = []
        for uid in selected_uids:
            try:
                site_stats = water.get_stats(uid)
                r = site_stats.iloc[0]
                sparsity = f"{r['nitrate_sparsity'] * 100:.1f}"
                start = str(r["start_date"])[:7]
                end = str(r["last_date"])[:7]
                lifespan = f"{r['lifespan']:.2f}"
            except Exception:
                sparsity = start = end = lifespan = "—"

            rows.append(
                html.Tr(
                    [
                        html.Td(
                            html.Span(
                                uid,
                                id={"type": "graph-site-btn", "index": uid},
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "fontWeight": "bold" if uid == active_uid else "normal",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                    "display": "block",
                                    "maxWidth": "90px",
                                },
                            ),
                            style={**_TD, "textAlign": "left", "paddingLeft": "0"},
                        ),
                        html.Td(sparsity, style={**_TD, "textAlign": "center"}),
                        html.Td(start, style={**_TD, "textAlign": "center"}),
                        html.Td(end, style={**_TD, "textAlign": "center"}),
                        html.Td(lifespan, style={**_TD, "textAlign": "center"}),
                        html.Td(
                            html.Span(
                                "×",
                                id={"type": "remove-site-btn", "index": uid},
                                n_clicks=0,
                                style={"cursor": "pointer", "color": "#bbb", "fontWeight": "bold", "fontSize": "14px"},
                            ),
                            style={**_TD, "textAlign": "right", "paddingRight": "0"},
                        ),
                    ]
                )
            )

        table = html.Table([header, html.Tbody(rows)], style={"width": "100%", "borderCollapse": "collapse"})
        return table, _CLEAR_BTN_VISIBLE, table

    @app.callback(
        Output("selected-site", "data"),
        Input({"type": "iwqis-marker", "index": ALL}, "n_clicks"),
        Input({"type": "remove-site-btn", "index": ALL}, "n_clicks"),
        Input("clear-selection-btn", "n_clicks"),
        Input("edit-control", "geojson"),
        State("selected-site", "data"),
        State("selection-mode", "value"),
        State("active-menu", "data"),
        prevent_initial_call=True,
    )
    def update_selected_sites(marker_clicks, remove_clicks, _clear, edit_geojson, current, mode, active_menu):
        current = current or []
        triggered = ctx.triggered_id

        if triggered == "clear-selection-btn":
            return []

        if isinstance(triggered, dict) and triggered.get("type") == "iwqis-marker":
            if not any(marker_clicks):
                return no_update
            uid = triggered["index"]
            if active_menu == "debug":
                # Single selection: clicking the selected site deselects; clicking another replaces
                return [] if uid in current else [uid]
            return [s for s in current if s != uid] if uid in current else current + [uid]

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
        Output("iem-bbox-layer", "children"),
        Input("iem-bbox-toggle", "value"),
    )
    def render_iem_bbox(value):
        if "show" not in value:
            return []
        return [
            dl.Rectangle(
                bounds=[[38.8, -97.7], [45.3, -87.4]],
                pathOptions={"color": colors.IEM_BBOX["stroke"], "weight": 2, "dashArray": "6 4", "fillOpacity": 0},
            )
        ]

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
                    "pane": "hydro-pane",
                    "style": {
                        "color": colors.HYDRO["stroke"],
                        "weight": 0.8,
                        "fillColor": colors.HYDRO["fill"],
                        "fillOpacity": 0.45,
                        "interactive": False,
                    },
                },
            ),
            dl.GeoJSON(
                url="/assets/iowa_flowlines.geojson",
                options={
                    "pane": "hydro-pane",
                    "style": {
                        "color": colors.HYDRO["stroke"],
                        "weight": 1.2,
                        "interactive": False,
                    },
                },
            ),
        ]

    _BASIN_STYLE = {
        "pane": "basin-pane",
        "color": colors.BASIN["stroke"],
        "weight": 2,
        "fillOpacity": 0.15,
        "fillColor": colors.BASIN["fill"],
        "interactive": False,
    }

    @app.callback(
        Output("upstream-layer", "children"),
        Input("basin-preferred-toggle", "value"),
        Input("basin-all-toggle", "value"),
        Input("selected-site", "data"),
        Input("preferred-basin-version", "data"),
    )
    def render_upstream_area(preferred_toggle, all_toggle, selected_uids, _version):
        layers = []
        selected_uids = selected_uids or []

        if "show" in all_toggle:
            try:
                gdf = basins.get_all_basins_union()
                layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN_STYLE}))
            except Exception:
                pass

        if "show" in preferred_toggle:
            for uid in selected_uids:
                try:
                    gdf = basins.get_basin(uid)
                    layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN_STYLE}))
                except Exception:
                    pass

        return layers

    _BASIN2_STYLE = {
        "pane": "basin-pane",
        "color": colors.BASIN_V2["stroke"],
        "weight": 2,
        "fillOpacity": 0.12,
        "fillColor": colors.BASIN_V2["fill"],
        "interactive": False,
    }

    @app.callback(
        Output("basin1-layer", "children"),
        Input("basin1-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_basin1(toggle, selected_uids):
        if "show" not in toggle:
            return []
        layers = []
        for uid in selected_uids or []:
            try:
                gdf = basins.get_basin(uid, type=1)
                layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN_STYLE}))
            except FileNotFoundError:
                pass
        return layers

    @app.callback(
        Output("basin2-layer", "children"),
        Input("basin2-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_basin2(toggle, selected_uids):
        if "show" not in toggle:
            return []
        layers = []
        for uid in selected_uids or []:
            try:
                gdf = basins.get_basin(uid, type=2)
                layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN2_STYLE}))
            except FileNotFoundError:
                pass
        return layers

    _BASIN3_STYLE = {
        "pane": "basin-pane",
        "color": colors.BASIN_V3["stroke"],
        "weight": 2,
        "fillOpacity": 0.12,
        "fillColor": colors.BASIN_V3["fill"],
        "interactive": False,
    }

    @app.callback(
        Output("basin3-layer", "children"),
        Input("basin3-toggle", "value"),
        Input("selected-site", "data"),
    )
    def render_basin3(toggle, selected_uids):
        if "show" not in toggle:
            return []
        layers = []
        for uid in selected_uids or []:
            try:
                gdf = basins.get_basin(uid, type=3)
                layers.append(dl.GeoJSON(data=json.loads(gdf.to_json()), options={"style": _BASIN3_STYLE}))
            except FileNotFoundError:
                pass
        return layers

    _PIN_BASIN_STYLE = {
        "pane": "basin-pane",
        "color": colors.PIN_BASIN_V1["stroke"],
        "weight": 2,
        "fillOpacity": 0.15,
        "fillColor": colors.PIN_BASIN_V1["fill"],
        "interactive": False,
    }

    _PIN_BASIN_V3_STYLE = {
        "pane": "basin-pane",
        "color": colors.PIN_BASIN_V3["stroke"],
        "weight": 2,
        "fillOpacity": 0.15,
        "fillColor": colors.PIN_BASIN_V3["fill"],
        "interactive": False,
    }

    @app.callback(
        Output("pin-basin-layer", "children"),
        Input("pin-basin-v1-toggle", "value"),
        Input("region-geom", "data"),
    )
    def render_pin_basin_v1(toggle, region_geom):
        if "show" not in toggle:
            return []
        if not region_geom or region_geom.get("type") != "Point":
            return []
        lng, lat = region_geom["coordinates"]
        try:
            geojson = delineate_basin_for_pin(lat, lng)
            return [dl.GeoJSON(data=geojson, options={"style": _PIN_BASIN_STYLE})]
        except Exception:
            return []

    @app.callback(
        Output("pin-basin-v3-layer", "children"),
        Input("pin-basin-v3-toggle", "value"),
        Input("region-geom", "data"),
    )
    def render_pin_basin_v3(toggle, region_geom):
        if "show" not in toggle:
            return []
        if not region_geom or region_geom.get("type") != "Point":
            return []
        lng, lat = region_geom["coordinates"]
        try:
            geojson = delineate_basin_v3_for_pin(lat, lng)
            return [dl.GeoJSON(data=geojson, options={"style": _PIN_BASIN_V3_STYLE})]
        except Exception:
            return []

    @app.callback(
        Output("rain-grid-layer", "children"),
        Input("rain-grid-toggle", "value"),
        Input("active-graph-site", "data"),
        Input("surplus-year-slider", "value"),
    )
    def render_rain_grid(toggle, active_uid, year):
        if "show" not in toggle or not active_uid:
            return []
        try:
            features = _rain_grid_features(active_uid, year)
        except FileNotFoundError:
            return _rain_grid_dots(active_uid)  # no rain grid built yet → centroid dots
        return [
            dl.GeoJSON(
                data=features,
                style=_GRID_STYLE_JS,
                hoverStyle=_GRID_HOVER_JS,
                onEachFeature=_GRID_ONEACH_JS,
                zoomToBounds=False,
                pane="rain-grid-pane",
            )
        ]

    @app.callback(
        Output("surplus-image-overlay", "url"),
        Output("surplus-image-overlay", "bounds"),
        Output("surplus-image-overlay", "opacity"),
        Input("surplus-heatmap-toggle", "value"),
        Input("active-graph-site", "data"),
        Input("selected-site", "data"),
        Input("surplus-year-slider", "value"),
        Input("surplus-opacity-slider", "value"),
    )
    def render_surplus_heatmap(toggle, active_uid, selected_uids, year, opacity):
        uid = active_uid or (selected_uids[0] if selected_uids else None)
        if "show" not in toggle or not uid:
            return _TRANSPARENT_PNG, _IOWA_BOUNDS, 0
        try:
            url, bounds = surplus.get_surplus_image_buffer(uid, year)
        except Exception as e:
            print(f"[surplus heatmap] {type(e).__name__}: {e}")
            return _TRANSPARENT_PNG, _IOWA_BOUNDS, 0
        return url, bounds, opacity

    @app.callback(
        Output("iowa-surplus-image-overlay", "url"),
        Output("iowa-surplus-image-overlay", "bounds"),
        Output("iowa-surplus-image-overlay", "opacity"),
        Input("iowa-surplus-heatmap-toggle", "value"),
        Input("surplus-year-slider", "value"),
        Input("surplus-opacity-slider", "value"),
    )
    def render_iowa_surplus_heatmap(toggle, year, opacity):
        if "show" not in toggle:
            return _TRANSPARENT_PNG, _IOWA_BOUNDS, 0
        try:
            url, bounds = surplus.get_iowa_surplus_image_buffer(year)
        except Exception as e:
            print(f"[iowa surplus heatmap] {type(e).__name__}: {e}")
            return _TRANSPARENT_PNG, _IOWA_BOUNDS, 0
        return url, bounds, opacity

    @app.callback(
        Output("iwqis-layer", "children"),
        Input("selected-site", "data"),
        Input("basin-review-flagged-only", "value"),
        Input("basin-review-unreviewed-only", "value"),
    )
    def render_iwqis_sites(selected_uids, flagged_only, unreviewed_only):
        visible_uids = None
        if "on" in (flagged_only or []) or "on" in (unreviewed_only or []):
            try:
                meta = basins.get_metadata()
                df = meta.copy()
                flag_cols = ["flag_area", "flag_river", "flag_not_contained", "flag_basin1_over_basin2"]
                if "on" in (flagged_only or []):
                    flag_df = df[flag_cols].fillna(False)
                    df = df[flag_df.any(axis=1)]
                if "on" in (unreviewed_only or []):
                    df = df[~df["reviewed"].fillna(False).astype(bool)]
                visible_uids = set(df["site_uid"])
            except FileNotFoundError:
                pass
        return make_iwqis_markers(selected_uids or [], visible_uids=visible_uids)

    @app.callback(
        Output("area-tool-container", "style"),
        Input("selection-mode", "value"),
    )
    def update_selection_tool(mode):
        if mode == "area":
            return {"display": "flex", "gap": "4px", "alignItems": "center"}
        return {"display": "none"}

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
            if mode != "pin" or not click_data:
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

    @app.callback(
        Output("explore-menu-content", "style"),
        Output("forecast-menu-content", "style"),
        Output("debug-menu-content", "style"),
        Output("menu-tab-explore", "style"),
        Output("menu-tab-forecast", "style"),
        Output("menu-tab-debug", "style"),
        Output("active-menu", "data"),
        Output("selected-site", "data", allow_duplicate=True),
        Input("menu-tab-explore", "n_clicks"),
        Input("menu-tab-forecast", "n_clicks"),
        Input("menu-tab-debug", "n_clicks"),
        State("selected-site", "data"),
        State("active-graph-site", "data"),
        prevent_initial_call=True,
    )
    def switch_menu(_, __, ___, selected_sites, active_site):
        _show = {"display": "block"}
        _hide = {"display": "none"}
        if ctx.triggered_id == "menu-tab-debug":
            trimmed = [active_site] if active_site else (selected_sites[:1] if selected_sites else [])
            return _hide, _hide, _show, _MENU_TAB_INACTIVE, _MENU_TAB_INACTIVE, _MENU_TAB_ACTIVE, "debug", trimmed
        if ctx.triggered_id == "menu-tab-forecast":
            return _hide, _show, _hide, _MENU_TAB_INACTIVE, _MENU_TAB_ACTIVE, _MENU_TAB_INACTIVE, "forecast", no_update
        return _show, _hide, _hide, _MENU_TAB_ACTIVE, _MENU_TAB_INACTIVE, _MENU_TAB_INACTIVE, "explore", no_update

    @app.callback(
        Output("active-area-tool", "data"),
        Input("draw-rect-btn", "n_clicks"),
        Input("draw-poly-btn", "n_clicks"),
        Input("selection-mode", "value"),
        prevent_initial_call=True,
    )
    def update_active_area_tool(_rect, _poly, mode):
        if ctx.triggered_id == "selection-mode":
            return "rect" if mode == "area" else None
        if ctx.triggered_id == "draw-rect-btn":
            return "rect"
        if ctx.triggered_id == "draw-poly-btn":
            return "poly"
        return no_update

    @app.callback(
        Output("draw-rect-btn", "style"),
        Output("draw-poly-btn", "style"),
        Input("active-area-tool", "data"),
    )
    def update_area_btn_styles(active_tool):
        return (
            _DRAW_BTN_ACTIVE if active_tool == "rect" else _DRAW_BTN_INACTIVE,
            _DRAW_BTN_ACTIVE if active_tool == "poly" else _DRAW_BTN_INACTIVE,
        )

    basin_editor.register_callbacks(app)
