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

import functools
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import dash_leaflet as dl
# Re-exported through `from map_common import *` (see __all__): map_panel and map_layout use these without importing dash themselves. no_update and ctx left with the last server callback.
from dash import ClientsideFunction, Input, Output, State, html, dcc, ALL
from dash_extensions.javascript import assign

from src.data import access, surplus_viz
from geo_utils import delineate_basin_for_pin, delineate_basin_v3_for_pin
from components import basin_editor
import bundle
import colors

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

# (_TRANSPARENT_PNG / _IOWA_BOUNDS were the placeholder url + extent for the statewide surplus
# ImageOverlay, retired with the heatmap.)

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

# The NHD filter/tolerance that produced the display GeoJSONs now live in
# widget/static/build_bundle.py (NHD_MIN_ORDER / NHD_SIMPLIFY_DEG); this module only consumes them.
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_FLOWLINES_ASSET = _ASSETS_DIR / "iowa_flowlines.geojson"
_WATERBODIES_ASSET = _ASSETS_DIR / "iowa_waterbodies.geojson"


def _require_nhd_assets():
    """Fail fast if the NHD display GeoJSONs are missing.

    These used to be BUILT here at import time, which meant importing the app could read 58 MB of
    parquet (and, with the Census fetch below, hit the network). Generation moved to
    widget/static/build_bundle.py so that importing the app is pure -- a hard requirement for the
    dash2html snapshot, which imports the module to render the layout.
    """
    missing = [p.name for p in (_FLOWLINES_ASSET, _WATERBODIES_ASSET) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing widget assets: {', '.join(missing)}.\n"
            "Run `python -m widget.static.build_bundle --only hydro` to generate them."
        )


_require_nhd_assets()


# Watercolor (Stadia/Stamen) is deliberately absent: it requires an API key for production traffic,
# so it would break on a public static build. The remaining three are keyless.
# ATTRIBUTION. OpenStreetMap tiles are ODbL and the OSMF tile usage policy requires a visible
# "© OpenStreetMap contributors"; Esri World Imagery requires its own credit. The attribution box is
# currently hidden in assets/custom.css, which is a deliberate choice rather than an oversight -- to
# comply instead, delete that rule and pass `attribution=` on the dl.TileLayer, e.g.
#     '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
# and hide only Leaflet's own prefix (`.leaflet-control-attribution a[href*="leafletjs"]`), which is
# a courtesy credit and not a licence condition.
TILE_URLS = {
    "street": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "humanitarian": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
}



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
    # Clears the docs button, which sits at left:12px in the same corner (assets/docs.css).
    "left": "64px",
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

# Sites Selected table. These used to live inside map_panel.register_callbacks, which stopped working when the table went clientside: the JS builds the rows, so the styles have to reach the browser through the ui-consts Store, and a style defined in a callback body never leaves Python.
_SITES_TABLE_COLUMNS = ["Site", "Sparsity (%)", "Start", "End", "Lifespan", "Basin Area (km²)"]
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
_CLEAR_BTN_VISIBLE = {
    "fontSize": "11px",
    "color": "#aaa",
    "cursor": "pointer",
    "textDecoration": "underline",
    "marginTop": "6px",
    "display": "block",
}
_CLEAR_BTN_HIDDEN = {**_CLEAR_BTN_VISIBLE, "display": "none"}
# fontWeight is set per row (bold for the active graph site), so it is deliberately absent here.
_UID_CELL = {
    "cursor": "pointer",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "whiteSpace": "nowrap",
    "display": "block",
    "maxWidth": "90px",
}
_REMOVE_CELL = {"cursor": "pointer", "color": "#bbb", "fontWeight": "bold", "fontSize": "14px"}


# The Iowa outline is a baked asset now; see load_iowa_geojson.
_IOWA_OUTLINE_ASSET = _ASSETS_DIR / "data" / "iowa_outline.geojson"


# Style constants the clientside callbacks need, gathered into one dict the layout ships to the
# browser as a dcc.Store. Passing them as State rather than restating them in JS keeps Python the
# single definition -- a tweak here changes the live app and the static build together.
def clientside_consts():
    return {
        "help_visible": _HELP_POPUP_STYLE,
        "help_hidden": {**_HELP_POPUP_STYLE, "display": "none"},
        "overlay_visible": _GRAPH_OVERLAY_VISIBLE,
        "overlay_hidden": _GRAPH_OVERLAY_HIDDEN,
        "tab_active": _MENU_TAB_ACTIVE,
        "tab_inactive": _MENU_TAB_INACTIVE,
        # Leaflet styles for the basin layers, keyed the way colors.basin_style() keys them so the
        # palette stays defined once in colors.py.
        "basin_style_preferred": colors.basin_style("preferred"),
        "basin_style_1": colors.basin_style(1),
        "basin_style_2": colors.basin_style(2),
        "basin_style_3": colors.basin_style(3),
        "hydro": {
            "urls": {
                "waterbodies": bundle.asset_url("iowa_waterbodies.geojson"),
                "flowlines": bundle.asset_url("iowa_flowlines.geojson"),
            },
            "styles": {
                "waterbodies": {
                    "color": colors.HYDRO["stroke"], "weight": 0.8,
                    "fillColor": colors.HYDRO["fill"], "fillOpacity": 0.45, "interactive": False,
                },
                "flowlines": {"color": colors.HYDRO["stroke"], "weight": 1.2, "interactive": False},
            },
        },
        "iem_bbox": {
            "bounds": [[38.8, -97.7], [45.3, -87.4]],
            "pathOptions": {"color": colors.IEM_BBOX["stroke"], "weight": 2, "dashArray": "6 4", "fillOpacity": 0},
        },
        # Site marker appearance by kind. The selected marker is drawn larger as well as recoloured, which is why the radius travels with the colours rather than being a constant in JS.
        "site_markers": {
            "selected": {"color": colors.SITE_SELECTED["stroke"], "fillColor": colors.SITE_SELECTED["fill"], "radius": 7},
            "usgs": {"color": colors.SITE_USGS["stroke"], "fillColor": colors.SITE_USGS["fill"], "radius": 5},
            "default": {"color": colors.SITE_DEFAULT["stroke"], "fillColor": colors.SITE_DEFAULT["fill"], "radius": 5},
        },
        "sites_table": {
            "columns": _SITES_TABLE_COLUMNS,
            "table": {"width": "100%", "borderCollapse": "collapse"},
            "th_left": {**_TH, "textAlign": "left", "paddingLeft": "0"},
            "th_center": {**_TH, "textAlign": "center"},
            "th_last": {**_TH, "borderBottom": "1px solid #ddd"},
            "td_left": {**_TD, "textAlign": "left", "paddingLeft": "0"},
            "td_center": {**_TD, "textAlign": "center"},
            "td_right": {**_TD, "textAlign": "right", "paddingRight": "0"},
            "uid": _UID_CELL,
            "remove": _REMOVE_CELL,
            "clear_visible": _CLEAR_BTN_VISIBLE,
            "clear_hidden": _CLEAR_BTN_HIDDEN,
        },
        # The pin. A forecast is computed at a REACH OUTLET, which stream-order-3 snapping puts a median 1.5 km from the click, so the marker moves there and these draw the journey: a dashed line back to the click and a small hollow dot marking it.
        "snap_connector": {"color": colors.SITE_SELECTED["stroke"], "weight": 1.5, "dashArray": "4 4", "opacity": 0.8},
        "snap_click": {"color": "#888", "weight": 1.5, "fillOpacity": 0, "opacity": 0.9},
        # Debug: the pin's NLDI basin in the v1 comparison colour. (No v3 twin -- that delineation is computed live and has no shipped counterpart.)
        "pin_basin_v1": colors.pin_basin_style("v1"),
        # The forecast's own drawing: the basin overlay at the snapped reach, and the figure colours.
        "forecast": {
            "basin_style": {"color": colors.FORECAST["basin"], "weight": 2, "fillOpacity": 0.05},
            "line": colors.FORECAST["line"],
            "alarm_fill": colors.FORECAST["alarm"],
        },
        # The three dash_extensions.assign() handles serialise to {"variable": "dashExtensions.default.functionN"}, which the browser resolves against assets/dashExtensions_default.js. Passing them through means the hover/popup behaviour is still written once, in Python, at the top of this module.
        "rain_grid": {
            "style": _GRID_STYLE_JS,
            "hoverStyle": _GRID_HOVER_JS,
            "onEachFeature": _GRID_ONEACH_JS,
            "nodata": _GRID_NODATA_COLOR,
        },
    }


def load_iowa_geojson():
    """The Iowa state outline, read from the baked asset.

    This used to download the Census TIGER zip on every module import, which made the app
    un-importable offline and un-snapshottable. build_bundle.py bakes it instead.
    """
    if not _IOWA_OUTLINE_ASSET.exists():
        raise FileNotFoundError(
            f"Missing {_IOWA_OUTLINE_ASSET.name}.\n"
            "Run `python -m widget.static.build_bundle --only iowa_outline` to generate it."
        )
    return json.loads(_IOWA_OUTLINE_ASSET.read_text())


iowa_geojson = load_iowa_geojson()
IWQIS_SITES = access.get_metadata()[["site_uid", "latitude", "longitude"]]


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


# The PRESENTATION-ONLY "bad sites" markers (_fake_bad_site_points / make_fake_bad_site_markers)
# were removed for the static build: 77 decorative dots sampled from river vertices with no data
# behind them, whose only purpose was to make the map look fully covered in a screen recording.


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
    """HTML tooltip string for one Voronoi cell: area + surplus + total N + crop list."""
    lines = [f"<b>Cell {int(row['node_id'])}</b>"]
    if pd.notna(row.get("cell_area")):
        area_km2 = row["cell_area"] / 1e6  # cell_area is m² (EPSG:5070)
        lines.append(f"Cell area: {area_km2:.2f} km²")
        if pd.notna(row.get("frac_cell_in_basin")):
            lines.append(f"Cell area in basin: {area_km2 * row['frac_cell_in_basin']:.2f} km²")
    if pd.notna(row.get("dist_to_sensor")):
        lines.append(f"Dist to sensor: {row['dist_to_sensor'] / 1e3:.1f} km")  # dist_to_sensor is m
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


@functools.lru_cache(maxsize=64)
def _rain_grid_features(uid, year, mode="surplus"):
    """GeoJSON (lon/lat) of a site's rain cells, each with a per-cell colour and a tooltip joining
    surplus + crop stats for `year`. `mode` picks the colour: 'surplus' (nitrogen-surplus gradient,
    the default) or 'crop' (dominant CDL crop, colors.CROP_COLORS). Raises FileNotFoundError if the
    rain grid hasn't been built. Cached per (uid, year, mode) -- the result is passed read-only to
    dl.GeoJSON, so callers must not mutate it."""
    grid = access.get_grid(uid).to_crs("EPSG:4326")
    cells = grid[["node_id", "geometry"] + [c for c in ("cell_area", "frac_cell_in_basin", "dist_to_sensor") if c in grid.columns]]

    try:
        s = access.get_surplus(uid)
        s = s[s["year"] == year][["node_id", "surplus_kgha", "total_kg_N"]]
    except FileNotFoundError:
        s = pd.DataFrame(columns=["node_id", "surplus_kgha", "total_kg_N"])

    try:
        c = access.get_crops(uid)
        c = c[c["year"] == year].drop(columns=["year"])
    except FileNotFoundError:
        c = pd.DataFrame(columns=["node_id"])
    crop_cols = [col for col in c.columns if col not in ("node_id", "global_node_id")]

    cells = cells.merge(s, on="node_id", how="left").merge(c, on="node_id", how="left")
    if mode == "crop":  # colour each cell by its dominant crop for the year
        present = [col for col in crop_cols if col in cells.columns]
        if present:
            filled = cells[present].fillna(0)
            dom = filled.idxmax(axis=1).where(filled.sum(axis=1) > 0)  # NaN where the cell has no crop data
            cells["color"] = dom.map(colors.CROP_COLORS).fillna(_GRID_NODATA_COLOR)
        else:
            cells["color"] = _GRID_NODATA_COLOR
    else:  # nitrogen-surplus gradient (default)
        cells["color"] = cells["surplus_kgha"].map(
            lambda v: surplus_viz.surplus_to_hex(v) if pd.notna(v) else _GRID_NODATA_COLOR
        )
    cells["tooltip"] = cells.apply(lambda r: _cell_tooltip(r, crop_cols), axis=1)
    return json.loads(cells[["node_id", "color", "tooltip", "geometry"]].to_json())


def _rain_grid_dots(uid):
    """Fallback: grid-cell centroids as dots, for sites without grid geometry yet."""
    try:
        grid = access.get_grid(uid)
    except FileNotFoundError:
        return []
    cells = grid[["lon", "lat"]].drop_duplicates()
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




# Export everything (incl. single-underscore helpers/constants) for `from map_common import *`.
__all__ = [n for n in list(globals()) if not n.startswith("__")]
