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

import geopandas as gpd
import dash_leaflet as dl
from dash import Input, Output, State, html, dcc, no_update, ALL, ctx

from data import iwqis_utils

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

TILE_URLS = {
    "street": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

# dl.EditControl's `draw` prop is only read once, at mount (dash-leaflet
# does not react to later changes to it), so the set of enabled draw tools
# is fixed here rather than toggled by a callback. Rectangle and polygon are
# both always available as separate buttons in the toolbar; which one the
# user clicks is up to them.
DRAW_TOOLS = {
    "rectangle": True,
    "polygon": True,
    "polyline": False,
    "circle": False,
    "circlemarker": False,
    "marker": False,
}


def load_iowa_geojson():
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    )
    return states[states["NAME"] == "Iowa"].__geo_interface__


iowa_geojson = load_iowa_geojson()
IWQIS_SITES = iwqis_utils.get_iwqis_sites()


def make_iwqis_markers(selected_uid=None):
    """Build small clickable circle markers for the IWQIS sites.

    Each marker uses bubblingMouseEvents=False so clicking it does not also
    trigger the map's click handler (which would otherwise also update the
    region selection). Each marker has a pattern-matching id,
    {"type": "iwqis-marker", "index": uid}, so on_iwqis_marker_click below
    can tell which site was clicked.

    The marker whose uid matches selected_uid is drawn in a different color
    to indicate it is currently selected.
    """
    sites = list(IWQIS_SITES[["uid", "latitude", "longitude"]].itertuples(index=False, name=None))
    return [
        dl.CircleMarker(
            id={"type": "iwqis-marker", "index": uid},
            center=[lat, lon],
            radius=7 if uid == selected_uid else 5,
            color="darkred" if uid == selected_uid else "darkgreen",
            fillColor="red" if uid == selected_uid else "limegreen",
            fillOpacity=0.8,
            weight=1,
            bubblingMouseEvents=False,
        )
        for (uid, lat, lon) in sites
    ]


def _overlay_style(top=None, bottom=None, left=None, right=None):
    style = {
        "position": "absolute",
        "zIndex": 1000,
        "background": "white",
        "padding": "6px 10px",
        "borderRadius": "4px",
        "border": "1px solid rgba(0,0,0,0.2)",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
    }
    for key, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if value is not None:
            style[key] = value
    return style


def layout():
    """Return the map panel: the Leaflet map plus its overlay controls."""
    return html.Div(
        style={"position": "relative", "width": "50%", "flexShrink": 0},
        children=[
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
                    dl.LayerGroup(id="mapunit-layer"),
                    dl.LayerGroup(id="iwqis-layer"),
                    dl.LayerGroup(id="marker-layer"),
                    dl.FeatureGroup([
                        dl.EditControl(
                            id="edit-control",
                            position="bottomright",
                            draw=DRAW_TOOLS,
                            edit={"edit": False, "remove": True},
                        )
                    ]),
                    dl.LayerGroup(id="forecast-layer"),
                ],
                style={"height": "600px", "width": "100%"},
            ),
            html.Div(
                [
                    dcc.RadioItems(
                        id="selection-mode",
                        options=[
                            {"label": " Point", "value": "point"},
                            {"label": " Area", "value": "area"},
                        ],
                        value="point",
                        style={"fontSize": "13px"},
                    ),
                    html.Div(
                        "Use the rectangle/polygon tools in the map's "
                        "toolbar (bottom-right) to draw an area.",
                        id="area-tool-container",
                        style={
                            "marginTop": "4px",
                            "marginLeft": "16px",
                            "maxWidth": "160px",
                            "fontSize": "12px",
                            "color": "#555",
                            "display": "none",
                        },
                    ),
                ],
                style=_overlay_style(top="10px", left="10px"),
            ),
            html.Div(
                dcc.RadioItems(
                    id="tile-selector",
                    options=[
                        {"label": " Street", "value": "street"},
                        {"label": " Satellite", "value": "satellite"},
                    ],
                    value="street",
                    style={"fontSize": "13px"},
                ),
                style=_overlay_style(bottom="24px", left="10px"),
            ),
            html.Div(
                dcc.Checklist(
                    id="iwqis-toggle",
                    options=[{"label": " Show IWQIS sites", "value": "show"}],
                    value=[],
                    style={"fontSize": "13px"},
                ),
                style=_overlay_style(top="10px", right="10px"),
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
        style = {"marginTop": "4px", "marginLeft": "16px", "maxWidth": "160px",
                 "fontSize": "12px", "color": "#555"}
        if mode != "area":
            style["display"] = "none"
        return style

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
