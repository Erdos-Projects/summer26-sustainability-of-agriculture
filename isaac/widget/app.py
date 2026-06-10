import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, Input, Output, callback, html, no_update, dcc, dash_table, ALL, ctx
import dash_leaflet as dl
import sda_utils
import iwqis_utils

# ── DATA INTERFACE ────────────────────────────────────────────────────────────
def get_location_data(lat: float, lon: float) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Called on every map click. Return a pd.DataFrame, or a list of pd.DataFrames,
    to display as tables below the map. An empty DataFrame or empty list shows nothing.
    Optionally use (title, df) tuples in the list to label each table.
    """
    crops, horizon, restrictions = sda_utils.get_tables_from_point(lat, lon)
    return [
        ("Crop", horizon)
        ("Horizons", horizon)
        ("Restrictions", horizon)
    ]
# ─────────────────────────────────────────────────────────────────────────────

# ── TIMESERIES DATA INTERFACE ─────────────────────────────────────────────────
def get_site_timeseries(uid) -> go.Figure | None:
    """
    Called when an IWQIS site marker is clicked. Return a Plotly Figure
    showing a timeseries for the site with the given uid, to display next
    to the map. Return None to clear the graph.
    """
    site_df = iwqis_utils.get_site_data(uid)
    agg_df = iwqis_utils.aggregate_by_interval(site_df, "nitrate_con", "1D")
    fig = px.line(agg_df.reset_index(), 
                  x=agg_df.index.name, 
                  y=agg_df.columns.tolist() if isinstance(agg_df, pd.DataFrame) else agg_df.name, 
                  title=f"{uid} Daily Avg. Nitrate Concentration", labels={'nitrate_con': "Nitrate mg/L"}
    )
    return fig
# ─────────────────────────────────────────────────────────────────────────────

IOWA_CENTER = [42.0, -93.5]
IOWA_ZOOM = 7

def load_iowa_geojson():
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    )
    return states[states["NAME"] == "Iowa"].__geo_interface__


iowa_geojson = load_iowa_geojson()

IWQIS_SITES = iwqis_utils.get_iwqis_sites()

def make_iwqis_markers(selected_uid=None):
    """Build small clickable circle markers for a list of (lat, lon) sites.

    Each marker uses bubblingMouseEvents=False so clicking it does not also
    trigger the map's click handler (which places the coordinate pin).
    Each marker has a pattern-matching id, {"type": "iwqis-marker", "index": i},
    so a future callback can listen for clicks on individual sites.

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

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Store(id="selected-site"),
        html.H1("Iowa Coordinate Picker", style={"marginBottom": "8px"}),
        html.Div(
            style={"display": "flex", "gap": "4px", "alignItems": "flex-start"},
            children=[
                html.Div(
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
                            ],
                            style={"height": "600px", "width": "100%"},
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
                            style={
                                "position": "absolute",
                                "bottom": "24px",
                                "left": "10px",
                                "zIndex": 1000,
                                "background": "white",
                                "padding": "6px 10px",
                                "borderRadius": "4px",
                                "border": "1px solid rgba(0,0,0,0.2)",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
                            },
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="iwqis-toggle",
                                options=[{"label": " Show IWQIS sites", "value": "show"}],
                                value=[],
                                style={"fontSize": "13px"},
                            ),
                            style={
                                "position": "absolute",
                                "top": "10px",
                                "right": "10px",
                                "zIndex": 1000,
                                "background": "white",
                                "padding": "6px 10px",
                                "borderRadius": "4px",
                                "border": "1px solid rgba(0,0,0,0.2)",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
                            },
                        ),
                    ],
                ),
                html.Div(
                    dcc.Graph(
                        id="timeseries-graph",
                        style={"width": "100%", "height": "600px"},
                        config={"responsive": True},
                        figure=go.Figure(),
                    ),
                    style={"flex": "1", "minWidth": "0"},
                ),
            ],
        ),
        html.Div(id="info-panel", style={"padding": "16px 0"}),
    ],
    style={
        "fontFamily": "sans-serif",
        "maxWidth": "1800px",
        "margin": "0 auto",
        "padding": "16px",
    },
)


TILE_URLS = {
    "street": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}


@callback(
    Output("tile-layer", "url"),
    Input("tile-selector", "value"),
)
def switch_tile_layer(value):
    return TILE_URLS[value]


@callback(
    Output("iwqis-layer", "children"),
    Input("iwqis-toggle", "value"),
    Input("selected-site", "data"),
)
def render_iwqis_sites(value, selected_uid):
    if "show" in value:
        return make_iwqis_markers(selected_uid)
    return []


@callback(
    Output("timeseries-graph", "figure"),
    Output("selected-site", "data"),
    Input({"type": "iwqis-marker", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_iwqis_marker_click(n_clicks_list):
    if not any(n_clicks_list):
        return no_update, no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update

    uid = triggered["index"]
    fig = get_site_timeseries(uid)
    return (fig if fig is not None else go.Figure()), uid


@callback(
    Output("marker-layer", "children"),
    Output("mapunit-layer", "children"),
    Output("info-panel", "children"),
    Input("map", "click_lat_lng"),
    prevent_initial_call=True,
)
def on_map_click(click_lat_lng):
    if not click_lat_lng:
        return no_update, no_update, no_update

    lat, lng = click_lat_lng

    marker = dl.Marker(
        position=[lat, lng],
        children=dl.Tooltip(f"{lat:.6f}, {lng:.6f}", permanent=True, direction="top"),
    )

    mapunit_geojson = sda_utils.get_mapunit_geojson_from_point(lat, lng)
    mapunit_layer = dl.GeoJSON(
        data=mapunit_geojson,
        options={
            "style": {
                "color": "orange",
                "weight": 2,
                "fillOpacity": 0.15,
                "fillColor": "orange",
            }
        },
    )

    result = get_location_data(lat, lng)

    coord_row = html.Div(
        [html.Strong("Selected: "), html.Code(f"{lat:.6f}, {lng:.6f}")],
        style={"marginBottom": "12px"},
    )

    # Normalise to a list of (title, df) pairs
    if isinstance(result, pd.DataFrame):
        pairs = [(None, result)]
    else:
        pairs = [(t, d) if isinstance(t, str) else (None, t)
                 for t, d in (r if isinstance(r, tuple) else (None, r) for r in result)]

    sections = []
    for title, df in pairs:
        if df is None or df.empty:
            continue
        if title:
            sections.append(html.H3(title, style={"marginTop": "20px", "marginBottom": "4px"}))
        sections.append(dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_table={"overflowX": "auto", "marginTop": "8px"},
            style_cell={"textAlign": "left", "padding": "6px 12px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "borderBottom": "2px solid #ddd"},
            page_size=20,
        ))

    data_section = html.Div(sections) if sections else html.P(
        "No data for this location.", style={"color": "#888"}
    )

    return [marker], [mapunit_layer], html.Div([coord_row, data_section])


if __name__ == "__main__":
    app.run(debug=True)
