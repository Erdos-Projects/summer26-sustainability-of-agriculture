"""Info panel: displays data about the currently selected region.

Reacts to two independent pieces of state set by the map panel:

- `region-geom`: the selected point or area. Drives the soil-data tables and
  the `mapunit-layer` map slot (currently implemented for point selections
  only; area selections show a placeholder until an area-based soil query is
  added).
- `selected-site`: an IWQIS site clicked on the map. Drives the timeseries
  graph.

Neither callback knows anything about the forecast model — this panel is
purely descriptive.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_leaflet as dl
from dash import Input, Output, html, dcc, dash_table, no_update

from data import sda_utils, iwqis_utils


# ── REGION DATA INTERFACE ─────────────────────────────────────────────────
def get_location_data(lat: float, lon: float) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Called for a point selection. Return a pd.DataFrame, or a list of
    pd.DataFrames, to display as tables below the map. An empty DataFrame or
    empty list shows nothing. Optionally use (title, df) tuples in the list
    to label each table.
    """
    crops, horizon, restrictions = sda_utils.get_tables_from_point(lat, lon)
    return [
        ("Crop", crops),
        ("Horizons", horizon),
        ("Restrictions", restrictions)
    ]
# ─────────────────────────────────────────────────────────────────────────


# ── TIMESERIES DATA INTERFACE ───────────────────────────────────────────────
def get_site_timeseries(uid) -> go.Figure | None:
    """
    Called when an IWQIS site marker is selected. Return a Plotly Figure
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
# ─────────────────────────────────────────────────────────────────────────


def _render_tables(pairs):
    """pairs: list of (title, df). Returns Dash elements, or a placeholder."""
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
    if sections:
        return html.Div(sections)
    return html.P("No data for this location.", style={"color": "#888"})


def timeseries_layout():
    return html.Div(
        dcc.Graph(
            id="timeseries-graph",
            style={"width": "100%", "height": "600px"},
            config={"responsive": True},
            figure=go.Figure(),
        ),
        style={"flex": "1", "minWidth": "0"},
    )


def region_info_layout():
    return html.Div(id="region-info-panel", style={"padding": "16px 0"})


def register_callbacks(app):
    @app.callback(
        Output("timeseries-graph", "figure"),
        Input("selected-site", "data"),
        prevent_initial_call=True,
    )
    def update_timeseries(uid):
        if uid is None:
            return go.Figure()
        fig = get_site_timeseries(uid)
        return fig if fig is not None else go.Figure()

    @app.callback(
        Output("mapunit-layer", "children"),
        Output("region-info-panel", "children"),
        Input("region-geom", "data"),
        prevent_initial_call=True,
    )
    def update_region_info(region_geom):
        if not region_geom:
            return no_update, no_update

        geom_type = region_geom.get("type")

        if geom_type == "Point":
            lng, lat = region_geom["coordinates"]

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

            coord_row = html.Div(
                [html.Strong("Selected point: "), html.Code(f"{lat:.6f}, {lng:.6f}")],
                style={"marginBottom": "12px"},
            )

            result = get_location_data(lat, lng)
            if isinstance(result, pd.DataFrame):
                pairs = [(None, result)]
            else:
                pairs = [(t, d) if isinstance(t, str) else (None, t)
                         for t, d in (r if isinstance(r, tuple) else (None, r) for r in result)]

            return [mapunit_layer], html.Div([coord_row, _render_tables(pairs)])

        # Area selection (Polygon/MultiPolygon from rectangle or polygon draw)
        coord_row = html.Div(
            [html.Strong("Selected area")],
            style={"marginBottom": "12px"},
        )
        placeholder = html.P(
            "Area-based information is not yet implemented.",
            style={"color": "#888"},
        )
        return [], html.Div([coord_row, placeholder])
