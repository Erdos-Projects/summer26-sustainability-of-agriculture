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

import plotly.graph_objects as go
from dash import Input, Output, html, dash_table, no_update

from data import water


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



def region_info_layout():
    return html.Div(id="region-info-panel", style={"padding": "16px 0"})


def register_callbacks(app):
    @app.callback(
        Output("timeseries-graph", "figure"),
        Input("active-graph-site", "data"),
        prevent_initial_call=True,
    )
    def update_timeseries(active_uid):
        if not active_uid:
            return go.Figure()
        return water.make_site_timeseries_plot(active_uid)

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

            coord_row = html.Div(
                [html.Strong("Selected point: "), html.Code(f"{lat:.6f}, {lng:.6f}")],
                style={"marginBottom": "12px"},
            )

            return [], html.Div([coord_row])

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
