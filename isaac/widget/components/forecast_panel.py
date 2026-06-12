"""Forecast panel: takes a nitrogen surplus value for the selected region and
asks the model whether downstream sites will exceed 10 mg/L nitrate within a
month.

This panel only deals with the model seam (data.geo_utils +
model_interface) and presentation of the result — it doesn't know how
regions are selected (that's map_panel) or how to describe a region
(that's info_panel).
"""

from dash import Input, Output, State, html, dcc

from data import geo_utils, iwqis_utils
import model_interface


def layout():
    return html.Div(
        id="forecast-panel",
        children=[
            html.H3("Forecast"),
            html.Div(
                [
                    html.Label("Average nitrogen surplus (kg/ha): ", style={"marginRight": "8px"}),
                    dcc.Input(id="surplus-input", type="number", value=0, style={"width": "100px"}),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Button("Run forecast", id="run-forecast-button", n_clicks=0),
            html.Div(id="forecast-results", style={"marginTop": "12px"}),
        ],
        style={"padding": "16px 0"},
    )


def _render_results(results):
    if not results:
        return html.P("No downstream sites identified for this region.", style={"color": "#888"})

    rows = []
    for uid, result in results.items():
        rows.append(
            html.Tr([
                html.Td(uid),
                html.Td("Yes" if result.exceeds_threshold else "No"),
                html.Td(f"{result.probability:.0%}"),
                html.Td(str(result.eta_days) if result.eta_days is not None else "—"),
            ])
        )

    return html.Table(
        [
            html.Thead(html.Tr([
                html.Th("Site"),
                html.Th("Exceeds 10 mg/L within 1 month?"),
                html.Th("Probability"),
                html.Th("ETA (days)"),
            ])),
            html.Tbody(rows),
        ],
        style={"fontSize": "13px", "borderCollapse": "collapse"},
    )


def register_callbacks(app):
    @app.callback(
        Output("forecast-results", "children"),
        Output("forecast-layer", "children"),
        Input("run-forecast-button", "n_clicks"),
        State("region-geom", "data"),
        State("surplus-input", "value"),
        prevent_initial_call=True,
    )
    def run_forecast(n_clicks, region_geom, surplus):
        if not region_geom:
            return html.P("Select a point or area on the map first.", style={"color": "#888"}), []

        forecast_region = geo_utils.normalize_for_forecast(region_geom)
        target_uids = geo_utils.get_downstream_sites(forecast_region, iwqis_utils.get_iwqis_sites())

        try:
            results = model_interface.forecast_exceedance(forecast_region, surplus, target_uids)
        except NotImplementedError:
            return html.P("Forecast model is not yet implemented.", style={"color": "#888"}), []

        return _render_results(results), []
