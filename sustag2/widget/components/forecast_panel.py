"""Forecast panel: drop a pin -> predicted nitrate + P(violation) timeseries at that point.

The UI (year dropdown, Run button, results, graph) lives in map_panel._build_forecast_section; this
module owns the run_forecast callback and the figure. The model seam is model_interface
(forecast_virtual_site -> the deploy build_virtual_basin/virtual_recipe/predict chain).
"""

import dash_leaflet as dl
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update
from plotly.subplots import make_subplots

import model_interface

_RED = "#c1121f"
_BLUE = "#3a94fa"


def _forecast_figure(vf, year) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(f"Predicted nitrate (mg/L) — {year}", "P(violation ≥ 10 mg/L)"),
    )
    # top: predicted nitrate (red) + basin precip (blue, twin)
    fig.add_trace(go.Scatter(x=vf.reg.index, y=vf.reg.to_numpy(), name="nitrate", line={"color": _RED, "width": 1.5}),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=vf.precip.index, y=vf.precip.to_numpy(), name="precip", line={"color": _BLUE, "width": 1},
                             opacity=0.5), row=1, col=1, secondary_y=True)
    fig.add_hline(y=10, line_dash="dash", line_color=_RED, opacity=0.4, row=1, col=1)
    # bottom: P(violation) (red) + basin precip (blue, twin)
    fig.add_trace(go.Scatter(x=vf.clf.index, y=vf.clf.to_numpy(), name="P(viol)", line={"color": _RED, "width": 1.5}),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=vf.precip.index, y=vf.precip.to_numpy(), name="precip", line={"color": _BLUE, "width": 1},
                             opacity=0.5, showlegend=False), row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text="mg/L", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="precip (in)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(range=[0, 1], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="precip (in)", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_layout(height=340, margin={"t": 30, "b": 30, "l": 44, "r": 44}, showlegend=False,
                      font={"size": 10})
    return fig


def register_callbacks(app):
    @app.callback(
        Output("forecast-graph", "figure"),
        Output("forecast-graph", "style"),
        Output("forecast-results", "children"),
        Output("forecast-layer", "children"),
        Input("run-forecast-button", "n_clicks"),
        State("region-geom", "data"),
        State("forecast-year", "value"),
        prevent_initial_call=True,
    )
    def run_forecast(n_clicks, region_geom, year):
        if not region_geom or region_geom.get("type") != "Point":
            return no_update, {"display": "none"}, \
                html.P("Drop a pin first (Pin drop selection mode).", style={"color": "#888"}), []

        lng, lat = region_geom["coordinates"]
        try:
            vf = model_interface.forecast_virtual_site(lat, lng, int(year))
        except Exception as e:
            return no_update, {"display": "none"}, \
                html.P(f"Forecast failed: {type(e).__name__}: {e}", style={"color": "#c00", "fontSize": "12px"}), []

        summary = html.P(
            f"Peak P(violation): {vf.peak_prob:.0%} · {vf.days_over} days predicted ≥ 10 mg/L",
            style={"fontSize": "12px", "marginTop": "6px"},
        )
        basin_layer = [dl.GeoJSON(data=vf.basin_geojson,
                                  options={"style": {"color": "#c026d3", "weight": 2, "fillOpacity": 0.05}})]
        return _forecast_figure(vf, year), {"display": "block", "height": "340px"}, summary, basin_layer
