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
    # bottom: alarm-day shading (behind), P(violation) (red) + basin precip (blue, twin)
    if vf.tau is not None and vf.alarms is not None:
        # faint red band over days flagged as alarms (P(violation) >= tau(beta)); hv step so each
        # flagged day is a full block, drawn first so the P(viol) line sits on top.
        fig.add_trace(
            go.Scatter(x=vf.alarms.index, y=vf.alarms.astype(float).to_numpy(),
                       line={"width": 0, "shape": "hv"}, fill="tozeroy",
                       fillcolor="rgba(193,18,31,0.13)", name="alarm", showlegend=False,
                       hoverinfo="skip"),
            row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=vf.clf.index, y=vf.clf.to_numpy(), name="P(viol)", line={"color": _RED, "width": 1.5}),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=vf.precip.index, y=vf.precip.to_numpy(), name="precip", line={"color": _BLUE, "width": 1},
                             opacity=0.5, showlegend=False), row=2, col=1, secondary_y=True)
    if vf.tau is not None:  # the decision threshold: alarm where P(violation) crosses this line
        fig.add_hline(y=vf.tau, line_dash="dot", line_color="#555", line_width=1, row=2, col=1,
                      annotation_text=f"τ={vf.tau:.2f}", annotation_font_size=9)

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
        State("forecast-beta", "value"),
        prevent_initial_call=True,
    )
    def run_forecast(n_clicks, region_geom, year, beta):
        if not region_geom or region_geom.get("type") != "Point":
            return no_update, {"display": "none"}, \
                html.P("Drop a pin first (Pin drop selection mode).", style={"color": "#888"}), []

        lng, lat = region_geom["coordinates"]
        try:
            vf = model_interface.forecast_virtual_site(lat, lng, int(year), beta=float(beta))
        except Exception as e:
            return no_update, {"display": "none"}, \
                html.P(f"Forecast failed: {type(e).__name__}: {e}", style={"color": "#c00", "fontSize": "12px"}), []

        peak_line = html.P(
            f"Peak P(violation): {vf.peak_prob:.0%} · {vf.days_over} days predicted ≥ 10 mg/L",
            style={"fontSize": "12px", "marginTop": "6px", "marginBottom": "2px"},
        )
        if vf.tau is not None:  # β operating point from the tuned classifier
            op_line = html.P(
                f"β={vf.beta:g} → alarm at P ≥ {vf.tau:.2f} · {int(vf.alarms.sum())} alarm days. "
                f"Expected: catches ~{vf.recall:.0%} of violations · ~{vf.fdr:.0%} of alarms false "
                f"(at ~{vf.base_rate:.0%} base-rate prevalence).",
                style={"fontSize": "12px", "marginTop": "0", "color": "#444"},
            )
            summary = html.Div([peak_line, op_line])
        else:
            summary = peak_line
        basin_layer = [dl.GeoJSON(data=vf.basin_geojson,
                                  options={"style": {"color": "#c026d3", "weight": 2, "fillOpacity": 0.05}})]
        return _forecast_figure(vf, year), {"display": "block", "height": "340px"}, summary, basin_layer
