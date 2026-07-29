"""Forecast panel: drop a pin -> predicted nitrate + P(violation) timeseries at that point.

The UI lives in map_panel._build_forecast_section; this registers the two callbacks, both CLIENTSIDE (assets/clientside/forecast.js). Nothing here runs the model -- the browser scores the same boosters whether the app is served from Flask or from a bucket, so local and published cannot drift. model_interface.py keeps the Python path as the parity reference.

The figures carry no covariates: a pin has no observed precip series, and the light recipes do not use precip.
"""

from dash import ClientsideFunction, Input, Output, State


def register_callbacks(app):
    # Fetch the snapped reach's chunk, assemble a year of rows, walk both boosters, draw. The server
    # callback this replaces called NLDI and delineated a basin per click; everything but the
    # arithmetic is precomputed now (widget/static/build_reaches.py).
    app.clientside_callback(
        ClientsideFunction(namespace="forecast", function_name="runForecast"),
        Output("forecast-graph", "figure"),
        Output("forecast-graph", "style"),
        Output("forecast-results", "children"),
        Output("forecast-layer", "children"),
        Output("forecast-download-fig", "data"),
        Output("download-forecast-row", "style"),
        Input("run-forecast-button", "n_clicks"),
        State("region-geom", "data"),
        State("forecast-year", "value"),
        State("forecast-beta", "value"),
        State("ui-consts", "data"),
        prevent_initial_call=True,
    )

    # The PNG, rendered by Plotly from the same figure spec. Was a server-side matplotlib render.
    app.clientside_callback(
        ClientsideFunction(namespace="forecast", function_name="downloadForecast"),
        Output("forecast-download", "data"),
        Input("download-forecast-button", "n_clicks"),
        State("forecast-download-fig", "data"),
        State("ui-consts", "data"),
        prevent_initial_call=True,
    )
