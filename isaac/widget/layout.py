"""Top-level page layout.

This is the only place that decides where each panel's output appears on
screen. The panels themselves (components/*) don't assume anything about
their position — map_panel exposes named LayerGroup slots ("mapunit-layer",
"forecast-layer") that info_panel and forecast_panel render into regardless
of where this file places the map.
"""

from dash import html, dcc

from components import map_panel, info_panel, forecast_panel


def build_layout():
    return html.Div(
        [
            dcc.Store(id="region-geom"),
            dcc.Store(id="selected-site"),
            html.H1("Iowa Nitrate Forecast Tool", style={"marginBottom": "8px"}),
            html.Div(
                style={"display": "flex", "gap": "4px", "alignItems": "flex-start"},
                children=[
                    map_panel.layout(),
                    html.Div(
                        style={"flex": "1", "minWidth": "0", "display": "flex",
                               "flexDirection": "column", "gap": "16px"},
                        children=[
                            info_panel.timeseries_layout(),
                            forecast_panel.layout(),
                        ],
                    ),
                ],
            ),
            info_panel.region_info_layout(),
        ],
        style={
            "fontFamily": "sans-serif",
            "maxWidth": "1800px",
            "margin": "0 auto",
            "padding": "16px",
        },
    )
