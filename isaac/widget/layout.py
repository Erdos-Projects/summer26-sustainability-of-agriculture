"""Top-level page layout.

The map spans the full width. Below it, the timeseries graph and forecast
panel sit side by side, with the soil/region info tables beneath both.

Panels don't assume anything about their position — map_panel exposes named
LayerGroup slots ("mapunit-layer", "forecast-layer") that info_panel and
forecast_panel render into regardless of where this file places the map.
"""

from dash import html, dcc

from components import map_panel, info_panel, forecast_panel


def build_layout():
    return html.Div(
        [
            dcc.Store(id="region-geom"),
            dcc.Store(id="selected-site"),
            html.H1("Iowa Nitrate Forecast Tool", style={"marginBottom": "8px"}),
            map_panel.layout(),
            html.Div(
                [
                    html.Div(
                        [
                            info_panel.timeseries_layout(),
                            forecast_panel.layout(),
                        ],
                        style={
                            "display": "flex",
                            "gap": "16px",
                            "alignItems": "flex-start",
                        },
                    ),
                    info_panel.region_info_layout(),
                ],
                style={"marginTop": "16px"},
            ),
        ],
        style={
            "fontFamily": "sans-serif",
            "maxWidth": "1800px",
            "margin": "0 auto",
            "padding": "16px",
        },
    )
