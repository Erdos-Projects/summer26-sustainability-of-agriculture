"""Top-level page layout.

The map spans the full width. Below it, the timeseries graph and forecast
panel sit side by side, with the soil/region info tables beneath both.

Panels don't assume anything about their position — map_panel exposes named
LayerGroup slots ("mapunit-layer", "forecast-layer") that info_panel and
forecast_panel render into regardless of where this file places the map.
"""

from dash import html, dcc

from components import map_panel, info_panel


def build_layout():
    return html.Div(
        style={"fontFamily": "sans-serif"},
        children=[
            dcc.Store(id="region-geom"),
            dcc.Store(id="selected-site"),
            dcc.Store(id="active-graph-site"),
            html.Div(
                html.H1("Iowa Nitrate Forecast Tool", style={"marginBottom": "8px"}),
                style={"maxWidth": "1800px", "margin": "0 auto", "padding": "16px 16px 0 16px"},
            ),
            map_panel.layout(),
            html.Div(
                info_panel.region_info_layout(),
                style={"maxWidth": "1800px", "margin": "0 auto", "padding": "16px"},
            ),
        ],
    )
