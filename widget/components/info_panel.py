"""Info panel: displays data about the currently selected site or region.

Reacts to state written by the map panel:

- `active-graph-site`: the site_uid of a monitoring site selected on the map
  or from the Sites Selected table.  Drives the timeseries graph via
  `_build_timeseries_figure`, which plots nitrate concentration on the primary
  y-axis with a secondary y-axis reserved for rainfall (placeholder, uncomment
  once rain parquets are available).

- `region-geom`: the selected point or area geometry.  Currently displays the
  coordinates of a point selection.  Area-based information is not yet
  implemented.

Neither callback knows anything about the forecast model — this panel is
purely descriptive.
"""

import pandas as pd
import plotly.graph_objects as go
from dash import ClientsideFunction, Input, Output, State, html, dcc, dash_table

from src.data import access
import bundle


def _bundle_aggs(interval):
    """(nitrate_agg, precip_agg) the bundle stored this interval's series with.

    Kept as a lookup rather than constants so the figure always reduces the way the shipped series were reduced; if build_bundle.SERIES_INTERVALS changes, the chart follows without a second edit."""
    for i, nit, pre in bundle.series_intervals():
        if i == interval:
            return nit, pre
    return "max", "mean"


def _render_tables(pairs):
    """pairs: list of (title, df). Returns Dash elements, or a placeholder."""
    sections = []
    for title, df in pairs:
        if df is None or df.empty:
            continue
        if title:
            sections.append(html.H3(title, style={"marginTop": "20px", "marginBottom": "4px"}))
        sections.append(
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                style_table={"overflowX": "auto", "marginTop": "8px"},
                style_cell={"textAlign": "left", "padding": "6px 12px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "borderBottom": "2px solid #ddd"},
                page_size=20,
            )
        )
    if sections:
        return html.Div(sections)
    return html.P("No data for this location.", style={"color": "#888"})


_SEASON_MONTHS_DAYS = [(3, 21), (6, 21), (9, 21), (12, 21)]
_BAR_THRESH = 200  # if there are more than this many bars, do a scatter for rain


def _build_timeseries_figure(
    site_uid: str, interval: str, agg_func_water: str, agg_func_rain: str, show_seasons: bool = False
) -> go.Figure:
    """Build the site timeseries figure with nitrate and rain traces.

    Nitrate is plotted on the primary y-axis.  A secondary y-axis is reserved
    for rainfall — uncomment the rain block below once the rain parquets are
    ready for the sites you care about.
    """
    fig = go.Figure()
    try:
        precip_col = f"precip_{interval.lower()}"
        # Basin-mean daily precip, then resample to the interval. (Basin-mean-then-resample
        # == the old per-cell-resample-then-spatial-mean for the linear sum/mean aggregators.)
        w = access.get_weather(site_uid)
        daily = w.groupby("date")["precip_in_1d"].mean()
        daily.index = pd.DatetimeIndex(daily.index)
        rain = daily.resample(interval).agg(agg_func_rain).rename(precip_col).reset_index()
        if rain.shape[0] < _BAR_THRESH:
            fig.add_trace(
                go.Bar(
                    x=rain["date"],
                    y=rain[precip_col],
                    name="Precip (in)",
                    yaxis="y2",
                    marker_color="#3a94fa",
                    opacity=0.5,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=rain["date"],
                    y=rain[precip_col],
                    name="Precip (in)",
                    yaxis="y2",
                    opacity=0.5,
                    line={"color": "#3a94fa", "width": 1},
                )
            )
        fig.update_layout(
            yaxis2=dict(
                title="Precipitation (in)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
    except FileNotFoundError:
        pass  # no rain data yet for this site — silently omit the trace

    # ── nitrate trace ─────────────────────────────────────────────────────────
    nitrate = access.aggregate_by_interval(site_uid=site_uid, interval=interval, agg_func=agg_func_water)
    fig.add_trace(
        go.Scatter(
            x=nitrate.index,
            y=nitrate.values,
            name="Nitrate (mg/L)",
            yaxis="y1",
            line={"color": "#6b21a8", "width": 1.5},
        )
    )

    fig.update_layout(
        yaxis={"title": "Nitrate (mg/L)"},
        xaxis={"title": None},
        legend={"orientation": "h", "y": -0.15},
        margin={"t": 20, "b": 40, "l": 50, "r": 50},
        xaxis_rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    if show_seasons:
        start_year = nitrate.index.min().year
        end_year = nitrate.index.max().year
        for year in range(start_year, end_year + 1):
            for month, day in _SEASON_MONTHS_DAYS:
                fig.add_vline(
                    x=pd.Timestamp(year, month, day, tz="UTC").isoformat(),
                    line_width=1,
                    line_color="black",
                    opacity=0.3,
                )

    fig.update_yaxes(fixedrange=True)
    fig.add_annotation(
        text=site_uid,
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": "#888"},
    )

    return fig


_POINT_ROW_VISIBLE = {"marginBottom": "12px"}
_POINT_ROW_HIDDEN = {**_POINT_ROW_VISIBLE, "display": "none"}


def region_info_layout():
    """The pin-drop readout, pre-rendered and hidden.

    This used to be built in a callback, which on the static site meant a control that renders and never updates. The shape never varied though -- a label and a coordinate pair -- so the skeleton is rendered here and the callback writes only the coordinate string and the row's visibility. The Polygon branch it also handled went with the polygon selector.
    """
    return html.Div(
        id="region-info-panel",
        style={"padding": "16px 0"},
        children=[
            dcc.Store(id="region-info-consts", data={"point_row_visible": _POINT_ROW_VISIBLE, "point_row_hidden": _POINT_ROW_HIDDEN}),
            html.Div(
                [html.Strong("Selected point: "), html.Code(id="region-point-coords")],
                id="region-point-row",
                style=_POINT_ROW_HIDDEN,
            ),
        ],
    )


def register_callbacks(app):
    # Built clientside from series/{uid}_{interval}.bin. _build_timeseries_figure above stays as
    # the reference implementation and is what the parity spot-check compares against; it is also
    # what the local Python path would use if the clientside function were ever removed.
    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="timeseries"),
        Output("timeseries-graph", "figure"),
        Input("active-graph-site", "data"),
        Input("aggregate-interval", "value"),
        Input("graph-toggle", "value"),
        prevent_initial_call=True,
    )

    # The mapunit-layer output went with this callback: it only ever returned [], and nothing else writes that LayerGroup -- it is the "reserved for future use" slot map_common documents.
    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="regionInfo"),
        Output("region-point-coords", "children"),
        Output("region-point-row", "style"),
        Input("region-geom", "data"),
        State("region-info-consts", "data"),
        prevent_initial_call=True,
    )
