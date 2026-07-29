"""Map panel layout: the section builders + the main layout(). Split from map_panel.py.

Shared constants, styles, and helpers come from map_common (star-imported)."""

from .map_common import *  # noqa: F401,F403

_LEGEND_BOX_STYLE = {
    "background": "rgba(255,255,255,0.95)",
    "border": "1px solid #ccc",
    "borderRadius": "6px",
    "padding": "6px 10px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.25)",
    "fontSize": "12px",
}


def _legend_swatch(fill, stroke):
    return html.Span(
        style={
            "display": "inline-block",
            "width": "12px",
            "height": "12px",
            "borderRadius": "50%",
            "backgroundColor": fill,
            "border": f"2px solid {stroke}",
            "marginRight": "6px",
            "flex": "0 0 auto",
        }
    )


def _legend_row(fill, stroke, label):
    return html.Div(
        [_legend_swatch(fill, stroke), html.Span(label)],
        style={"display": "flex", "alignItems": "center", "marginTop": "3px"},
    )


def _build_legend():
    """Collapsible site-source legend (IWQIS vs USGS). Positioned by the bottom-left legend stack."""
    return html.Details(
        id="map-legend",
        open=True,
        style=_LEGEND_BOX_STYLE,
        children=[
            html.Summary("Legend", style={"cursor": "pointer", "fontWeight": 600, "outline": "none"}),
            html.Div(
                [
                    _legend_row(colors.SITE_DEFAULT["fill"], colors.SITE_DEFAULT["stroke"], "IWQIS site"),
                    _legend_row(colors.SITE_USGS["fill"], colors.SITE_USGS["stroke"], "USGS site"),
                ],
                style={"marginTop": "4px"},
            ),
        ],
    )


def _crop_legend():
    """Skinny vertical list of the dominant-crop colours (no header). Shown when the rain grid is on
    and grid-colour mode is 'crop' (see render_grid_color_legend)."""
    return html.Div(
        [_legend_row(colors.CROP_COLORS[c], colors.CROP_COLORS[c], c.replace("_", " ")) for c in colors.CROP_COLORS],
        style=_LEGEND_BOX_STYLE,
    )


def _surplus_legend():
    """Vertical nitrogen-surplus colour scale. Sampled from the ACTUAL cell colour map
    (surplus_viz.surplus_to_hex -> YlOrRd over the global min/max) so the bar matches the grid, low
    (bottom) -> high (top). Shown when the rain grid is on and grid-colour mode is 'surplus'."""
    lo, hi = surplus_viz._min_surplus(), surplus_viz._max_surplus()
    n = 12
    stops = [surplus_viz.surplus_to_hex(lo + i / (n - 1) * (hi - lo)) for i in range(n)]
    bar = html.Div(
        style={
            "width": "12px",
            "height": "80px",
            "borderRadius": "3px",
            "border": "1px solid #bbb",
            "background": "linear-gradient(to top, " + ", ".join(stops) + ")",
        }
    )
    ends = html.Div(
        [html.Span("high", style={"fontSize": "10px"}), html.Span("low", style={"fontSize": "10px"})],
        style={
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "height": "80px",
            "marginLeft": "6px",
        },
    )
    return html.Div([bar, ends], style={**_LEGEND_BOX_STYLE, "display": "flex", "alignItems": "stretch"})


def _build_selection_section():
    details = html.Details(
        [
            html.Summary("Selection", style=_SECTION_LABEL_SUMMARY),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "8px"},
                children=[
                    dcc.RadioItems(
                        id="selection-mode",
                        options=[
                            {"label": " Pin Drop", "value": "pin"},
                            {"label": " Point", "value": "point"},
                        ],
                        value=colors.default("selection-mode"),
                        style={"display": "flex", "gap": "10px"},
                        labelStyle={"fontSize": "13px", "display": "flex", "alignItems": "center", "cursor": "pointer"},
                        inputStyle={"marginRight": "3px"},
                    ),
                ],
            ),
            html.Hr(style={"margin": "8px 0", "borderColor": "#e8e8e8", "borderStyle": "solid"}),
            html.Div(
                id="sites-selected-list",
                style={"marginTop": "4px"},
            ),
            html.Span(
                "clear selection",
                id="clear-selection-btn",
                n_clicks=0,
                style={"display": "none"},
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="selection-help-btn",
        close_id="selection-help-close-btn",
        popup_id="selection-help-popup",
        heading="Selection",
        body=[
            html.Strong("Tool Select", style={"fontSize": "11px"}),
            html.P("Pin Drop -- click on the map to drop a coordinate pin.", style=_HP),
            html.P("Point -- (default) select a monitoring site by clicking its marker.", style=_HP),
            html.Strong("Sites Selected", style={"fontSize": "11px"}),
            html.P("List of selected sites. Click the site name to display its timeseries.", style=_HP),
            html.P("Site -- the unique site identifier", style=_HP),
            html.P("Sparsity (%) -- % of rows with non-nan nitrate_concentration", style=_HP),
            html.P("Start -- earliest collection date", style=_HP),
            html.P("End -- last collection date", style=_HP),
            html.P("Lifespan -- difference in years between start and end", style={**_HP, "margin": "2px 0 0 0"}),
        ],
    )


def _agg_readout():
    """Static label naming the aggregations the stored series were built with.

    They used to be user-selectable (4 water x 4 rain), but the bundle stores one series pair per interval at fixed aggregations, so the choice is made at build time now. Showing it keeps the chart self-describing rather than leaving the reader to guess how a point was reduced.
    """
    intervals = bundle.series_intervals()
    nit, pre = (intervals[0][1], intervals[0][2]) if intervals else ("max", "mean")
    return [
        html.Label("Aggregation", style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"}),
        html.Div(
            f"nitrate: {nit} · precip: {pre}",
            style={"fontSize": "12px", "color": "#888", "padding": "5px 0"},
        ),
    ]


def _build_graph_display_section():
    details = html.Details(
        [
            html.Summary("Graph Display", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="graph-toggle",
                options=[
                    {"label": " Show graph", "value": "show"},
                    {"label": " Display seasons", "value": "seasons"},
                ],
                value=colors.default("graph-toggle"),
                style={**_CHECKBOX_STYLE, **_CHECKBOX_ROW, "marginTop": "6px"},
                labelStyle=_CHECKBOX_LABEL,
            ),
            # Interval options come from the BUNDLE, not a hard-coded list: build_bundle ships one
            # series pair per (site, interval) at fixed aggregations, so offering an interval it did
            # not build gives a control that changes nothing. The two Agg Method dropdowns are gone
            # for the same reason -- the aggregations are baked into the stored series.
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Aggregation Interval",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="aggregate-interval",
                                options=[{"label": i, "value": i} for i, _, _ in bundle.series_intervals()],
                                value=colors.default("aggregate-interval"),
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        _agg_readout(),
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginTop": "8px"},
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="graph-help-btn",
        close_id="graph-help-close-btn",
        popup_id="graph-help-popup",
        heading="Graph Display",
        body=[
            html.Strong("Show graph", style={"fontSize": "11px"}),
            html.P(
                "Toggles visibility of the timeseries chart overlay on the map. The chart appears when a monitoring site is selected.",
                style=_HP,
            ),
            html.Strong("Display seasons", style={"fontSize": "11px"}),
            html.P(
                "Overlays thin vertical lines at each solstice and equinox (Mar 21, Jun 21, Sep 21, Dec 21).", style=_HP
            ),
            html.Strong("Aggregate Interval", style={"fontSize": "11px"}),
            html.P(
                [
                    "Resampling period for the timeseries. See ",
                    html.A(
                        "the pandas user guide",
                        href="https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases",
                        target="_blank",
                        style={"color": "#3b82f6"},
                    ),
                    " on offset aliases.",
                ],
                style=_HP,
            ),
            html.Strong("Agg Method (Water / Rain)", style={"fontSize": "11px"}),
            html.P(
                "How values within each interval are combined — chosen separately for water (nitrate) and rain "
                "(precipitation): sum, mean, max, or min.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _forecast_years():
    """Forecast target years, read from the bundle manifest.

    This used to glob src/data/interim/weather_global_*.parquet, which meant building the layout
    touched the 4.6 GB weather store -- one of the import-time dependencies that made the app
    impossible to snapshot. The manifest declares the range instead (build_bundle.FORECAST_YEARS).
    """
    return bundle.forecast_years()


def _build_forecast_section():
    years = _forecast_years()
    default_year = 2017 if 2017 in years else (years[-1] if years else 2017)
    details = html.Details(
        [
            html.Summary("Forecast", style=_SECTION_LABEL_SUMMARY),
            html.Div(
                [
                    html.P(
                        "Drop a pin (Pin drop mode in Explore Tab), pick a year, and run the model to predict "
                        "nitrate at that ungauged point.",
                        style={**_HP, "marginTop": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Year:", style={"fontSize": "12px", "marginRight": "6px"}),
                            dcc.Dropdown(
                                id="forecast-year",
                                options=[{"label": str(y), "value": y} for y in years],
                                value=default_year,
                                clearable=False,
                                style={
                                    "width": "110px",
                                    "fontSize": "12px",
                                    "display": "inline-block",
                                    "verticalAlign": "middle",
                                },
                            ),
                        ],
                        style={"marginBottom": "8px", "marginTop": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Recall emphasis (β):",
                                style={"fontSize": "12px", "marginBottom": "2px", "display": "block"},
                            ),
                            dcc.Slider(
                                id="forecast-beta",
                                min=0.5,
                                max=4,
                                step=0.5,
                                value=2,
                                marks={0.5: "0.5", 1: "1", 2: "2", 3: "3", 4: "4"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.P(
                                "Higher β flags more violations at the cost of more false alarms.",
                                style={"fontSize": "11px", "color": "#888", "marginTop": "0", "marginBottom": "0"},
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Button("Run forecast", id="run-forecast-button", n_clicks=0, style={"fontSize": "12px"}),
                    dcc.Loading(
                        [
                            html.Div(id="forecast-results", style={"marginTop": "8px", "fontSize": "12px"}),
                            dcc.Graph(id="forecast-graph", style={"display": "none"}, config={"displayModeBar": False}),
                        ],
                        type="default",
                    ),
                    html.Div(
                        html.Button(
                            "Download figure", id="download-forecast-button", n_clicks=0, style={"fontSize": "12px"}
                        ),
                        id="download-forecast-row",
                        style={"display": "none"},
                    ),
                    dcc.Download(id="forecast-download"),
                    dcc.Store(id="forecast-download-fig"),
                ]
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="forecast-help-btn",
        close_id="forecast-help-close-btn",
        popup_id="forecast-help-popup",
        heading="Forecast",
        body=[
            html.Strong("Pin + year", style={"fontSize": "11px"}),
            html.P(
                "Drop a pin anywhere in Iowa (Pin drop selection mode) and choose a year. The model "
                "delineates the upstream basin at that point and predicts the daily nitrate "
                "concentration and violation probability there for the chosen year.",
                style=_HP,
            ),
            html.Strong("Recall emphasis (β)", style={"fontSize": "11px"}),
            html.P(
                "Sets the alarm cutoff on the predicted violation probability via the F-beta operating "
                "point. Higher β weights recall over precision (β=2 counts catching a violation "
                "4× as much as avoiding a false alarm), which lowers the cutoff so more days are "
                "flagged. Flagged (alarm) days are shaded on the P(violation) chart.",
                style=_HP,
            ),
            html.Strong("Results readout", style={"fontSize": "11px"}),
            html.P(
                "For the chosen β, reports the alarm-day count plus the model's honest catch rate "
                "(recall = % of real violations flagged) and false-alarm share (% of alarms that are "
                "false) — estimated on held-out sites at ~the base-rate prevalence.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_map_display_section():
    details = html.Details(
        [
            html.Summary("Map Display Options", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="hydro-toggle",
                options=[{"label": " Show rivers & lakes", "value": "show"}],
                value=colors.default("hydro-toggle"),
                style={**_CHECKBOX_STYLE, "marginTop": "6px"},
                labelStyle=_CHECKBOX_LABEL,
            ),
            # ── basin display ─────────────────────────────────────────────────
            html.Div("basin display", style=_SUBSECTION_LABEL),
            html.Div(
                [
                    dcc.Checklist(
                        id="basin-preferred-toggle",
                        options=[{"label": " Show basin", "value": "show"}],
                        value=colors.default("basin-preferred-toggle"),
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                    dcc.Checklist(
                        id="basin-all-toggle",
                        options=[{"label": " Show all basins", "value": "show"}],
                        value=colors.default("basin-all-toggle"),
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
                style=_CHECKBOX_ROW,
            ),
            # ── rain ──────────────────────────────────────────────────────────
            html.Div("rain", style=_SUBSECTION_LABEL),
            html.Div(
                [
                    dcc.Checklist(
                        id="rain-grid-toggle",
                        options=[{"label": " Show site rain grid", "value": "show"}],
                        value=colors.default("rain-grid-toggle"),
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
                style=_CHECKBOX_ROW,
            ),
            # The statewide "nitrogen surplus" block (Show Iowa heatmap + Year + Opacity) was
            # retired for the static build -- 18 pre-rendered PNGs, 31 MB, for one overlay. The YEAR
            # slider survived the cull because it is shared: it also picks which annual surplus /
            # crop layer colours the rain-grid cells above. It now lives with the rain grid, which
            # is its only remaining consumer. Opacity was heatmap-only and is gone with it.
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px", "marginTop": "4px"},
                children=[
                    html.Label(
                        "Year",
                        style={"fontSize": "11px", "color": "#555", "whiteSpace": "nowrap", "width": "40px"},
                    ),
                    html.Div(
                        dcc.Slider(
                            id="surplus-year-slider",
                            min=2000,
                            max=2017,
                            step=1,
                            value=colors.default("surplus-year-slider"),
                            marks=None,
                            # Shown only while dragging: always_visible parked a 20XX box under the slider permanently.
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="mouseup",
                        ),
                        style={"flex": "1", "marginTop": "-8px", "marginBottom": "-8px"},
                    ),
                ],
            ),
        ],
        open=True,
    )
    return _wrap_with_help(
        details,
        btn_id="map-display-help-btn",
        close_id="map-display-help-close-btn",
        popup_id="map-display-help-popup",
        heading="Map Display Options",
        body=[
            html.Strong("Show rivers & lakes", style={"fontSize": "11px"}),
            html.P("Toggles the NHD hydrography overlay (streams, rivers, and waterbodies).", style=_HP),
            html.Strong("Show basin", style={"fontSize": "11px"}),
            html.P(
                "Displays the preferred drainage basin for each selected site (purple). See Basin Editor in the Debug menu to compare individual basin types.",
                style=_HP,
            ),
            html.Strong("Show all basins", style={"fontSize": "11px"}),
            html.P("Displays the dissolved union of all monitoring site drainage basins.", style=_HP),
            html.Strong("Show site rain grid", style={"fontSize": "11px"}),
            html.P(
                "Displays the IEM precipitation grid cell positions for the active graph site.",
                style=_HP,
            ),
            html.Strong("Year", style={"fontSize": "11px"}),
            html.P(
                "Picks which annual surplus / crop layer (2000-2017) colours the rain-grid cells.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_presentation_section():
    """Display options tuned for screen-recording / presentation. Currently the grid-cell colour
    mode; more toggles will be added here. Controls read their defaults from colors.default()."""
    return html.Details(
        [
            html.Summary("Presentation Display Options", style=_SECTION_LABEL_SUMMARY),
            html.Div("grid cell color", style=_SUBSECTION_LABEL),
            dcc.RadioItems(
                id="grid-color-mode",
                options=[
                    {"label": " Nitrogen surplus", "value": "surplus"},
                    {"label": " Dominant crop", "value": "crop"},
                ],
                value=colors.default("grid-color-mode"),
                style={"fontSize": "13px", "marginTop": "4px"},
            ),
        ],
        open=True,
    )


def _build_map_layers_section():
    details = html.Details(
        [
            html.Summary("Map Layers", style=_SECTION_LABEL_SUMMARY),
            dcc.RadioItems(
                id="tile-selector",
                options=[
                    {"label": " Street", "value": "street"},
                    {"label": " Satellite", "value": "satellite"},
                    {"label": " Humanitarian", "value": "humanitarian"},
                ],
                value=colors.default("tile-selector"),
                style={"fontSize": "13px", "marginTop": "6px"},
            ),
        ],
        open=False,
    )
    return _wrap_with_help(
        details,
        btn_id="map-layers-help-btn",
        close_id="map-layers-help-close-btn",
        popup_id="map-layers-help-popup",
        heading="Map Layers",
        body=[
            html.Strong("Street", style={"fontSize": "11px"}),
            html.P("Standard OpenStreetMap tiles.", style=_HP),
            html.Strong("Satellite", style={"fontSize": "11px"}),
            html.P("ESRI World Imagery satellite tiles.", style=_HP),
            html.Strong("Humanitarian", style={"fontSize": "11px"}),
            html.P(
                "OpenStreetMap Humanitarian style — high contrast, designed for crisis mapping.",
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
    )


def _build_debugging_section():
    return html.Details(
        [
            html.Summary("Debugging", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="iem-bbox-toggle",
                options=[{"label": " Show IEM footprint", "value": "show"}],
                value=colors.default("iem-bbox-toggle"),
                style={"fontSize": "13px", "marginTop": "6px"},
            ),
            # live map-view readout (updates as you pan/zoom) -- handy for choosing IOWA_CENTER/ZOOM
            html.Div(
                id="map-view-readout",
                style={"fontSize": "12px", "fontFamily": "monospace", "color": "#555", "marginTop": "8px"},
            ),
            # set the view directly: zoom + center (lat, lon) -> Apply flies the map there
            html.Div(
                [
                    dcc.Input(
                        id="map-zoom-input",
                        type="number",
                        min=1,
                        max=18,
                        step=1,
                        placeholder="zoom",
                        style={"width": "56px", "fontSize": "12px"},
                    ),
                    dcc.Input(
                        id="map-center-input",
                        type="text",
                        placeholder="lat, lon",
                        style={"width": "120px", "fontSize": "12px"},
                    ),
                    html.Button("Apply", id="map-view-apply", n_clicks=0, style={"fontSize": "12px"}),
                ],
                style={"display": "flex", "gap": "4px", "alignItems": "center", "marginTop": "6px"},
            ),
        ],
        open=False,
    )


def layout():
    """Return the full-viewport layout: map (80 %) left, tools panel (20 %) right."""
    return html.Div(
        style={"display": "flex", "height": "100vh", "overflow": "hidden"},
        children=[
            dcc.Store(id="active-menu", data="explore"),
            # Constants the clientside callbacks read, shipped from Python so there is exactly one
            # definition of each. See widget/assets/clientside/.
            dcc.Store(id="tile-urls", data=TILE_URLS),
            dcc.Store(id="ui-consts", data=clientside_consts()),
            *[dcc.Store(id=f"basin-type-{b}", data=b) for b in (1, 2, 3)],
            dcc.Store(id="preferred-basin-version", data=0),
            # ── Map (80 %) ─────────────────────────────────────────────────
            html.Div(
                style={"position": "relative", "flex": "0 0 68%", "width": "68%"},
                children=[
                    dl.Map(
                        id="map",
                        center=IOWA_CENTER,
                        zoom=IOWA_ZOOM,
                        zoomControl=False,
                        children=[
                            dl.TileLayer(id="tile-layer"),
                            dl.GeoJSON(
                                data=_IOWA_MASK,
                                options={
                                    "style": {
                                        "fillColor": "black",
                                        "fillOpacity": 0.2,
                                        "weight": 0,
                                        "interactive": False,
                                    }
                                },
                            ),
                            dl.GeoJSON(
                                data=iowa_geojson,
                                # interactive:False so this statewide outline's transparent fill
                                # doesn't capture mouse events over all of Iowa (it would block
                                # hover on the rain-grid cells beneath it).
                                options={
                                    "style": {"color": "#555", "weight": 2, "fillOpacity": 0, "interactive": False}
                                },
                            ),
                            dl.LayerGroup(id="iem-bbox-layer"),
                            dl.Pane(name="hydro-pane", style={"zIndex": 410}),
                            dl.Pane(name="rain-grid-pane", style={"zIndex": 415}),
                            dl.Pane(name="basin-pane", style={"zIndex": 420}),
                            dl.Pane(name="sites-pane", style={"zIndex": 430}),
                            dl.LayerGroup(id="mapunit-layer"),
                            dl.LayerGroup(id="hydro-layer"),
                            dl.LayerGroup(id="rain-grid-layer"),
                            dl.LayerGroup(id="upstream-layer"),
                            dl.LayerGroup(id="basin1-layer"),
                            dl.LayerGroup(id="basin2-layer"),
                            dl.LayerGroup(id="basin3-layer"),
                            dl.LayerGroup(id="pin-basin-layer"),
                            dl.LayerGroup(id="pin-basin-v3-layer"),
                            dl.LayerGroup(id="iwqis-layer"),
                            dl.LayerGroup(id="marker-layer"),
                            dl.LayerGroup(id="forecast-layer"),
                        ],
                        style={"height": "100vh", "width": "100%"},
                    ),
                    html.Div(
                        id="map-graph-overlay",
                        style=_GRAPH_OVERLAY_HIDDEN,
                        children=[
                            html.Button(
                                "×",
                                id="close-graph-btn",
                                n_clicks=0,
                                style={
                                    "position": "absolute",
                                    "top": "4px",
                                    "right": "6px",
                                    "background": "none",
                                    "border": "none",
                                    "fontSize": "18px",
                                    "lineHeight": "1",
                                    "cursor": "pointer",
                                    "color": "#666",
                                    "padding": "2px 6px",
                                    "zIndex": 1,
                                },
                            ),
                            dcc.Graph(
                                id="timeseries-graph",
                                style={"width": "100%", "height": "100%"},
                                config={"responsive": True, "scrollZoom": True},
                                figure=go.Figure(),
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "position": "absolute",
                            "bottom": "12px",
                            "left": "12px",
                            "zIndex": 600,
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "6px",
                            "alignItems": "flex-start",
                        },
                        children=[
                            # Both legends are rendered up front and hidden; the clientside callback
                            # flips `style` rather than building a tree in JS. The output space here
                            # is exactly two variants plus "off", so pre-rendering is strictly
                            # simpler than constructing either one in JavaScript.
                            html.Div(_crop_legend(), id="grid-color-legend-crop", style={"display": "none"}),
                            html.Div(_surplus_legend(), id="grid-color-legend-surplus", style={"display": "none"}),
                            _build_legend(),
                        ],
                    ),
                ],
            ),
            # ── Tools panel (20 %) ─────────────────────────────────────────
            html.Div(
                id="tools-panel",
                style=_PANEL_STYLE,
                children=[
                    # Menu selector tabs (full-bleed, no side padding inherited)
                    html.Div(
                        style={"display": "flex", "margin": "0 -12px 12px -12px"},
                        children=[
                            html.Button("Explore", id="menu-tab-explore", n_clicks=0, style=_MENU_TAB_ACTIVE),
                            html.Button("Forecast", id="menu-tab-forecast", n_clicks=0, style=_MENU_TAB_INACTIVE),
                            html.Button("Debug", id="menu-tab-debug", n_clicks=0, style=_MENU_TAB_INACTIVE),
                        ],
                    ),
                    # Explore Menu
                    html.Div(
                        id="explore-menu-content",
                        style={"display": "block"},
                        children=[
                            _build_selection_section(),
                            _hr(),
                            _build_graph_display_section(),
                            _hr(),
                            _build_map_display_section(),
                            _hr(),
                            _build_presentation_section(),
                            _hr(),
                            _build_map_layers_section(),
                        ],
                    ),
                    # Forecast Menu
                    html.Div(
                        id="forecast-menu-content",
                        style={"display": "none"},
                        children=[
                            _build_forecast_section(),
                        ],
                    ),
                    # Debug Menu
                    html.Div(
                        id="debug-menu-content",
                        style={"display": "none"},
                        children=[
                            _hr(),
                            # Wrapped here rather than inside basin_editor: map_common imports that
                            # module, so it cannot import _wrap_with_help back.
                            _wrap_with_help(
                                basin_editor.layout(),
                                btn_id="basin-editor-help-btn",
                                close_id="basin-editor-help-close-btn",
                                popup_id="basin-editor-help-popup",
                                heading="Basin Editor",
                                body=[
                                    html.P(
                                        "Used to manually configure and edit basins during development.",
                                        style=_HP,
                                    ),
                                    html.P(
                                        'Kept for deployment as it provides a "search by name" box for sites and '
                                        "displays useful basin information.",
                                        style={**_HP, "margin": "2px 0 0 0"},
                                    ),
                                ],
                            ),
                            _hr(),
                            _build_debugging_section(),
                        ],
                    ),
                ],
            ),
        ],
    )


# ── Rain-grid Voronoi cells (rendered as one interactive GeoJSON layer) ────────
# Cells are coloured by surplus (matching the surplus heatmap PNGs), highlight on
# hover, and bind a tooltip with surplus / total-N / crop stats. The style,
# hover-style and tooltip-binding run client-side as JS (one layer scales far
# better than thousands of per-cell components).
