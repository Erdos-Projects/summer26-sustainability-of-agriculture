"""Map panel layout: the section builders + the main layout(). Split from map_panel.py.

Shared constants, styles, and helpers come from map_common (star-imported)."""

from .map_common import *  # noqa: F401,F403


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
                            {"label": " Area", "value": "area"},
                        ],
                        value="point",
                        style={"display": "flex", "gap": "10px"},
                        labelStyle={"fontSize": "13px", "display": "flex", "alignItems": "center", "cursor": "pointer"},
                        inputStyle={"marginRight": "3px"},
                    ),
                    html.Div(
                        id="area-tool-container",
                        style={"display": "none"},
                        children=[
                            _draw_btn(_rect_icon(), "draw-rect-btn", "Draw rectangle"),
                            _draw_btn(_poly_icon(), "draw-poly-btn", "Draw polygon"),
                            _draw_btn(_trash_icon(), "draw-delete-btn", "Clear drawn area"),
                        ],
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
            html.P("Area -- draw a rectangle or polygon to bulk select monitoring sites inside it.", style=_HP),
            html.Strong("Sites Selected", style={"fontSize": "11px"}),
            html.P("List of selected sites. Click the site name to display its timeseries.", style=_HP),
            html.P("Site -- the unique site identifier", style=_HP),
            html.P("Sparsity (%) -- % of rows with non-nan nitrate_concentration", style=_HP),
            html.P("Start -- earliest collection date", style=_HP),
            html.P("End -- last collection date", style=_HP),
            html.P("Lifespan -- difference in years between start and end", style={**_HP, "margin": "2px 0 0 0"}),
        ],
    )


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
                value=["show"],
                style={**_CHECKBOX_STYLE, **_CHECKBOX_ROW, "marginTop": "6px"},
                labelStyle=_CHECKBOX_LABEL,
            ),
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
                                options=[
                                    {"label": "1D", "value": "1D"},
                                    {"label": "3D", "value": "3D"},
                                    {"label": "1W", "value": "1W"},
                                    {"label": "2W", "value": "2W"},
                                    {"label": "1MS", "value": "1MS"},
                                    {"label": "3MS", "value": "3MS"},
                                    {"label": "1YS", "value": "1YS"},
                                ],
                                value="1D",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Agg Method (Water)",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="agg-func-water",
                                options=[
                                    {"label": "sum", "value": "sum"},
                                    {"label": "mean", "value": "mean"},
                                    {"label": "max", "value": "max"},
                                    {"label": "min", "value": "min"},
                                ],
                                value="mean",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Agg Method (Rain)",
                                style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="agg-func-rain",
                                options=[
                                    {"label": "sum", "value": "sum"},
                                    {"label": "mean", "value": "mean"},
                                    {"label": "max", "value": "max"},
                                    {"label": "min", "value": "min"},
                                ],
                                value="sum",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
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
    """Weather years available for a virtual-site forecast (trimmed so target_year keeps its
    ±2-month lookback buffer)."""
    yrs = sorted(access._weather_global_files())
    return yrs[1:-1] if len(yrs) > 2 else yrs


def _build_forecast_section():
    years = _forecast_years()
    default_year = 2017 if 2017 in years else (years[-1] if years else 2017)
    details = html.Details(
        [
            html.Summary("Forecast", style=_SECTION_LABEL_SUMMARY),
            html.Div(
                [
                    html.P(
                        "Drop a pin (Pin drop mode), pick a year, and run the model to predict "
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
                                style={"width": "110px", "fontSize": "12px", "display": "inline-block",
                                       "verticalAlign": "middle"},
                            ),
                        ],
                        style={"marginBottom": "8px", "marginTop": "6px"},
                    ),
                    html.Button("Run forecast", id="run-forecast-button", n_clicks=0, style={"fontSize": "12px"}),
                    dcc.Loading(
                        [
                            html.Div(id="forecast-results", style={"marginTop": "8px", "fontSize": "12px"}),
                            dcc.Graph(id="forecast-graph", style={"display": "none"}, config={"displayModeBar": False}),
                        ],
                        type="default",
                    ),
                ]
            ),
        ],
        open=False,
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
        ],
    )


def _build_map_display_section():
    details = html.Details(
        [
            html.Summary("Map Display Options", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="hydro-toggle",
                options=[{"label": " Show rivers & lakes", "value": "show"}],
                value=["show"],
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
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                    dcc.Checklist(
                        id="basin-all-toggle",
                        options=[{"label": " Show all basins", "value": "show"}],
                        value=[],
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
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
                style=_CHECKBOX_ROW,
            ),
            # ── nitrogen surplus ──────────────────────────────────────────────
            html.Div("nitrogen surplus", style=_SUBSECTION_LABEL),
            html.Div(
                style={"display": "flex", "gap": "10px", "marginTop": "4px"},
                children=[
                    # Left column — checkboxes
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                        children=[
                            dcc.Checklist(
                                id="surplus-heatmap-toggle",
                                options=[{"label": " Show site heatmap", "value": "show"}],
                                value=[],
                                style=_CHECKBOX_STYLE,
                                labelStyle=_CHECKBOX_LABEL,
                            ),
                            dcc.Checklist(
                                id="iowa-surplus-heatmap-toggle",
                                options=[{"label": " Show Iowa heatmap", "value": "show"}],
                                value=[],
                                style=_CHECKBOX_STYLE,
                                labelStyle=_CHECKBOX_LABEL,
                            ),
                        ],
                    ),
                    # Right column — sliders
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                children=[
                                    html.Label(
                                        "Year",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#555",
                                            "whiteSpace": "nowrap",
                                            "width": "40px",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="surplus-year-slider",
                                            min=2000,
                                            max=2017,
                                            step=1,
                                            value=2017,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            updatemode="mouseup",
                                        ),
                                        style={"flex": "1", "marginTop": "-8px", "marginBottom": "-8px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                children=[
                                    html.Label(
                                        "Opacity",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#555",
                                            "whiteSpace": "nowrap",
                                            "width": "40px",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="surplus-opacity-slider",
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=0.8,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            updatemode="mouseup",
                                        ),
                                        style={"flex": "1", "marginTop": "-8px", "marginBottom": "-8px"},
                                    ),
                                ],
                            ),
                        ],
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
                style={**_HP, "margin": "2px 0 0 0"},
            ),
        ],
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
                    {"label": " Watercolor", "value": "watercolor"},
                ],
                value="street",
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
            html.P("OpenStreetMap Humanitarian style — high contrast, designed for crisis mapping.", style=_HP),
            html.Strong("Watercolor", style={"fontSize": "11px"}),
            html.P("Stadia/Stamen watercolor tiles — artistic style.", style={**_HP, "margin": "2px 0 0 0"}),
        ],
    )


def _build_debugging_section():
    return html.Details(
        [
            html.Summary("Debugging", style=_SECTION_LABEL_SUMMARY),
            dcc.Checklist(
                id="iem-bbox-toggle",
                options=[{"label": " Show IEM footprint", "value": "show"}],
                value=[],
                style={"fontSize": "13px", "marginTop": "6px"},
            ),
        ],
        open=False,
    )


def layout():
    """Return the full-viewport layout: map (80 %) left, tools panel (20 %) right."""
    return html.Div(
        style={"display": "flex", "height": "100vh", "overflow": "hidden"},
        children=[
            dcc.Store(id="active-area-tool", data=None),
            dcc.Store(id="active-menu", data="explore"),
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
                            dl.ZoomControl(position="bottomleft"),
                            dl.GeoJSON(
                                data=iowa_geojson,
                                # interactive:False so this statewide outline's transparent fill
                                # doesn't capture mouse events over all of Iowa (it would block
                                # hover on the rain-grid cells beneath it).
                                options={"style": {"color": "#555", "weight": 2, "fillOpacity": 0, "interactive": False}},
                            ),
                            dl.LayerGroup(id="iem-bbox-layer"),
                            dl.Pane(name="hydro-pane", style={"zIndex": 410}),
                            dl.Pane(name="surplus-pane", style={"zIndex": 413}),
                            dl.Pane(name="rain-grid-pane", style={"zIndex": 415}),
                            dl.Pane(name="basin-pane", style={"zIndex": 420}),
                            dl.Pane(name="sites-pane", style={"zIndex": 430}),
                            dl.LayerGroup(id="mapunit-layer"),
                            dl.LayerGroup(id="hydro-layer"),
                            dl.ImageOverlay(
                                id="surplus-image-overlay",
                                url=_TRANSPARENT_PNG,
                                bounds=_IOWA_BOUNDS,
                                opacity=0,
                                pane="surplus-pane",
                            ),
                            dl.ImageOverlay(
                                id="iowa-surplus-image-overlay",
                                url=_TRANSPARENT_PNG,
                                bounds=_IOWA_BOUNDS,
                                opacity=0,
                                pane="surplus-pane",
                            ),
                            dl.LayerGroup(id="rain-grid-layer"),
                            dl.LayerGroup(id="upstream-layer"),
                            dl.LayerGroup(id="basin1-layer"),
                            dl.LayerGroup(id="basin2-layer"),
                            dl.LayerGroup(id="basin3-layer"),
                            dl.LayerGroup(id="pin-basin-layer"),
                            dl.LayerGroup(id="pin-basin-v3-layer"),
                            dl.LayerGroup(id="iwqis-layer"),
                            dl.LayerGroup(id="marker-layer"),
                            dl.FeatureGroup(
                                [
                                    dl.EditControl(
                                        id="edit-control",
                                        position="bottomleft",
                                        draw=DRAW_TOOLS,
                                        edit={"edit": False, "remove": True},
                                    )
                                ]
                            ),
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
                            html.Div(id="debug-sites-table", style={"marginBottom": "8px"}),
                            _hr(),
                            basin_editor.layout(),
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
