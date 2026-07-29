"""Map panel: registers every map callback and re-exports layout() (so layout.py can still call
map_panel.layout()). Shared helpers/constants live in map_common; the UI builders in map_layout.
Split purely for navigability -- behavior is unchanged."""

from .map_common import *  # noqa: F401,F403
from .map_layout import layout, _crop_legend, _surplus_legend  # noqa: F401


def register_callbacks(app):
    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="mapViewReadout"),
        Output("map-view-readout", "children"),
        Input("map", "zoom"),
        Input("map", "center"),
    )

    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="mapViewport"),
        Output("map", "viewport"),
        Input("map-view-apply", "n_clicks"),
        State("map-zoom-input", "value"),
        State("map-center-input", "value"),
        State("map", "center"),
        State("map", "zoom"),
        prevent_initial_call=True,
    )

    # One JS function, five registrations. The close button's id is passed as a literal State so the
    # shared implementation can tell an open click from a close click.
    for _popup_id, _btn_id, _close_id in [
        ("selection-help-popup", "selection-help-btn", "selection-help-close-btn"),
        ("graph-help-popup", "graph-help-btn", "graph-help-close-btn"),
        ("forecast-help-popup", "forecast-help-btn", "forecast-help-close-btn"),
        ("map-display-help-popup", "map-display-help-btn", "map-display-help-close-btn"),
        ("map-layers-help-popup", "map-layers-help-btn", "map-layers-help-close-btn"),
    ]:
        app.clientside_callback(
            ClientsideFunction(namespace="ui", function_name="helpPopup"),
            Output(_popup_id, "style"),
            Input(_btn_id, "n_clicks"),
            Input(_close_id, "n_clicks"),
            State("ui-consts", "data"),
            prevent_initial_call=True,
        )

    # The six data columns come from sites.json, which build_bundle joined once; the live version called access.get_basin_area per row, and that is a full site-view build each time.
    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="sitesTable"),
        Output("sites-selected-list", "children"),
        Output("clear-selection-btn", "style"),
        Input("selected-site", "data"),
        Input("active-graph-site", "data"),
        State("ui-consts", "data"),
    )

    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="selectedSites"),
        Output("selected-site", "data"),
        Input({"type": "iwqis-marker", "index": ALL}, "n_clicks"),
        Input({"type": "remove-site-btn", "index": ALL}, "n_clicks"),
        Input("clear-selection-btn", "n_clicks"),
        State("selected-site", "data"),
        State("active-menu", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="activeGraphSite"),
        Output("active-graph-site", "data"),
        Input({"type": "graph-site-btn", "index": ALL}, "n_clicks"),
        Input("selected-site", "data"),
        State("active-graph-site", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="graphOverlay"),
        Output("map-graph-overlay", "style"),
        Input("selected-site", "data"),
        Input("close-graph-btn", "n_clicks"),
        Input("graph-toggle", "value"),
        State("ui-consts", "data"),
        prevent_initial_call=True,
    )

    # TILE_URLS travels as a State rather than being duplicated in JS, so adding a basemap stays a
    # one-line change in map_common and cannot leave the two definitions disagreeing.
    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="tileUrl"),
        Output("tile-layer", "url"),
        Input("tile-selector", "value"),
        State("tile-urls", "data"),
    )

    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="iemBbox"),
        Output("iem-bbox-layer", "children"),
        Input("iem-bbox-toggle", "value"),
        State("ui-consts", "data"),
    )

    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="hydro"),
        Output("hydro-layer", "children"),
        Input("hydro-toggle", "value"),
        State("ui-consts", "data"),
    )

    # basin_style() already carries `pane`, so the emitted dl.GeoJSON needs only a url + style;
    # Leaflet fetches and parses the polygon itself and no geometry is touched in JS.
    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="upstream"),
        Output("upstream-layer", "children"),
        Input("basin-preferred-toggle", "value"),
        Input("basin-all-toggle", "value"),
        Input("selected-site", "data"),
        Input("preferred-basin-version", "data"),
        State("ui-consts", "data"),
    )

    # One JS function, three registrations -- the basin version rides along as a literal State so
    # the implementation is shared (this replaces the _register_basin_renderer closure factory).
    for _bt in (1, 2, 3):
        app.clientside_callback(
            ClientsideFunction(namespace="layers", function_name="basinVersion"),
            Output(f"basin{_bt}-layer", "children"),
            Input(f"basin{_bt}-toggle", "value"),
            Input("selected-site", "data"),
            State(f"basin-type-{_bt}", "data"),
            State("ui-consts", "data"),
        )

    @app.callback(
        Output("pin-basin-layer", "children"),
        Input("pin-basin-v1-toggle", "value"),
        Input("region-geom", "data"),
    )
    def render_pin_basin_v1(toggle, region_geom):
        if "show" not in toggle:
            return []
        if not region_geom or region_geom.get("type") != "Point":
            return []
        lng, lat = region_geom["coordinates"]
        try:
            geojson = delineate_basin_for_pin(lat, lng)
            return [dl.GeoJSON(data=geojson, options={"style": colors.pin_basin_style("v1")})]
        except Exception:
            return []

    @app.callback(
        Output("pin-basin-v3-layer", "children"),
        Input("pin-basin-v3-toggle", "value"),
        Input("region-geom", "data"),
    )
    def render_pin_basin_v3(toggle, region_geom):
        if "show" not in toggle:
            return []
        if not region_geom or region_geom.get("type") != "Point":
            return []
        lng, lat = region_geom["coordinates"]
        try:
            geojson = delineate_basin_v3_for_pin(lat, lng)
            return [dl.GeoJSON(data=geojson, options={"style": colors.pin_basin_style("v3")})]
        except Exception:
            return []

    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="gridColorLegend"),
        Output("grid-color-legend-crop", "style"),
        Output("grid-color-legend-surplus", "style"),
        Input("rain-grid-toggle", "value"),
        Input("grid-color-mode", "value"),
        Input("active-graph-site", "data"),
    )

    # _rain_grid_features in map_common stays as the reference implementation: it is what the parity spot-check compares the browser's cells against, colour and tooltip alike.
    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="rainGrid"),
        Output("rain-grid-layer", "children"),
        Input("rain-grid-toggle", "value"),
        Input("active-graph-site", "data"),
        Input("surplus-year-slider", "value"),
        Input("grid-color-mode", "value"),
        State("ui-consts", "data"),
    )

    # The statewide Iowa surplus heatmap (render_iowa_surplus_heatmap) was retired for the static
    # build: 18 pre-rendered PNGs at ~1.7 MB each, 31 MB in total, for a single overlay -- by a wide
    # margin the largest asset in the bundle. The surplus-year-slider it shared with the rain grid
    # survives; only the heatmap toggle and its opacity slider are gone.

    # make_iwqis_markers in map_common stays as the reference implementation for the parity check.
    app.clientside_callback(
        ClientsideFunction(namespace="layers", function_name="iwqisSites"),
        Output("iwqis-layer", "children"),
        Input("selected-site", "data"),
        Input("basin-review-flagged-only", "value"),
        Input("basin-review-unreviewed-only", "value"),
        State("ui-consts", "data"),
    )

    # The pin SNAPS. A forecast is computed at a reach outlet, not at the click, and restricting to
    # stream order >= 3 puts that outlet a median 1.5 km away -- so the marker moves to the modelled
    # location and draws a connector back to where the user clicked. region-geom carries both.
    app.clientside_callback(
        ClientsideFunction(namespace="forecast", function_name="regionGeom"),
        Output("region-geom", "data"),
        Output("marker-layer", "children"),
        Input("map", "clickData"),
        State("selection-mode", "value"),
        State("ui-consts", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="ui", function_name="switchMenu"),
        Output("explore-menu-content", "style"),
        Output("forecast-menu-content", "style"),
        Output("debug-menu-content", "style"),
        Output("menu-tab-explore", "style"),
        Output("menu-tab-forecast", "style"),
        Output("menu-tab-debug", "style"),
        Output("active-menu", "data"),
        Output("selected-site", "data", allow_duplicate=True),
        Input("menu-tab-explore", "n_clicks"),
        Input("menu-tab-forecast", "n_clicks"),
        Input("menu-tab-debug", "n_clicks"),
        State("selected-site", "data"),
        State("active-graph-site", "data"),
        State("ui-consts", "data"),
        prevent_initial_call=True,
    )

    basin_editor.register_callbacks(app)
