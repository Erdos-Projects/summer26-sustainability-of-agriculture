"""Basin Editor panel section and callbacks (READ-ONLY).

Owns the Basin Editor block in the Debug menu: site selection, flag display, site metadata, and the basin display table. The map-layer callbacks that *render* basin polygons stay in map_panel, because they output to map LayerGroups.

READ-ONLY. The "Set preferred basin" radio and its Confirm button are gone: they called access.update_basin, which writes preferred_basin.csv, and a static site cannot write. They were already inert behind a deploy safeguard (colors.DEBUG_MODE_ON, now removed with them), so nothing that worked before stops working -- the control that did nothing is simply no longer drawn. Curating a basin selection is a local task against the Python app plus the builder, not something to do from a published page.

Layout follows the newer sustag design: a v1/v2/v3 x site/pin table with each checkbox carrying its basin's area, which makes the cross-check comparison (does the snap agree with the containing catchment and the D8 raster?) readable at a glance instead of requiring three toggles and a mental diff.
"""

from dash import ClientsideFunction, Input, Output, State, html, dcc

# ── Style constants ───────────────────────────────────────────────────────────
# Duplicated from map_panel to avoid a circular import.

_SECTION_LABEL_SUMMARY = {
    "fontSize": "11px",
    "fontWeight": "bold",
    "color": "#888",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "cursor": "pointer",
    "userSelect": "none",
}
_SUBSECTION_LABEL = {
    "fontSize": "10px",
    "fontWeight": "600",
    "color": "#bbb",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "marginTop": "8px",
    "marginBottom": "3px",
    "borderBottom": "1px solid #eee",
    "paddingBottom": "2px",
}
_CHECKBOX_STYLE = {"fontSize": "13px"}
_CHECKBOX_LABEL = {"fontSize": "13px", "display": "flex", "alignItems": "center", "cursor": "pointer"}
_AREA_LABEL_STYLE = {"fontSize": "10px", "color": "#888", "marginLeft": "5px", "minWidth": "48px"}

# Delineation rows. v0 (auth) is omitted: it is a uid lookup with no pin-drop equivalent, and it is only ever the PREFERRED basin, reachable through the "Show basin" layer in the Explore tab.
_BASIN_METHODS = [
    ("v1", "snap"),  # nearest-flowline snap -- the shipped delineation
    ("v2", "catch"),  # containing catchment (/comid/position) -- cross-check
    ("v3", "D8"),  # raster flood-fill -- cross-check
]

# Which rows have a live pin-drop equivalent. Pin delineation is DEFERRED with the forecast, so these cells render but stay inert for now; v2 has no pin layer in this repo at all.
_PIN_METHODS = {"v1", "v3"}

_FLAG_LABELS = {
    "flag_area": "large area",
    "flag_river": "near major river",
    "flag_not_contained": "not contained",
    "flag_basin1_over_basin0": "snap over auth",
    "flag_area_mismatch_v2": "v1/v2 area mismatch",
    "flag_area_mismatch_v3": "v1/v3 area mismatch",
}

# The two readout lines. Only the colours that VARY with the data are consts the JS picks between; the rest are set on the spans at layout time and never written again.
_MUTED = {"color": "#555", "marginRight": "8px"}
_DIM = {"color": "#888"}
_LOCATION_KNOWN = {"color": "#555", "marginRight": "8px"}
_LOCATION_UNKNOWN = {**_LOCATION_KNOWN, "color": "#999"}
_FLAGS_SOME = {"color": "#dc2626"}
_FLAGS_NONE = {"color": "#16a34a"}


def _clientside_consts():
    """Styles and labels the basin-editor readouts need in the browser.

    Shipped in this panel's own Store rather than folded into ui-consts so the section stays self-contained; map_common owns the map's constants and this owns its own.
    """
    return {
        "location_known": _LOCATION_KNOWN,
        "location_unknown": _LOCATION_UNKNOWN,
        "flags_some": _FLAGS_SOME,
        "flags_none": _FLAGS_NONE,
        "flag_labels": _FLAG_LABELS,
    }


def _display_cell(toggle_id: str, area_id: str) -> html.Div:
    """A display checkbox with an area label to its right (filled by callback)."""
    return html.Div(
        [
            dcc.Checklist(
                id=toggle_id,
                options=[{"label": "", "value": "show"}],
                value=[],
                style={"fontSize": "13px", "margin": "0"},
                labelStyle={"margin": "0"},
            ),
            html.Span("", id=area_id, style=_AREA_LABEL_STYLE),
        ],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-start"},
    )


def _basin_display_table() -> html.Table:
    """    Rows v1/v2/v3 x columns site/pin. Site cells reuse basin{1,2,3}-toggle (map_panel renders the stored parquets); pin cells drive the live pin-drop overlays. Each checkbox shows the area of its basin (km2) to the right -- site areas from preferred_basin.csv, pin areas on drop."""
    hdr_style = {"fontSize": "10px", "color": "#888", "fontWeight": "600", "padding": "2px 8px", "textAlign": "center"}
    row_label_style = {"fontSize": "11px", "color": "#555", "padding": "2px 8px", "whiteSpace": "nowrap"}
    header = html.Tr([html.Th("", style=hdr_style), html.Th("site", style=hdr_style), html.Th("pin", style=hdr_style)])
    body = []
    for i, (v, label) in enumerate(_BASIN_METHODS, start=1):
        pin_cell = (
            _display_cell(f"pin-basin-{v}-toggle", f"pin-basin-{v}-area")
            if v in _PIN_METHODS
            else html.Span("—", style={"fontSize": "11px", "color": "#ccc"})
        )
        body.append(
            html.Tr(
                [
                    html.Td(f"{v} ({label})", style=row_label_style),
                    html.Td(_display_cell(f"basin{i}-toggle", f"basin{i}-area"), style={"padding": "2px 8px"}),
                    html.Td(pin_cell, style={"padding": "2px 8px"}),
                ]
            )
        )
    return html.Table([html.Thead(header), html.Tbody(body)], style={"borderCollapse": "collapse"})


def layout():
    return html.Details(
        [
            html.Summary("Basin Editor", style=_SECTION_LABEL_SUMMARY),
            dcc.Store(id="basin-editor-consts", data=_clientside_consts()),
            # ──
            # Site filter + dropdown + flags ────────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "12px", "marginTop": "6px"},
                children=[
                    dcc.Checklist(
                        id="basin-review-flagged-only",
                        options=[{"label": " Flagged sites only", "value": "on"}],
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                    dcc.Checklist(
                        id="basin-review-unreviewed-only",
                        options=[{"label": " Unreviewed only", "value": "on"}],
                        value=[],
                        style=_CHECKBOX_STYLE,
                        labelStyle=_CHECKBOX_LABEL,
                    ),
                ],
            ),
            dcc.Dropdown(
                id="basin-review-site-dropdown",
                placeholder="Select site...",
                clearable=True,
                style={"fontSize": "12px", "marginTop": "4px"},
            ),
            # Both readouts are pre-rendered skeletons: the shape never varies, so the callbacks write the strings (and the one colour that depends on the data) instead of a tree.
            html.Div(
                id="basin-review-site-meta",
                style={"fontSize": "11px", "marginTop": "4px"},
                children=[
                    html.Span(id="basin-review-location", style=_LOCATION_UNKNOWN),
                    html.Span(id="basin-review-detail", style=_DIM),
                ],
            ),
            html.Div(
                id="basin-review-flags",
                style={"fontSize": "11px", "marginTop": "4px"},
                children=[
                    html.Span(id="basin-review-status", style=_MUTED),
                    html.Span(id="basin-review-flag-text", style=_FLAGS_NONE),
                ],
            ),
            # ──
            # Basin display table ──────────────────────────────────────────
            html.Div(
                style={"marginTop": "8px"},
                children=[
                    html.Div("Basin display", style=_SUBSECTION_LABEL),
                    _basin_display_table(),
                ],
            ),
        ],
        open=True,
    )


def register_callbacks(app):
    """Everything here reads sites.json, which build_bundle joins from the site, stats and basin metadata tables -- so the whole panel comes out of one cached fetch rather than three parquet reads per interaction."""
    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="basinDropdown"),
        Output("basin-review-site-dropdown", "options"),
        Output("basin-review-site-dropdown", "value"),
        Input("basin-review-flagged-only", "value"),
        Input("basin-review-unreviewed-only", "value"),
        Input("preferred-basin-version", "data"),
        Input("selected-site", "data"),
        State("active-menu", "data"),
    )

    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="siteFromDropdown"),
        Output("selected-site", "data", allow_duplicate=True),
        Input("basin-review-site-dropdown", "value"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="basinSiteMeta"),
        Output("basin-review-location", "children"),
        Output("basin-review-location", "style"),
        Output("basin-review-detail", "children"),
        Input("basin-review-site-dropdown", "value"),
        State("basin-editor-consts", "data"),
    )

    # Site column areas: the v1/v2/v3 rows read area1/area2/area3 from preferred_basin.csv (km2).
    app.clientside_callback(
        ClientsideFunction(namespace="panels", function_name="basinFlags"),
        Output("basin-review-status", "children"),
        Output("basin-review-flag-text", "children"),
        Output("basin-review-flag-text", "style"),
        Output("basin1-area", "children"),
        Output("basin2-area", "children"),
        Output("basin3-area", "children"),
        Input("basin-review-site-dropdown", "value"),
        State("basin-editor-consts", "data"),
    )
