/* Clientside callbacks for the side panels (timeseries figure, sites table, basin-editor readouts).
 *
 * Everything here reads sites.json, which build_bundle joins from the site metadata, the nitrate stats and the basin metadata -- so the six-column Sites Selected table and the whole Basin Editor come out of one 65 KB fetch that is cached for the page's lifetime. The live app instead ran access.get_basin_area per row, which is a full flow-field build each time.
 *
 * Where a readout was a fixed shape with varying text (the basin-editor status and metadata lines), the SKELETON is rendered in Python and only the strings and the one varying colour are written from here -- so those styles never had to cross into JS at all.
 *
 * The figure is built here rather than precomputed because the bundle already ships the series in their compact form: build_bundle stores one nitrate/precip pair per (site, interval) at FIXED aggregations, so there is no resampling left to do -- this reads two typed arrays and emits a Plotly spec. Precomputing the figure JSON instead would inline ISO date strings and cost roughly 3x the bytes of the binary series it would replace.
 *
 * The spec below mirrors info_panel._build_timeseries_figure. If that changes, change this too -- the parity spot-check in the export verification is what catches a divergence.
 */

(function () {
    window.dash_clientside = window.dash_clientside || {};
    const NO = () => window.dash_clientside.no_update;
    const B = () => window.dash_clientside.bundle;
    const h = (type, props) => B().h(type, props);
    const trig = () => B().triggeredId();

    const DASH = "—";  // what every readout shows where the value is missing
    const isNum = (v) => typeof v === "number" && !Number.isNaN(v);
    const thousands = (v) => Math.round(v).toLocaleString("en-US");  // Python's "{:,.0f}"
    const flagOn = (v) => (v || []).indexOf("on") !== -1;
    const hasBasinMeta = (s) => s.basin_type !== undefined && s.basin_type !== null;
    const flagKeys = (s) => Object.keys(s).filter((k) => k.startsWith("flag_"));
    const fmtArea = (v) => (isNum(v) ? `${thousands(v)} km²` : "");

    const NITRATE = "#6b21a8";
    const PRECIP = "#3a94fa";
    const BAR_THRESH = 200; // below this many points, precip reads better as bars than a line
    // Astronomical season starts, as [month, day]. Drawn as vertical rules when "seasons" is on.
    const SEASONS = [[3, 21], [6, 21], [9, 21], [12, 21]];

    const isoFromEpochDays = (d) => new Date(d * 86400000).toISOString().slice(0, 10);
    // Float32Array carries NaN for gaps; Plotly wants null to break a line there.
    const nullNaN = (a) => Array.from(a, (v) => (Number.isNaN(v) ? null : v));

    window.dash_clientside.panels = {
        timeseries: function (activeUid, interval, graphToggle) {
            const blank = {data: [], layout: {margin: {t: 20, b: 40, l: 50, r: 50}}};
            if (!activeUid || !interval) return blank;
            if ((graphToggle || []).indexOf("show") === -1) return window.dash_clientside.no_update;
            const showSeasons = (graphToggle || []).indexOf("seasons") !== -1;

            return B().series(activeUid, interval).then(({n, columns}) => {
                const [days, nitrate, precip] = columns;
                const x = Array.from(days, isoFromEpochDays);

                const traces = [];
                // Precip first so it sits behind the nitrate line.
                const precipCommon = {x: x, y: nullNaN(precip), name: "Precip (in)", yaxis: "y2", opacity: 0.5};
                traces.push(
                    n < BAR_THRESH
                        ? {...precipCommon, type: "bar", marker: {color: PRECIP}}
                        : {...precipCommon, type: "scatter", mode: "lines", line: {color: PRECIP, width: 1}}
                );
                traces.push({
                    x: x,
                    y: nullNaN(nitrate),
                    name: "Nitrate (mg/L)",
                    yaxis: "y1",
                    type: "scatter",
                    mode: "lines",
                    line: {color: NITRATE, width: 1.5},
                });

                const shapes = [];
                if (showSeasons && n) {
                    const y0 = Number(x[0].slice(0, 4));
                    const y1 = Number(x[x.length - 1].slice(0, 4));
                    for (let y = y0; y <= y1; y++) {
                        for (const [m, d] of SEASONS) {
                            const at = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
                            shapes.push({
                                type: "line", xref: "x", yref: "paper", x0: at, x1: at, y0: 0, y1: 1,
                                line: {width: 1, color: "black"}, opacity: 0.3,
                            });
                        }
                    }
                }

                return {
                    data: traces,
                    layout: {
                        // fixedrange on y: the range selector drives x, and letting y zoom
                        // independently makes the two axes disagree about scale.
                        yaxis: {title: {text: "Nitrate (mg/L)"}, fixedrange: true},
                        yaxis2: {
                            title: {text: "Precipitation (in)"},
                            overlaying: "y", side: "right", showgrid: false, fixedrange: true,
                        },
                        xaxis: {
                            title: null,
                            rangeselector: {
                                buttons: [
                                    {count: 1, label: "1M", step: "month", stepmode: "backward"},
                                    {count: 3, label: "3M", step: "month", stepmode: "backward"},
                                    {count: 1, label: "1Y", step: "year", stepmode: "backward"},
                                    {count: 3, label: "3Y", step: "year", stepmode: "backward"},
                                    {step: "all", label: "All"},
                                ],
                            },
                        },
                        legend: {orientation: "h", y: -0.15},
                        margin: {t: 20, b: 40, l: 50, r: 50},
                        shapes: shapes,
                        annotations: [{
                            text: activeUid, xref: "paper", yref: "paper", x: 1, y: 1,
                            xanchor: "right", yanchor: "bottom", showarrow: false,
                            font: {size: 11, color: "#888"},
                        }],
                    },
                };
            }).catch(() => blank); // a site with no series bin renders empty rather than throwing
        },

        /* Sites Selected: one row per selected site, with a clickable uid (sets the graph site) and a × (drops it). Both carry pattern-matching ids, which clientside callbacks support unchanged. */
        sitesTable: function (selectedUids, activeUid, consts) {
            const c = consts.sites_table;
            const header = h("Thead", {children: h("Tr", {children: [
                ...c.columns.map((name, i) => h("Th", {children: name, style: i === 0 ? c.th_left : c.th_center})),
                h("Th", {children: "", style: c.th_last}),  // the × column
            ]})});
            const selected = selectedUids || [];
            const table = (children) => h("Table", {children: children, style: c.table});
            if (!selected.length) return [table([header]), c.clear_hidden];

            return B().sitesById().then((byUid) => {
                const rows = selected.map((uid) => {
                    const s = byUid.get(uid);
                    const cells = [
                        s && isNum(s.nitrate_sparsity) ? (s.nitrate_sparsity * 100).toFixed(1) : DASH,
                        s && s.start_date ? s.start_date.slice(0, 7) : DASH,
                        s && s.last_date ? s.last_date.slice(0, 7) : DASH,
                        s && isNum(s.lifespan) ? s.lifespan.toFixed(2) : DASH,
                        s && isNum(s.basin_area_km2) ? thousands(s.basin_area_km2) : DASH,
                    ];
                    return h("Tr", {children: [
                        h("Td", {
                            children: h("Span", {
                                children: uid,
                                id: {type: "graph-site-btn", index: uid},
                                n_clicks: 0,
                                style: {...c.uid, fontWeight: uid === activeUid ? "bold" : "normal"},
                            }),
                            style: c.td_left,
                        }),
                        ...cells.map((v) => h("Td", {children: v, style: c.td_center})),
                        h("Td", {
                            children: h("Span", {
                                children: "×",
                                id: {type: "remove-site-btn", index: uid},
                                n_clicks: 0,
                                style: c.remove,
                            }),
                            style: c.td_right,
                        }),
                    ]});
                });
                return [table([header, h("Tbody", {children: rows})]), c.clear_visible];
            });
        },

        /* Pin-drop readout. The area branch went with the polygon selector, so a Point is the only geometry that reaches here.
         *
         * Reports the SNAPPED location, because that is the one the forecast describes -- and says how far it is from the click, so a pin that jumped 4 km to the nearest order-3 stream reads as a fact rather than an oddity. */
        regionInfo: function (regionGeom, consts) {
            if (!regionGeom || regionGeom.type !== "Point") return ["", consts.point_row_hidden];
            const [lng, lat] = regionGeom.coordinates;
            const snap = regionGeom.snap;
            if (!snap) return [`${lat.toFixed(5)}, ${lng.toFixed(5)} — no stream within 10 km`, consts.point_row_visible];
            const [olat, olon] = snap.outlet;
            const where = `${olat.toFixed(5)}, ${olon.toFixed(5)}`;
            const moved = snap.moved_m < 1 ? "" : ` (moved ${snap.moved_m < 950 ? Math.round(snap.moved_m) + " m" : (snap.moved_m / 1000).toFixed(1) + " km"} from the click)`;
            return [`${where} — ${snap.name || "unnamed stream"}, ${Math.round(snap.total_da_sqkm).toLocaleString("en-US")} km² upstream${moved}`, consts.point_row_visible];
        },

        /* Basin Editor site list, narrowed by the two filter checkboxes. Sites with no basin metadata are dropped -- the Python original got that for free by iterating the basin metadata table. */
        basinDropdown: function (flaggedOnly, unreviewedOnly, _version, selectedSites, activeMenu) {
            const wantFlagged = flagOn(flaggedOnly);
            const wantUnreviewed = flagOn(unreviewedOnly);
            const followSelection =
                trig() === "selected-site" && activeMenu === "debug" && selectedSites && selectedSites.length === 1;

            return B().sites().then((sites) => {
                const options = sites
                    .filter((s) => hasBasinMeta(s))
                    .filter((s) => !wantFlagged || flagKeys(s).some((k) => s[k]))
                    .filter((s) => !wantUnreviewed || !s.reviewed)
                    .map((s) => ({label: s.site_uid, value: s.site_uid}));
                return [options, followSelection ? selectedSites[0] : NO()];
            });
        },

        siteFromDropdown: function (siteUid) {
            if (trig() !== "basin-review-site-dropdown") return NO();
            return siteUid ? [siteUid] : [];
        },

        /* "<river> at <town>, <state>" plus the reported drainage area and the source registry. Composed from the columns site_location_metadata.csv actually has; see the Python docstring on why there is no get_site_desc here. */
        basinSiteMeta: function (siteUid, consts) {
            const blank = ["", consts.location_unknown, ""];
            if (!siteUid) return blank;
            return B().sitesById().then((byUid) => {
                const s = byUid.get(siteUid);
                if (!s) return blank;

                const where = [s.river, s.town].filter(Boolean).join(" at ");
                const location = where && s.state ? `${where}, ${s.state}` : (where || s.nickname || "");

                const detail = [];
                // draining_area is the operator-reported drainage area in SQUARE MILES (the IWQIS registry's spelling and unit).
                if (isNum(s.draining_area)) detail.push(`reported drainage: ${thousands(s.draining_area)} mi²`);
                detail.push(`source: ${s.source}`);

                return [
                    location || "unnamed location",
                    location ? consts.location_known : consts.location_unknown,
                    detail.join(" · "),
                ];
            });
        },

        /* Basin status line plus the three site-column areas in the display table. Flag labels arrive from Python, and any flag column without one falls back to its own name -- so a new entry in _make_basins.FLAG_COLS surfaces here rather than being dropped. */
        basinFlags: function (siteUid, consts) {
            const blank = ["", "", consts.flags_none, "", "", ""];
            if (!siteUid) return blank;
            return B().sitesById().then((byUid) => {
                const s = byUid.get(siteUid);
                if (!s || !hasBasinMeta(s)) return blank;

                const active = flagKeys(s)
                    .filter((k) => s[k])
                    .map((k) => consts.flag_labels[k] || k.replace(/^flag_/, "").replace(/_/g, " "));

                const status = `v${s.basin_type} · ${s.selection_mode || "auto"} · ${s.reviewed ? "reviewed" : "unreviewed"}`;
                return [
                    status,
                    `flags: ${active.length ? active.join(", ") : "none"}`,
                    active.length ? consts.flags_some : consts.flags_none,
                    fmtArea(s.area1), fmtArea(s.area2), fmtArea(s.area3),
                ];
            });
        },
    };
})();
