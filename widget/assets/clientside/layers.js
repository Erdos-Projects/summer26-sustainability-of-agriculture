/* Clientside callbacks for the map layers and legends.
 *
 * The guiding trick here: most of these outputs LOOK like they need a component tree built in JavaScript, and almost none of them do.
 *
 *   - Enumerable outputs (the two legends, the IEM footprint) are pre-rendered into the layout and switched with a `style` write.
 *   - Geometry layers emit a dl.GeoJSON component whose `url` points into the bundle, so LEAFLET fetches and parses the polygons. No geometry is ever handled here.
 *
 * A Dash component is just {namespace, type, props}, so emitting one is an object literal.
 */

(function () {
    window.dash_clientside = window.dash_clientside || {};
    const NO = () => window.dash_clientside.no_update;
    const B = () => window.dash_clientside.bundle;

    const SHOW = {display: "block"};
    const HIDE = {display: "none"};
    const on = (v) => (v || []).indexOf("show") !== -1;

    /* A dash_leaflet component as plain JSON; the constructor lives in bundle.js now that three namespaces emit components. Resolved at call time, not load time. */
    const dl = (type, props) => B().dl(type, props);

    const geojsonUrl = (url, options) => dl("GeoJSON", {url: url, options: options});

    // Checklist value for the basin-review filters ("show" is the map-layer toggles' value).
    const flagOn = (v) => (v || []).indexOf("on") !== -1;
    const hasBasinMeta = (s) => s.basin_type !== undefined && s.basin_type !== null;
    const anyFlag = (s) => Object.keys(s).some((k) => k.startsWith("flag_") && s[k]);

    const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(v);
    // Python's "{:,.0f}"; used for total N and basin areas so the tooltips read the same either way.
    const thousands = (v) => Math.round(v).toLocaleString("en-US");

    window.dash_clientside.layers = {
        /* Two pre-rendered legends, one visible at a time -- and neither when the grid is off or no site is active. */
        gridColorLegend: function (rainToggle, colorMode, activeUid) {
            if (!on(rainToggle) || !activeUid) return [HIDE, HIDE];
            return colorMode === "crop" ? [SHOW, HIDE] : [HIDE, SHOW];
        },

        /* One fixed rectangle. A dl.LayerGroup is a Leaflet layer, not a DOM node, so it has no `style` to toggle -- the component is emitted or not. Bounds and pathOptions come from the consts Store so the literal stays defined in Python. */
        iemBbox: function (value, consts) {
            return on(value) ? [dl("Rectangle", consts.iem_bbox)] : [];
        },

        /* Waterbodies then flowlines, both by URL. Styles arrive from Python via the consts Store so the palette stays defined in one place. */
        hydro: function (value, consts) {
            if (!on(value)) return [];
            const {urls, styles} = consts.hydro;
            return [
                geojsonUrl(urls.waterbodies, {pane: "hydro-pane", style: styles.waterbodies}),
                geojsonUrl(urls.flowlines, {pane: "hydro-pane", style: styles.flowlines}),
            ];
        },

        /* Preferred basin per selected site, plus the dissolved union of all of them. One GeoJSON component per layer; Leaflet does the fetching. */
        upstream: function (preferredToggle, allToggle, selectedUids, _version, consts) {
            const out = [];
            if (on(allToggle)) {
                out.push(geojsonUrl(B().dataUrl("basins/union.geojson"), {
                    pane: "basin-pane", style: consts.basin_style_preferred, interactive: false,
                }));
            }
            if (on(preferredToggle)) {
                for (const uid of selectedUids || []) {
                    out.push(geojsonUrl(B().dataUrl(`basins/${uid}_preferred.geojson`), {
                        pane: "basin-pane", style: consts.basin_style_preferred, interactive: false,
                    }));
                }
            }
            return out;
        },

        /* One factory for the v1/v2/v3 comparison layers; the basin type and its style come in as State so the three registrations share an implementation. */
        basinVersion: function (toggle, selectedUids, btype, consts) {
            if (!on(toggle)) return [];
            const style = consts[`basin_style_${btype}`];
            return (selectedUids || []).map((uid) =>
                geojsonUrl(B().dataUrl(`basins/${uid}_v${btype}.geojson`), {
                    pane: "basin-pane", style: style, interactive: false,
                })
            );
        },

        /* One CircleMarker per monitoring site, optionally narrowed by the basin-review filters. Marker colours and radii come from the consts Store so colors.py stays the palette. */
        iwqisSites: function (selectedUids, flaggedOnly, unreviewedOnly, consts) {
            const selected = new Set(selectedUids || []);
            const wantFlagged = flagOn(flaggedOnly);
            const wantUnreviewed = flagOn(unreviewedOnly);
            return B().sites().then((sites) => sites
                .filter((s) => {
                    if (!wantFlagged && !wantUnreviewed) return true;
                    // A site with no basin metadata cannot satisfy either filter, matching the Python join.
                    if (!hasBasinMeta(s)) return false;
                    if (wantFlagged && !anyFlag(s)) return false;
                    if (wantUnreviewed && s.reviewed) return false;
                    return true;
                })
                .map((s) => {
                    const kind = selected.has(s.site_uid) ? "selected" : (s.source === "USGS" ? "usgs" : "default");
                    const m = consts.site_markers[kind];
                    return dl("CircleMarker", {
                        id: {type: "iwqis-marker", index: s.site_uid},
                        center: [s.lat, s.lon],
                        radius: m.radius,
                        color: m.color,
                        fillColor: m.fillColor,
                        fillOpacity: 0.8,
                        weight: 1,
                        pane: "sites-pane",
                        // so a marker click does not also reach the map's click handler and drop a pin
                        bubblingMouseEvents: false,
                    });
                })
            );
        },

        /* The one genuinely computational display callback: the active site's Voronoi cells, coloured and captioned from the year's covariates.
         *
         * Three artifacts are joined on global_node_id -- the site's cell index, the global grid geometry, and the year's surplus/crop arrays -- which is exactly the join _rain_grid_features does in pandas. Everything it needs is a typed array or a cached fetch, so a year-slider drag re-colours without touching the network after the first year.
         *
         * A site with no cell index (or a year with no covariates) yields an empty layer rather than the Python fallback's centroid dots: the dots existed for sites whose grid geometry had not been built yet, and the bundle only ships cells for sites that have one. */
        rainGrid: function (toggle, activeUid, year, colorMode, consts) {
            if (!on(toggle) || !activeUid) return [];
            const mode = colorMode || "surplus";
            const g = consts.rain_grid;

            return Promise.all([
                B().siteCells(activeUid),
                B().gridIndex(),
                B().palette(),
                B().manifest(),
                B().surplus(year).catch(() => null),
                B().crops(year).catch(() => null),
            ]).then(([cells, grid, pal, man, sur, crp]) => Promise.all([
                sur ? B().indexById(`surplus_${year}`, sur.columns[0]) : null,
                crp ? B().indexById(`crops_${year}`, crp.ids) : null,
            ]).then(([surIdx, crpIdx]) => {
                const classes = man.coverage.crop_classes;
                const K = classes.length;
                const [nodeIds, gnids, fracs, dists] = cells.columns;
                const features = [];

                for (let i = 0; i < cells.n; i++) {
                    const gnid = gnids[i];
                    const cell = grid.get(gnid);
                    if (!cell) continue;

                    const si = surIdx ? surIdx.get(gnid) : undefined;
                    const surplus = si === undefined ? null : sur.columns[1][si];
                    const totalN = si === undefined ? null : sur.columns[2][si];
                    const ci = crpIdx ? crpIdx.get(gnid) : undefined;
                    const counts = ci === undefined ? null : crp.counts.subarray(ci * K, ci * K + K);

                    features.push({
                        type: "Feature",
                        geometry: cell.geometry,
                        properties: {
                            node_id: nodeIds[i],
                            color: cellColor(mode, surplus, counts, classes, pal, g.nodata),
                            tooltip: cellTooltip(
                                nodeIds[i], cell.properties.cell_area, fracs[i], dists[i],
                                surplus, totalN, counts, classes
                            ),
                        },
                    });
                }

                return [dl("GeoJSON", {
                    data: {type: "FeatureCollection", features: features},
                    style: g.style,
                    hoverStyle: g.hoverStyle,
                    onEachFeature: g.onEachFeature,
                    zoomToBounds: false,
                    pane: "rain-grid-pane",
                })];
            })).catch(() => []);
        },
    };

    /* Cell fill. Mirrors the colour half of _rain_grid_features. */
    function cellColor(mode, surplus, counts, classes, pal, nodata) {
        if (mode === "crop") {
            if (!counts) return nodata;
            let best = -1, bestClass = null, total = 0;
            for (let k = 0; k < classes.length; k++) {
                total += counts[k];
                if (counts[k] > best) {  // strict >, so ties keep the first class -- pandas idxmax does the same
                    best = counts[k];
                    bestClass = classes[k];
                }
            }
            return total ? (pal.crops[bestClass] || nodata) : nodata;
        }
        if (!isNum(surplus)) return nodata;
        /* build_bundle sampled surplus_viz.surplus_to_hex at 256 points, which recovers matplotlib's YlOrRd table entry for entry -- so this has to index it the way matplotlib does.
         *
         * Colormap.__call__ takes floor(t * N) and clips to N-1; it does NOT round to the nearest of N-1 intervals. The two rules disagree for a quarter of all t, one LUT step apart, which is exactly the kind of difference that looks like noise in a screenshot and is in fact a systematically wrong palette. */
        const s = pal.surplus;
        const t = Math.min(1, Math.max(0, (surplus - s.lo) / ((s.hi - s.lo) || 1)));
        return s.lut[Math.min(255, Math.floor(t * 256))];
    }

    /* Cell caption. Mirrors _cell_tooltip line for line, including which lines are omitted when a value is missing -- the tooltip is the only place the joined covariates are legible, so a divergence here is a divergence nobody would notice. */
    function cellTooltip(nodeId, cellArea, frac, dist, surplus, totalN, counts, classes) {
        const lines = [`<b>Cell ${nodeId}</b>`];
        if (isNum(cellArea)) {
            const km2 = cellArea / 1e6;  // cell_area is m² (EPSG:5070)
            lines.push(`Cell area: ${km2.toFixed(2)} km²`);
            if (isNum(frac)) lines.push(`Cell area in basin: ${(km2 * frac).toFixed(2)} km²`);
        }
        if (isNum(dist)) lines.push(`Dist to sensor: ${(dist / 1e3).toFixed(1)} km`);  // dist_to_sensor is m
        if (isNum(surplus)) {
            lines.push(`Surplus: ${surplus.toFixed(0)} kg/ha`);
            lines.push(`Total N: ${thousands(totalN)} kg`);
        } else {
            lines.push("Surplus: n/a (outside data)");
        }
        if (counts) {
            const present = [];
            let total = 0;
            for (let k = 0; k < classes.length; k++) {
                if (counts[k] > 0) present.push([classes[k], counts[k]]);
                total += counts[k];
            }
            if (total) {
                present.sort((a, b) => b[1] - a[1]);
                lines.push("Crops:");
                for (const [name, v] of present) lines.push(`&nbsp;&nbsp;${name}: ${Math.round((v / total) * 100)}%`);
            }
        }
        return lines.join("<br>");
    }
})();
