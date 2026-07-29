/* The virtual-site forecast, run entirely in the browser.
 *
 * This is the last thing on the static site that needed a server. A dropped pin is snapped to an NHD reach, the reach's precomputed light feature row is fetched, a year of rows is assembled, and both XGBoost boosters are walked here -- no API call, no Python.
 *
 * WHY THE FEATURES ARE PRECOMPUTED RATHER THAN BUILT HERE. Turning a basin polygon into a feature row needs frac_cell_in_basin, i.e. the CLIPPED AREA of every Voronoi cell against the basin (src/data/site_view.py::_grid_basin_fractions). That weight drives both the area-weighted weather mean and every crop/surplus aggregate, so approximating it -- centroid in/out, say -- shifts every feature away from what the models were trained on, in a way only a parity harness would ever catch. widget/static/build_bundle.py computes the rows with the same recipes._agg_block the training path uses, which removes the divergence by construction instead of testing for it.
 *
 * What is left here is arithmetic: broadcast the static blocks, compute the calendar terms, reconstruct one weather series from its projection, and walk the trees.
 */

(function () {
    window.dash_clientside = window.dash_clientside || {};
    const B = () => window.dash_clientside.bundle;
    const NO = () => window.dash_clientside.no_update;

    /* ── booster ──────────────────────────────────────────────────────────────
     * Decodes what build_bundle._pack_booster writes. Layout: int32 [n_trees, stride, n_features, objective], float32 base_margin, then four flat n_trees*stride arrays -- split_indices int16, left_children int16 (-1 = leaf), default_left uint8, split_conditions float32.
     */
    const OBJ_IDENTITY = 0, OBJ_LOGISTIC = 1;

    function decodeBooster(buf) {
        const h = new DataView(buf);
        const nTrees = h.getInt32(0, true);
        const stride = h.getInt32(4, true);
        const nFeatures = h.getInt32(8, true);
        const objective = h.getInt32(12, true);
        const baseMargin = h.getFloat32(16, true);
        const n = nTrees * stride;
        let o = 20;
        const splitIdx = new Int16Array(buf, o, n); o += n * 2;
        const left = new Int16Array(buf, o, n); o += n * 2;
        const defaultLeft = new Uint8Array(buf, o, n); o += n;
        // The float array is 4-byte aligned only by luck of the preceding sizes, so copy rather than view.
        const cond = new Float32Array(buf.slice(o, o + n * 4));
        return {nTrees, stride, nFeatures, objective, baseMargin, splitIdx, left, defaultLeft, cond};
    }

    /* One row through every tree. `x` MUST be a Float32Array of feature values in the booster's own
     * feature order; a missing value is NaN and takes the node's default branch.
     *
     * Float32 is not a micro-optimisation, it is required for agreement. XGBoost's DMatrix stores
     * features as float32 and compares them against float32 thresholds. Feeding float64 values
     * flips any split where the value sits between the float32 and float64 renderings of the
     * threshold, and a flip near a root swaps a whole subtree: on a row landing exactly on a
     * threshold that cost 1.5e-1 of margin, against 2.9e-7 with float32. predict() enforces the
     * type rather than trusting callers.
     *
     * For a leaf, `cond` holds the leaf weight (the learning rate is already baked in), and the
     * right child is always left+1 -- see the packer, which asserts both. */
    function margin(m, x) {
        let sum = 0;
        for (let t = 0; t < m.nTrees; t++) {
            const base = t * m.stride;
            let i = base;
            while (m.left[i] >= 0) {
                const v = x[m.splitIdx[i]];
                // A missing feature is NaN; note `v !== v` also catches undefined, which Number.isNaN does not.
                const goLeft = v !== v ? m.defaultLeft[i] === 1 : v < m.cond[i];
                i = base + (goLeft ? m.left[i] : m.left[i] + 1);
            }
            sum += m.cond[i];
        }
        return sum + m.baseMargin;
    }

    const sigmoid = (z) => 1 / (1 + Math.exp(-z));

    function predictRow(m, x) {
        if (!(x instanceof Float32Array)) throw new TypeError("booster rows must be Float32Array; see margin()");
        const z = margin(m, x);
        return m.objective === OBJ_LOGISTIC ? sigmoid(z) : z;
    }

    /* Score a whole matrix: `rows` is an array of Float32Array, one per day. */
    function predict(m, rows) {
        const out = new Float64Array(rows.length);
        for (let i = 0; i < rows.length; i++) out[i] = predictRow(m, rows[i]);
        return out;
    }

    /* ── pin -> COMID ─────────────────────────────────────────────────────────
     * Mirrors src/build/_make_basins.snap_comid, which works in EPSG:5070 METRES. Reproducing it
     * means projecting here rather than measuring in degrees: its 25 m tie tolerance is a real
     * distance, and a degree-space approximation would make it mean something different by latitude.
     */

    // EPSG:5070, NAD83 / Conus Albers on GRS80.
    const ALBERS = {a: 6378137.0, f: 1 / 298.257222101, lat0: 23, lon0: -96, lat1: 29.5, lat2: 45.5};

    /* Albers Equal Area forward, ellipsoidal (not the spherical approximation -- that is hundreds
     * of metres out at Iowa's latitude, which would swamp the tie tolerance). */
    const albers = (function () {
        const {a, f, lat0, lon0, lat1, lat2} = ALBERS;
        const e2 = 2 * f - f * f;
        const e = Math.sqrt(e2);
        const rad = Math.PI / 180;
        const q = (phi) => {
            const s = Math.sin(phi);
            return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * Math.log((1 - e * s) / (1 + e * s)));
        };
        const m = (phi) => Math.cos(phi) / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
        const p1 = lat1 * rad, p2 = lat2 * rad, p0 = lat0 * rad;
        const m1 = m(p1), m2 = m(p2), q1 = q(p1), q2 = q(p2);
        const n = (m1 * m1 - m2 * m2) / (q2 - q1);
        const C = m1 * m1 + n * q1;
        const rho0 = (a * Math.sqrt(C - n * q(p0))) / n;
        return function (lat, lon) {
            const rho = (a * Math.sqrt(C - n * q(lat * rad))) / n;
            const theta = n * (lon - lon0) * rad;
            return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
        };
    })();

    /* Squared distance from a point to a segment, the standard projection-onto-segment clamp. */
    function segDist2(px, py, ax, ay, bx, by) {
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy;
        let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const qx = ax + t * dx - px, qy = ay + t * dy - py;
        return qx * qx + qy * qy;
    }

    // Radii snap_comid searches, in order; the FIRST one with any hit decides the candidate set.
    // Not merely an index optimisation: a reach 495 m away and one 515 m away are both within the
    // 25 m tie tolerance of each other, yet the 500 m pass sees only the first. Reproducing the
    // ladder reproduces that.
    const SNAP_RADII = [500, 2000, 10000];
    const TIE_TOLERANCE_M = 25.0;

    function decodeSnapIndex(buf) {
        const h = new DataView(buf);
        const nReaches = h.getInt32(0, true);
        const nVerts = h.getInt32(4, true);
        let o = 8;
        const take = (Ctor, n) => { const a = new Ctor(buf.slice(o, o + n * 4)); o += n * 4; return a; };
        const comid = take(Int32Array, nReaches);
        const totDa = take(Float32Array, nReaches);
        const start = take(Int32Array, nReaches);
        const count = take(Int32Array, nReaches);
        const outletLat = take(Float32Array, nReaches);
        const outletLon = take(Float32Array, nReaches);
        const x = take(Float32Array, nVerts);
        const y = take(Float32Array, nVerts);
        return {nReaches, comid, totDa, start, count, outletLat, outletLon, x, y};
    }

    /* {comid, distance_m, outlet:[lat,lon], moved_m} for the reach a pin belongs to, or null when
     * nothing is within 10 km (which snap_comid raises on -- outside the flowline extent, so there
     * is no forecast to give).
     *
     * `outlet` is the reach's downstream end, which is where the precomputed feature row was built
     * and therefore the location the forecast actually describes. `moved_m` is how far that is from
     * where the user clicked -- worth surfacing, because restricting to stream order >= 3 moves a
     * pin a median 1.5 km and occasionally most of 10. */
    function snapComid(index, lat, lon) {
        const [px, py] = albers(lat, lon);
        const d2 = new Float64Array(index.nReaches);
        for (let r = 0; r < index.nReaches; r++) {
            const s = index.start[r], n = index.count[r];
            let best = Infinity;
            for (let i = s; i < s + n - 1; i++) {
                const v = segDist2(px, py, index.x[i], index.y[i], index.x[i + 1], index.y[i + 1]);
                if (v < best) best = v;
            }
            d2[r] = n === 1 ? (index.x[s] - px) ** 2 + (index.y[s] - py) ** 2 : best;
        }

        for (const radius of SNAP_RADII) {
            const rr = radius * radius;
            let dmin = Infinity;
            for (let r = 0; r < index.nReaches; r++) if (d2[r] <= rr && d2[r] < dmin) dmin = d2[r];
            if (dmin === Infinity) continue;
            // Among the candidates within the tie tolerance of the closest, take the largest
            // upstream drainage area -- which resolves confluences toward the mainstem.
            //
            // On an exact DRAINAGE-AREA tie this keeps the lowest COMID, because reaches are stored
            // in COMID order and the comparison is strict. snap_comid cannot be matched here: its
            // idxmax runs over flowlines.iloc[sindex.query(...)], i.e. STRtree traversal order, so
            // its winner is not reproducible from the data at all. Measured over 600 pins this is
            // the only disagreement, and it was two halves of one stream 0.2 m apart sharing a
            // TotDASqKM of 249.3801 -- the same basin either way. Deterministic beats bit-identical.
            const cutoff = (Math.sqrt(dmin) + TIE_TOLERANCE_M) ** 2;
            let pick = -1;
            for (let r = 0; r < index.nReaches; r++) {
                if (d2[r] > rr || d2[r] > cutoff) continue;
                if (pick < 0 || index.totDa[r] > index.totDa[pick]) pick = r;
            }
            const outlet = [index.outletLat[pick], index.outletLon[pick]];
            const [ox, oy] = albers(outlet[0], outlet[1]);
            return {
                comid: index.comid[pick],
                distance_m: Math.sqrt(dmin),
                outlet: outlet,
                moved_m: Math.hypot(ox - px, oy - py),
                total_da_sqkm: index.totDa[pick],
            };
        }
        return null;
    }

    /* ── the pin ──────────────────────────────────────────────────────────────
     * A click is not a location the model can answer for. Every forecast is computed at a REACH
     * OUTLET from a precomputed row, and restricting to stream order >= 3 puts that outlet a median
     * 1.5 km from where the user clicked (p90 4.3 km, worst case most of 10). Leaving the marker
     * under the cursor would show a place nothing was computed about.
     *
     * So the marker moves to the outlet, a dashed line records where the click was, and the tooltip
     * says which stream it landed on and how far it travelled. The alternative -- silently
     * forecasting a catchment kilometres from the pin -- is the same "looks alive, isn't" failure
     * the clientside conversion existed to remove.
     */
    const fmtKm = (m) => (m < 950 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);

    function pinLayer(click, snap, names, consts) {
        const dl = (type, props) => B().dl(type, props);
        if (!snap) {
            return [dl("Marker", {
                position: click,
                children: dl("Tooltip", {children: "No stream within 10 km — no forecast here", permanent: true, direction: "top"}),
            })];
        }
        const name = (names && names[snap.comid]) || "unnamed stream";
        const label = snap.moved_m < 1
            ? name
            : `${name} — pin moved ${fmtKm(snap.moved_m)} to the reach outlet`;
        return [
            // The click, kept visible so the move is legible rather than mysterious.
            dl("Polyline", {positions: [click, snap.outlet], pathOptions: consts.snap_connector}),
            dl("CircleMarker", {center: click, radius: 3, pathOptions: consts.snap_click}),
            dl("Marker", {
                position: snap.outlet,
                children: dl("Tooltip", {children: label, permanent: true, direction: "top"}),
            }),
        ];
    }

    /* ── the feature frame ────────────────────────────────────────────────────
     * Assembling a year of rows for one reach. Everything here is arithmetic over precomputed
     * blocks -- see the header for why the blocks themselves are precomputed.
     *
     * Column ORDER comes from the model's own feature list, never a list written here: the light
     * recipes' column set is basin-dependent (a bucket exists only if cells fall in it) and moves
     * whenever the recipe is retuned. Resolving by name against the booster means a retrain drops
     * in, and anything the model wants that the reach store cannot supply arrives as NaN, which is
     * exactly how predict() treats an absent distance ring.
     */

    /* Mirrors build_forecast.chunk_of -- a pure function of the outlet, so no artifact carries a
     * chunk id and the two sides cannot disagree about where a reach lives. */
    function chunkOf(lat, lon, meta) {
        const [lat0, lon0, lat1, lon1] = meta.bbox;
        const [rows, cols] = meta.grid;
        const r = Math.min(rows - 1, Math.max(0, Math.floor(((lat - lat0) / (lat1 - lat0)) * rows)));
        const c = Math.min(cols - 1, Math.max(0, Math.floor(((lon - lon0) / (lon1 - lon0)) * cols)));
        return r * cols + c;
    }

    // Field order inside a reach's per-year block; must match build_forecast._row_values.
    const STATIC_FIELDS = ["lat", "lon", "basin_area_m2", "mean_dist_to_sensor", "max_dist_to_sensor", "log_basin_area"];

    function decodeChunk(buf) {
        const h = new DataView(buf);
        const n = h.getInt32(0, true), rank = h.getInt32(4, true);
        const nYears = h.getInt32(8, true), nBuckets = h.getInt32(12, true), nCrops = h.getInt32(16, true);
        const perYear = nCrops * nBuckets + nBuckets + nCrops + 1;
        const perTask = nYears * perYear;
        let o = 20;
        const take = (Ctor, count, width) => { const a = new Ctor(buf.slice(o, o + count * width)); o += count * width; return a; };
        const comid = take(Int32Array, n, 4);
        const statics = take(Float32Array, n * 6, 4);
        const lags = take(Int8Array, n * 2, 1);
        const offset = take(Float32Array, n, 4);
        const coef = take(Float32Array, n * rank, 4);
        const reg = take(Float32Array, n * perTask, 4);
        const clf = take(Float32Array, n * perTask, 4);
        const at = new Map();
        for (let i = 0; i < n; i++) at.set(comid[i], i);
        return {n, rank, nYears, nBuckets, nCrops, perYear, perTask, comid, statics, lags, offset, coef,
                blocks: {reg, clf}, at};
    }

    function decodeModes(buf) {
        const h = new DataView(buf);
        const k = h.getInt32(0, true), nDays = h.getInt32(4, true);
        const days = new Int32Array(buf.slice(8, 8 + nDays * 4));
        const Vt = new Float32Array(buf.slice(8 + nDays * 4, 8 + nDays * 4 + k * nDays * 4));
        return {k, nDays, days, Vt};
    }

    function decodeCrossSite(buf) {
        const n = new DataView(buf).getInt32(0, true);
        let o = 4;
        const days = new Int32Array(buf.slice(o, o + n * 4)); o += n * 4;
        const cols = {};
        for (const name of ["rest_of_state_nitrate_lag1", "rest_of_state_nitrate_lag2",
                            "rest_of_state_nitrate_lag3", "rest_of_state_nitrate_lag5",
                            "roll_n_avg_except_this7d"]) {
            cols[name] = new Float32Array(buf.slice(o, o + n * 4));
            o += n * 4;
        }
        const at = new Map();
        for (let i = 0; i < n; i++) at.set(days[i], i);
        return {n, days, cols, at};
    }

    const dayOfYear = (epochDay) => {
        const d = new Date(epochDay * 86400000);
        return Math.floor((d - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400000) + 1;
    };

    /* One reach, one year, one task -> {dates, rows}. `rows` are Float32Array in `feat` order. */
    function assemble(reach, year, task, feat, meta, modes, cross) {
        const {chunk, i} = reach;
        // The spine is the weather calendar, not a synthetic Jan-Dec range: the weather store drops
        // a day here and there, and target_year_spine takes the dates that exist.
        const spine = [];
        for (let d = 0; d < modes.nDays; d++) {
            if (new Date(modes.days[d] * 86400000).getUTCFullYear() === year) spine.push(d);
        }

        const yearIdx = meta.years.indexOf(year);
        if (yearIdx < 0) throw new Error(`year ${year} is not in the reach store (${meta.years})`);
        const block = chunk.blocks[task];
        const base = i * chunk.perTask + yearIdx * chunk.perYear;
        const lag = chunk.lags[i * 2 + (task === "reg" ? 0 : 1)];

        // Per-year values, constant across the year's rows.
        const yearly = new Map();
        meta.crop_classes.forEach((cls, ci) => {
            for (let b = 0; b < chunk.nBuckets; b++) {
                yearly.set(`pct_${cls.toLowerCase()}_b${b}`, block[base + ci * chunk.nBuckets + b]);
            }
        });
        for (let b = 0; b < chunk.nBuckets; b++) {
            yearly.set(`surplus_kgha_norm_b${b}`, block[base + chunk.nCrops * chunk.nBuckets + b]);
        }
        const expOff = base + chunk.nCrops * chunk.nBuckets + chunk.nBuckets;
        // The exp-decay tag carries lambda (Corn_expT2000), so these are keyed by PREFIX and matched
        // that way below -- the packer cannot know what lambda a future recipe will use.
        const expByPrefix = new Map();
        meta.crop_classes.forEach((cls, ci) => expByPrefix.set(`${cls}_expT`, block[expOff + ci]));
        expByPrefix.set("surplus_kgha_expT", block[expOff + chunk.nCrops]);

        const statics = new Map(STATIC_FIELDS.map((f, k) => [f, chunk.statics[i * 6 + k]]));

        const rows = spine.map((d) => {
            const x = new Float32Array(feat.length);
            const ang = (2 * Math.PI * dayOfYear(modes.days[d])) / 365.25;
            const cal = {doy_sin: Math.sin(ang), doy_cos: Math.cos(ang),
                         doy_sin2: Math.sin(2 * ang), doy_cos2: Math.cos(2 * ang)};
            // Weather: reconstruct the basin mean from the shared modes, then shift by the reach's
            // travel-time lag -- row t takes the value from t-lag, as lag_buckets does.
            const src = Math.max(0, d - lag);
            let fm = chunk.offset[i];
            for (let j = 0; j < chunk.rank; j++) fm += chunk.coef[i * chunk.rank + j] * modes.Vt[j * modes.nDays + src];
            const ci = cross.at.get(modes.days[d]);

            for (let f = 0; f < feat.length; f++) {
                const name = feat[f];
                let v;
                if (name === "fuel_moisture_1000h") v = fm;
                else if (name in cal) v = cal[name];
                else if (statics.has(name)) v = statics.get(name);
                else if (yearly.has(name)) v = yearly.get(name);
                else if (cross.cols[name]) v = ci === undefined ? NaN : cross.cols[name][ci];
                else {
                    const pre = [...expByPrefix.keys()].find((p) => name.startsWith(p));
                    v = pre === undefined ? NaN : expByPrefix.get(pre);
                }
                x[f] = v === undefined ? NaN : v;
            }
            return x;
        });
        return {dates: spine.map((d) => modes.days[d]), rows};
    }

    const snapIndex = () => B().memo("snap_index", () => B().buffer("forecast/snap_index.bin").then(decodeSnapIndex));
    const reachNames = () => B().json("forecast/reach_names.json");
    const chunkMeta = () => B().json("forecast/reach_chunks.json");
    const modes = () => B().memo("weather_modes", () => B().buffer("forecast/weather_modes.bin").then(decodeModes));
    const crossSite = () => B().memo("cross_site", () => B().buffer("forecast/cross_site.bin").then(decodeCrossSite));
    const chunk = (cid) => B().memo(`chunk:${cid}`, () => B().buffer(`forecast/reaches/${cid}.bin`).then(decodeChunk));
    const model = (task) => B().memo(`model:${task}`, () => Promise.all([
        B().buffer(`models/${task}.bin`).then(decodeBooster), B().json(`models/${task}.json`),
    ]).then(([booster, meta]) => ({booster, feat: meta.feat, beta_table: meta.beta_table, base_rate: meta.base_rate})));

    window.dash_clientside.forecast = {
        /* Pin drop. Replaces ui.regionGeom, which dropped the marker under the cursor.
         *
         * region-geom now carries the snap alongside the raw click, so every consumer -- the
         * forecast, the readout, the debug basin overlay -- describes the same place. Keeping the
         * click in the store as well means nothing has silently lost what the user actually did. */
        regionGeom: function (clickData, mode, consts) {
            if (mode !== "pin" || !clickData || !clickData.latlng) return [NO(), NO()];
            const {lat, lng} = clickData.latlng;
            const click = [lat, lng];
            return Promise.all([snapIndex(), reachNames()]).then(([index, names]) => {
                const snap = snapComid(index, lat, lng);
                return [
                    {
                        type: "Point",
                        coordinates: [lng, lat],
                        snap: snap && {
                            comid: snap.comid,
                            outlet: snap.outlet,
                            moved_m: snap.moved_m,
                            distance_m: snap.distance_m,
                            total_da_sqkm: snap.total_da_sqkm,
                            name: names[snap.comid] || null,
                        },
                    },
                    pinLayer(click, snap, names, consts),
                ];
            });
        },

        _pinLayer: pinLayer,
        // exported for the parity harness; the callbacks land in a later step
        _chunkOf: chunkOf,
        _decodeChunk: decodeChunk,
        _decodeModes: decodeModes,
        _decodeCrossSite: decodeCrossSite,
        _assemble: assemble,
        _decodeBooster: decodeBooster,
        _margin: margin,
        _predict: predict,
        _predictRow: predictRow,
        _albers: albers,
        _decodeSnapIndex: decodeSnapIndex,
        _snapComid: snapComid,
    };
})();
