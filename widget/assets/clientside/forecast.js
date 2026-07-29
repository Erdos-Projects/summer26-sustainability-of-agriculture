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
        // Weather is per task AND per ring: the two tasks cut their rings at different distances, so each is its own projection against the shared modes. Indexed [reach * nBuckets + ring], the packer having already resolved which lag belongs to which ring.
        const lags = {reg: take(Int8Array, n * nBuckets, 1), clf: take(Int8Array, n * nBuckets, 1)};
        const offset = {reg: take(Float32Array, n * nBuckets, 4), clf: take(Float32Array, n * nBuckets, 4)};
        const coef = {reg: take(Float32Array, n * nBuckets * rank, 4), clf: take(Float32Array, n * nBuckets * rank, 4)};
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

    const CALENDAR = ["doy_sin", "doy_cos", "doy_sin2", "doy_cos2"];

    /* Where every model feature comes from -- one {kind, ...} per name, in `feat` order.
     *
     * Resolved ONCE per forecast rather than per day: the source of a column cannot change between rows, and a name that matches nothing is a `null` kind rather than a silent NaN, which is what the publish-time skew check reads (widget/static/export.py).
     */
    function plan(chunk, i, year, task, feat, meta) {
        const yearIdx = meta.years.indexOf(year);
        if (yearIdx < 0) throw new Error(`year ${year} is not in the reach store (${meta.years})`);
        const base = i * chunk.perTask + yearIdx * chunk.perYear;
        const nb = chunk.nBuckets;

        const yearly = new Map();
        meta.crop_classes.forEach((cls, ci) => {
            for (let b = 0; b < nb; b++) yearly.set(`pct_${cls.toLowerCase()}_b${b}`, base + ci * nb + b);
        });
        for (let b = 0; b < nb; b++) yearly.set(`surplus_kgha_norm_b${b}`, base + chunk.nCrops * nb + b);

        // Exp-decay columns match by EXACT name, taken from the manifest -- their tag carries the recipe's lambda (Corn_expT2000), and matching a bare Corn_expT by prefix would feed one lambda's values into a model trained at another. That is the skew deploy.predict._assert_no_skew refuses to score through, so it must not resolve here either.
        const expOff = base + chunk.nCrops * nb + nb;
        const expNames = (meta.expT_cols || {})[task] || [];
        const expByName = new Map(expNames.map((name, k) => [name, expOff + k]));

        // Weather column names come from the manifest, not from here: WEATHER_KEEP is a recipe setting and renaming it must not need a JS edit.
        const wbase = (meta.weather_cols || ["fuel_moisture_1000h"])[0];
        const weather = new Map();
        for (let b = 0; b < nb; b++) weather.set(`${wbase}_b${b}`, b);
        if (nb === 1) weather.set(wbase, 0);  // an unbucketed weather block carries no _b suffix

        return feat.map((name) => {
            const s = STATIC_FIELDS.indexOf(name);
            if (CALENDAR.indexOf(name) >= 0) return {kind: "calendar", name: name};
            if (s >= 0) return {kind: "static", at: i * 6 + s};
            if (weather.has(name)) return {kind: "weather", bucket: weather.get(name)};
            if (yearly.has(name)) return {kind: "yearly", at: yearly.get(name)};
            if (cross_cols_has(name)) return {kind: "cross", name: name};
            if (expByName.has(name)) return {kind: "expT", at: expByName.get(name)};
            return {kind: null, name: name};
        });
    }

    // Names the cross-site file carries; fixed by build_forecast.build_cross_site's column order.
    const CROSS_COLS = ["rest_of_state_nitrate_lag1", "rest_of_state_nitrate_lag2", "rest_of_state_nitrate_lag3",
                        "rest_of_state_nitrate_lag5", "roll_n_avg_except_this7d"];
    const cross_cols_has = (name) => CROSS_COLS.indexOf(name) >= 0;

    /* One reach, one year, one task -> {dates, rows}. `rows` are Float32Array in `feat` order. */
    function assemble(reach, year, task, feat, meta, modes, cross) {
        const {chunk, i} = reach;
        // The spine is the weather calendar, not a synthetic Jan-Dec range: the weather store drops a day here and there, and target_year_spine takes the dates that exist.
        const spine = [];
        for (let d = 0; d < modes.nDays; d++) {
            if (new Date(modes.days[d] * 86400000).getUTCFullYear() === year) spine.push(d);
        }

        const steps = plan(chunk, i, year, task, feat, meta);
        const block = chunk.blocks[task];
        const nb = chunk.nBuckets;
        const lags = chunk.lags[task], offs = chunk.offset[task], coefs = chunk.coef[task];

        const rows = spine.map((d) => {
            const x = new Float32Array(feat.length);
            const ang = (2 * Math.PI * dayOfYear(modes.days[d])) / 365.25;
            const cal = {doy_sin: Math.sin(ang), doy_cos: Math.cos(ang),
                         doy_sin2: Math.sin(2 * ang), doy_cos2: Math.cos(2 * ang)};
            const ci = cross.at.get(modes.days[d]);

            // Weather per RING: reconstruct that ring's basin mean from the shared modes, then shift by its own travel-time lag -- row t takes the value from t-lag, as lag_buckets does. An absent ring is NaN, which is what the recipe emits and what the booster routes down its default branch.
            const fm = new Float64Array(nb);
            for (let b = 0; b < nb; b++) {
                const o = offs[i * nb + b];
                if (o !== o) { fm[b] = NaN; continue; }
                const src = Math.max(0, d - lags[i * nb + b]);
                let v = o;
                const c0 = (i * nb + b) * chunk.rank;
                for (let j = 0; j < chunk.rank; j++) v += coefs[c0 + j] * modes.Vt[j * modes.nDays + src];
                fm[b] = v;
            }

            for (let f = 0; f < steps.length; f++) {
                const p = steps[f];
                let v = NaN;
                if (p.kind === "calendar") v = cal[p.name];
                else if (p.kind === "static") v = chunk.statics[p.at];
                else if (p.kind === "weather") v = fm[p.bucket];
                else if (p.kind === "yearly" || p.kind === "expT") v = block[p.at];
                else if (p.kind === "cross") v = ci === undefined ? NaN : cross.cols[p.name][ci];
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

    const basin = (comid) => B().memo(`basin:${comid}`, () => B().buffer(`forecast/basins/${comid}.bin`).then(decodeBasin));

    /* ── the basin overlay ────────────────────────────────────────────────────
     * build_forecast.pack_basin's layout: [n:i32][lon:f32 x n][lat:f32 x n], one exterior ring. Returned as a GeoJSON Feature rather than a Leaflet positions array because that is what the server-side layer passed and the styling options carry over unchanged.
     */
    function decodeBasin(buf) {
        const n = new DataView(buf).getInt32(0, true);
        const lon = new Float32Array(buf.slice(4, 4 + n * 4));
        const lat = new Float32Array(buf.slice(4 + n * 4, 4 + n * 8));
        const ring = new Array(n);
        for (let i = 0; i < n; i++) ring[i] = [lon[i], lat[i]];  // GeoJSON is lon,lat
        return {type: "Feature", properties: {}, geometry: {type: "Polygon", coordinates: [ring]}};
    }

    /* ── the forecast ─────────────────────────────────────────────────────────
     * The numeric core, kept separate from the callback that formats it: the parity harness scores a
     * reach here and compares against model_interface.forecast_virtual_site, which it could not do
     * through a function that returns Dash components.
     */

    /* Mirrors deploy.predict.threshold_for_beta -- the beta_table row nearest the slider, or null for a model that ships untuned, which is the case the caller renders without an alarm threshold. Ties keep the FIRST row, as Python's min() does. */
    function operatingPoint(meta, beta) {
        const table = meta.beta_table;
        if (!table || !table.length) return null;
        let best = table[0];
        for (const row of table) {
            if (Math.abs(row.beta - beta) < Math.abs(best.beta - beta)) best = row;
        }
        return {beta: best.beta, tau: best.tau, recall: best.recall, fdr: best.fdr, base_rate: meta.base_rate};
    }

    /* Score one snapped reach for one year. -> {dates, reg, clf, op, comid} with `dates` as epoch days.
     *
     * Both models are walked over the same spine (it comes from the shared weather modes, so the two tasks cannot disagree about which days exist), each against its OWN feature list -- the light REG and CLF recipes do not carry the same columns.
     *
     * Throws with a legible message rather than returning a null: a reach whose row is missing is a real outcome (two COMIDs are NLDI tombstones, and the reach store stops at stream order 3), and the callback turns the message into the panel's text.
     */
    function forecastFor(comid, outlet, year, beta) {
        return Promise.all([chunkMeta(), modes(), crossSite(), model("reg"), model("clf")]).then(
            ([meta, md, cross, reg, clf]) => {
                const cid = chunkOf(outlet[0], outlet[1], meta);
                return chunk(cid).then((ch) => {
                    if (!ch.at.has(comid)) {
                        throw new Error(`no precomputed row for COMID ${comid} — it is outside the forecastable set`);
                    }
                    const at = {chunk: ch, i: ch.at.get(comid)};
                    const r = assemble(at, year, "reg", reg.feat, meta, md, cross);
                    const c = assemble(at, year, "clf", clf.feat, meta, md, cross);
                    return {
                        comid: comid,
                        dates: r.dates,
                        reg: predict(reg.booster, r.rows),
                        clf: predict(clf.booster, c.rows),
                        op: operatingPoint(clf, beta),
                    };
                });
            }
        );
    }

    /* ── the figure ───────────────────────────────────────────────────────────
     * Two stacked panels, NO covariates: predicted nitrate over the 10 mg/L line, and P(violation)
     * with the alarm band shaded behind it. Built here rather than shipped as a figure spec because
     * it is a function of the beta slider, which moves without refetching anything.
     *
     * Hand-built rather than make_subplots': the domains, the shared x and the two subplot titles are
     * the whole of what that helper was doing here.
     */
    const isoDay = (epochDay) => new Date(epochDay * 86400000).toISOString().slice(0, 10);

    function betaText(op, daysOver, sep) {
        if (!op) return `${daysOver} days predicted ≥ 10 mg/L`;
        return `Raises alarm when violation probability ≥ ${op.tau.toFixed(2)} –– ${daysOver} days predicted ≥ 10 mg/L`
            + `${sep}Expected: catches ~${Math.round(op.recall * 100)}% of violations, ~${Math.round(op.fdr * 100)}% of alarms false`;
    }

    /* `opts` carries what differs between the on-screen figure and the downloaded one: the latter is
     * rendered at 2500x2000 with the title and the operating-point sentence baked in, since a PNG
     * leaves the page with nothing around it to explain what it shows. */
    function figureSpec(f, year, consts, opts) {
        const o = opts || {};
        const x = f.dates.map(isoDay);
        const style = consts.forecast;
        const big = !!o.presentation;
        const s = (n) => (big ? n * 2.6 : n);  // one scale for every font/width, so the two renders match

        const data = [
            {type: "scatter", mode: "lines", x: x, y: Array.from(f.reg), name: "nitrate",
             line: {color: style.line, width: s(1.5)}, hovertemplate: "%{x}<br>%{y:.2f} mg/L<extra></extra>"},
        ];
        if (f.op) {  // the alarm band goes in first so it sits behind the probability trace
            data.push({type: "scatter", mode: "lines", x: x, y: Array.from(f.clf, (p) => (p >= f.op.tau ? 1 : 0)),
                       xaxis: "x2", yaxis: "y2", line: {width: 0, shape: "hv"}, fill: "tozeroy",
                       fillcolor: style.alarm_fill, name: "alarm", showlegend: false, hoverinfo: "skip"});
        }
        data.push({type: "scatter", mode: "lines", x: x, y: Array.from(f.clf), name: "P(viol)",
                   xaxis: "x2", yaxis: "y2", line: {color: style.line, width: s(1.5)},
                   hovertemplate: "%{x}<br>P = %{y:.2f}<extra></extra>"});

        const shapes = [
            // the 10 mg/L standard, on the nitrate panel
            {type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 10, y1: 10,
             line: {color: style.line, width: s(1), dash: "dash"}, opacity: 0.4},
        ];
        const annotations = [
            {text: big ? "Predicted nitrate (mg/L)" : `Predicted nitrate (mg/L) — ${year}`,
             xref: "paper", yref: "paper", x: 0.5, y: 1.0, xanchor: "center", yanchor: "bottom",
             showarrow: false, font: {size: s(12)}},
            {text: "P(violation ≥ 10 mg/L)", xref: "paper", yref: "paper", x: 0.5, y: 0.44,
             xanchor: "center", yanchor: "bottom", showarrow: false, font: {size: s(12)}},
        ];
        if (f.op) {
            shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: f.op.tau, y1: f.op.tau,
                         line: {color: "#555", width: s(1), dash: "dot"}});
            annotations.push({text: `τ=${f.op.tau.toFixed(2)}`, xref: "paper", x: 1, yref: "y2", y: f.op.tau,
                              xanchor: "right", yanchor: "bottom", showarrow: false, font: {size: s(9), color: "#555"}});
        }
        if (o.title) {
            annotations.push({text: o.title, xref: "paper", yref: "paper", x: 0.5, y: 1.06,
                              xanchor: "center", yanchor: "bottom", showarrow: false, font: {size: s(16)}});
        }
        if (o.caption) {
            annotations.push({text: o.caption.replace(/\n/g, "<br>"), xref: "paper", yref: "paper", x: 0.5, y: -0.1,
                              xanchor: "center", yanchor: "top", showarrow: false, font: {size: s(11), color: "#333"}});
        }

        return {
            data: data,
            layout: {
                height: o.height || 340,
                margin: o.margin || {t: 30, b: 30, l: 44, r: 44},
                showlegend: false,
                font: {size: s(10)},
                // one x, shown once: `matches` keeps the two panels locked when the user zooms either
                xaxis: {domain: [0, 1], anchor: "y", matches: "x2", showticklabels: false},
                xaxis2: {domain: [0, 1], anchor: "y2"},
                yaxis: {domain: [0.57, 1], title: {text: "mg/L"}},
                yaxis2: {domain: [0, 0.43], range: [0, 1], title: {text: "P(violation)"}},
                shapes: shapes,
                annotations: annotations,
            },
        };
    }

    /* A pin label with no site name to use, matching forecast_panel._coord_label's '42.03°N, 93.62°W'. */
    function coordLabel(lat, lon) {
        return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
    }

    const P = (text, style) => B().h("P", {children: text, style: style});

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

        /* Run the forecast at the snapped reach. The server callback this replaces called NLDI,
         * delineated a basin, built the features and ran XGBoost, all per click; everything but the
         * arithmetic is now precomputed, so this is a chunk fetch plus two tree walks.
         *
         * A failure here is a MESSAGE, not a throw: the reach store stops at stream order 3 and two
         * COMIDs are NLDI tombstones, so "no row for this reach" is a real answer the panel has to be
         * able to give.
         */
        runForecast: function (nClicks, regionGeom, year, beta, consts) {
            const hideDl = {display: "none"};
            const fail = (msg, color) => [NO(), {display: "none"}, P(msg, {color: color || "#888", fontSize: "12px"}),
                                          [], null, hideDl];
            if (!regionGeom || regionGeom.type !== "Point") {
                return fail("Drop a pin first (Pin drop selection mode).");
            }
            if (!regionGeom.snap) {
                return fail("No stream within 10 km of that pin — there is no reach to forecast.");
            }
            const snap = regionGeom.snap;
            const [lng, lat] = regionGeom.coordinates;
            const label = coordLabel(lat, lng);

            return Promise.all([forecastFor(snap.comid, snap.outlet, Number(year), Number(beta)), basin(snap.comid)])
                .then(([f, poly]) => {
                    const daysOver = f.reg.reduce((k, v) => k + (v >= 10 ? 1 : 0), 0);
                    const peak = f.clf.length ? Math.max.apply(null, Array.from(f.clf)) : NaN;
                    const pct = (v) => `${Math.round(v * 100)}%`;

                    const lines = [P(`Peak P(violation): ${pct(peak)} · ${daysOver} days predicted ≥ 10 mg/L`,
                                     {fontSize: "12px", marginTop: "6px", marginBottom: "2px"})];
                    if (f.op) {
                        const alarms = f.clf.reduce((k, p) => k + (p >= f.op.tau ? 1 : 0), 0);
                        lines.push(P(
                            `β=${beta} → alarm at P ≥ ${f.op.tau.toFixed(2)} · ${alarms} alarm days. `
                            + `Expected: catches ~${pct(f.op.recall)} of violations · ~${pct(f.op.fdr)} of alarms false`
                            + (f.op.base_rate == null ? "." : ` (at ~${pct(f.op.base_rate)} base-rate prevalence).`),
                            {fontSize: "12px", marginTop: "0", color: "#444"}
                        ));
                    }

                    const payload = {dates: Array.from(f.dates), reg: Array.from(f.reg), clf: Array.from(f.clf),
                                     op: f.op, year: Number(year), label: label, comid: f.comid};
                    return [
                        figureSpec(f, Number(year), consts),
                        {display: "block", height: "340px"},
                        B().h("Div", {children: lines}),
                        [B().dl("GeoJSON", {data: poly, options: {style: consts.forecast.basin_style}})],
                        payload,
                        {display: "block", marginTop: "6px"},
                    ];
                })
                .catch((e) => fail(`Forecast failed: ${e.message || e}`, "#c00"));
        },

        /* The download. Plotly renders the PNG the page already has the figure for, which is what
         * removes the last server dependency -- the callback this replaces rendered it with
         * matplotlib. Same two panels at 2500x2000, with the title and the operating-point sentence
         * baked in, because a PNG leaves the page without them.
         */
        downloadForecast: function (nClicks, payload, consts) {
            if (!payload) return NO();
            const f = {dates: payload.dates, reg: payload.reg, clf: payload.clf, op: payload.op};
            const daysOver = f.reg.reduce((k, v) => k + (v >= 10 ? 1 : 0), 0);
            const spec = figureSpec(f, payload.year, consts, {
                presentation: true,
                height: 2000,
                margin: {t: 220, b: 260, l: 150, r: 150},
                title: payload.label
                    ? `Predictions for ${payload.label} using ${payload.year} example data`
                    : `Predictions using ${payload.year} example data`,
                caption: betaText(f.op, daysOver, "\n"),
            });
            return window.Plotly.toImage(spec, {format: "png", width: 2500, height: 2000}).then((url) => ({
                content: url.split(",")[1],
                filename: `nitrate_forecast_${payload.year}.png`,
                base64: true,
                type: "image/png",
            }));
        },

        _pinLayer: pinLayer,
        // exported for the parity harness, which scores a reach headlessly and compares against
        // model_interface.forecast_virtual_site
        _forecastFor: forecastFor,
        _plan: plan,
        _figureSpec: figureSpec,
        _operatingPoint: operatingPoint,
        _decodeBasin: decodeBasin,
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
