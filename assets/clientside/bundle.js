/* Read side of the precomputed asset bundle, for clientside callbacks.
 *
 * The JS mirror of widget/bundle.py. Everything a converted callback needs comes from widget/assets/data/, which build_bundle.py writes and export.py copies into the static site.
 *
 * URLS MUST STAY RELATIVE ("assets/data/..."). The site is published as a GitHub project page under user.github.io/<repo>/, and dash2html's index.html patch does not rewrite URLs embedded in the _dash-layout JSON or built at runtime here. A leading slash works locally and 404s only once deployed, which is the worst way to find out.
 *
 * Caching is per-URL and permanent: the bundle is immutable for the lifetime of a page load, and several callbacks read the same artifacts (sites.json in particular) on every interaction.
 */

(function () {
    const cache = new Map();

    /* Fetch once, then hand back the same promise. Storing the PROMISE rather than the resolved value collapses concurrent callers -- several callbacks fire together on a selection change and would otherwise each start their own request for sites.json. */
    function once(url, loader) {
        if (!cache.has(url)) cache.set(url, loader(url));
        return cache.get(url);
    }

    const dataUrl = (rel) => "assets/data/" + String(rel).replace(/^\/+/, "");
    const assetUrl = (rel) => "assets/" + String(rel).replace(/^\/+/, "");

    const json = (rel) => once(dataUrl(rel), (u) => fetch(u).then((r) => {
        if (!r.ok) throw new Error(`bundle: ${u} -> ${r.status}`);
        return r.json();
    }));

    /* Decode the length-prefixed pack build_bundle._pack writes: [n:int32] followed by each column's raw bytes, in the order documented at the call site that produced the file. `types` names the dtypes in that same order, e.g. ["i4","i4","f4","f4"] for site_cells.
     *
     * Returns {n, columns:[TypedArray, ...]}. The typed arrays are VIEWS on one ArrayBuffer, so this copies nothing; callers must not mutate them (the buffer is cached and shared). */
    const CTOR = {i4: Int32Array, u4: Uint32Array, f4: Float32Array, u2: Uint16Array};

    function pack(rel, types) {
        return once(dataUrl(rel) + "#" + types.join(","), () =>
            fetch(dataUrl(rel)).then((r) => {
                if (!r.ok) throw new Error(`bundle: ${dataUrl(rel)} -> ${r.status}`);
                return r.arrayBuffer();
            }).then((buf) => {
                const n = new DataView(buf).getInt32(0, true);
                let off = 4;
                const columns = types.map((t) => {
                    const C = CTOR[t];
                    if (!C) throw new Error(`bundle: unknown dtype ${t}`);
                    const a = new C(buf, off, n);
                    off += n * C.BYTES_PER_ELEMENT;
                    return a;
                });
                return {n, columns};
            })
        );
    }

    /* A Dash component as plain JSON. dash-renderer instantiates {namespace, type, props} exactly as it would a Python-constructed one, so building a tree here is object literals -- no React and no component imports. Lives in the shared module because three namespaces now emit components. */
    const component = (namespace, type, props) => ({namespace: namespace, type: type, props: props});
    const dl = (type, props) => component("dash_leaflet", type, props);
    const h = (type, props) => component("dash_html_components", type, props);
    /* dcc components built here rather than in the Python layout load their async chunk on first
     * use instead of at page boot -- which for Markdown+MathJax is 2.4 MB deferred. */
    const dcc = (type, props) => component("dash_core_components", type, props);

    /* Named accessors for the artifacts more than one callback needs. Layouts documented in build_bundle.py at the _pack call that writes each one -- keep the dtype lists in step. */
    const sites = () => json("sites.json");
    const palette = () => json("palette.json");
    const manifest = () => json("manifest.json");

    // [n][node_id:i4][global_node_id:i4][frac_cell_in_basin:f4][dist_to_sensor:f4]
    const siteCells = (uid) => pack(`site_cells/${uid}.bin`, ["i4", "i4", "f4", "f4"]);
    // [n][days_since_epoch:i4][nitrate:f4][precip:f4]
    const series = (uid, interval) => pack(`series/${uid}_${interval}.bin`, ["i4", "f4", "f4"]);
    // [n][global_node_id:i4][surplus_kgha:f4][total_kg_N:f4]
    const surplus = (year) => pack(`covariates/surplus_${year}.bin`, ["i4", "f4", "f4"]);

    /* crops_{year}.bin is [n][global_node_id:i4][counts:u4 x 8] -- one interleaved block rather than parallel columns, so it does not fit `pack` and is decoded here. Class order is manifest.coverage.crop_classes. */
    function crops(year) {
        const rel = `covariates/crops_${year}.bin`;
        return once(dataUrl(rel) + "#crops", () =>
            fetch(dataUrl(rel)).then((r) => r.arrayBuffer()).then((buf) => {
                const n = new DataView(buf).getInt32(0, true);
                return {
                    n,
                    ids: new Int32Array(buf, 4, n),
                    // Row-major, one row per cell. The row width is coverage.crop_classes.length -- taken from the manifest by the caller rather than hardcoded here, so adding a crop class does not silently misalign every row.
                    counts: new Uint32Array(buf, 4 + 4 * n),
                };
            })
        );
    }

    /* One reach's NLDI basin outline as a GeoJSON Feature, from build_forecast.pack_basin: [n:i32][lon:f32 x n][lat:f32 x n], a single exterior ring. Shared because two namespaces draw it -- the forecast's own overlay and the map's basin layers -- off the one cached fetch. */
    function reachBasin(comid) {
        return memo(`basin:${comid}`, () => buffer(`forecast/basins/${comid}.bin`).then((buf) => {
            const n = new DataView(buf).getInt32(0, true);
            const lon = new Float32Array(buf.slice(4, 4 + n * 4));
            const lat = new Float32Array(buf.slice(4 + n * 4, 4 + n * 8));
            const ring = new Array(n);
            for (let i = 0; i < n; i++) ring[i] = [lon[i], lat[i]];  // GeoJSON is lon,lat
            return {type: "Feature", properties: {}, geometry: {type: "Polygon", coordinates: [ring]}};
        }));
    }

    /* Index a decoded pack's id column for O(1) joins. Cached alongside the pack because the rain grid joins the same covariate arrays on every year/mode change. */
    function indexById(rel, idColumn) {
        return once(rel + "#index", () => {
            const m = new Map();
            for (let i = 0; i < idColumn.length; i++) m.set(idColumn[i], i);
            return Promise.resolve(m);
        });
    }

    /* sites.json keyed by site_uid. Several panels look sites up by uid rather than iterating. */
    const sitesById = () => once("sites.json#byuid", () =>
        sites().then((rows) => new Map(rows.map((r) => [r.site_uid, r])))
    );

    /* Raw bytes of an artifact, cached. For packs whose layout belongs to another namespace (the
     * forecast ones), so each decoder can live beside the code that reads it. */
    const buffer = (rel) => once(dataUrl(rel) + "#buf", () => fetch(dataUrl(rel)).then((r) => {
        if (!r.ok) throw new Error(`bundle: ${dataUrl(rel)} -> ${r.status}`);
        return r.arrayBuffer();
    }));

    /* Memoise anything by a string key, sharing this one cache -- so a namespace can hold a decoded
     * structure (a snap index, a booster) without growing its own cache and its own bugs. */
    const memo = (key, factory) => once("memo:" + key, factory);

    /* The global Voronoi grid indexed by global_node_id.
     *
     * This is the one heavy fetch in the bundle (6.4 MB): geometry ships ONCE for all 22,877 cells rather than per site, which is what keeps the 83 per-site indices at 552 KB in total. It is fetched lazily -- only when the rain grid is first switched on -- and then cached with everything else, so a session that never opens the grid never pays for it. */
    const gridIndex = () => once("grid.geojson#index", () =>
        json("grid.geojson").then((fc) => new Map(fc.features.map((f) => [f.properties.global_node_id, f])))
    );

    /* Dash's clientside callback_context reports the trigger as a STRING prop_id. For a pattern-matching id that string is the serialised dict in Dash's canonical (key-sorted) order, e.g. '{"index":"WQS0003","type":"iwqis-marker"}.n_clicks' -- so it has to be split at the LAST dot and parsed. Returns the id (object or string), or null when nothing triggered. */
    function triggeredId() {
        const t = (window.dash_clientside.callback_context || {}).triggered || [];
        if (!t.length || !t[0].prop_id) return null;
        const prop = t[0].prop_id;
        const raw = prop.slice(0, prop.lastIndexOf("."));
        if (!raw.startsWith("{")) return raw;
        try {
            return JSON.parse(raw);
        } catch (e) {
            return raw;
        }
    }

    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.bundle = {
        dataUrl, assetUrl, json, pack, buffer, memo, indexById, triggeredId,
        component, dl, h, dcc,
        sites, sitesById, palette, manifest, siteCells, series, surplus, crops, gridIndex, reachBasin,
        no_update: window.dash_clientside.no_update,
    };
})();
