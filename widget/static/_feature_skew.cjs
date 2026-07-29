/* Which model features the browser can resolve from the shipped bundle, for widget/static/export.py.
 *
 * The JS side of deploy.predict._assert_no_skew. A feature the browser cannot place gets NaN, and NaN is a legitimate value here (an absent distance ring), so a renamed or newly bucketed column reads as a plausible forecast with that column silently missing -- which is exactly what a stale assemble() did with fuel_moisture_1000h_b0/_b1/_b2.
 *
 * Absent rings are NOT this: plan() resolves by name against the packed layout, so every ring has a name whether or not any cell falls in it.
 *
 * Usage: node _feature_skew.cjs   ->   {"reg": [...unresolved], "clf": [...]} on stdout
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const WIDGET = path.resolve(__dirname, "..");

global.fetch = async (url) => {
    const f = path.join(WIDGET, url);
    if (!fs.existsSync(f)) return {ok: false, status: 404};
    const b = fs.readFileSync(f);
    return {
        ok: true,
        status: 200,
        json: async () => JSON.parse(b.toString()),
        arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
    };
};

global.window = global;
global.dash_clientside = {no_update: "NO_UPDATE", callback_context: {triggered: []}};
for (const f of ["bundle.js", "forecast.js"]) {
    vm.runInThisContext(fs.readFileSync(path.join(WIDGET, "assets", "clientside", f), "utf8"), {filename: f});
}
const F = global.dash_clientside.forecast;
const B = global.dash_clientside.bundle;

(async () => {
    const meta = await B.json("forecast/reach_chunks.json").catch(() => null);
    if (!meta || !meta.chunks || !meta.chunks.length) {
        throw new Error("no packed reach chunks in the bundle -- run build_reaches, then build_bundle --only reaches");
    }
    const chunk = F._decodeChunk(await B.buffer(`forecast/reaches/${meta.chunks[0]}.bin`));
    const year = meta.years[0];

    const out = {};
    for (const task of ["reg", "clf"]) {
        const model = await B.json(`models/${task}.json`);
        const steps = F._plan(chunk, 0, year, task, model.feat, meta);
        out[task] = steps.map((p, k) => (p.kind === null ? model.feat[k] : null)).filter(Boolean);
    }
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    // Message first: export.py surfaces only the head of this, and a stack there says nothing useful.
    process.stderr.write(String((e && e.message) || e) + "\n" + String((e && e.stack) || "") + "\n");
    process.exit(1);
});
