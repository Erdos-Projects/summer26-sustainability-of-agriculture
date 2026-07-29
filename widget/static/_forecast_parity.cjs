/* The browser's forecast, run headlessly under Node, for widget/static/check_forecast.py.
 *
 * Loads the two clientside modules exactly as the page does -- same files, same order, same bundle
 * accessors -- against a `fetch` that reads widget/assets/data/ off disk. So what this scores is the
 * SHIPPED artifact set, not a reimplementation of it: a mispacked chunk or a stale manifest fails
 * here the same way it would in a browser.
 *
 * Usage: node _forecast_parity.cjs <cases.json>   ->   results JSON on stdout
 * A case is {comid, lat, lon, year, beta} and names the reach outright: the snap is NOT exercised here. Snapping is a separate question (_make_basins.snap_comid vs forecast.js::snapComid), and letting it run would mean each side forecasting whichever reach it chose, which measures geometry rather than implementation.
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
    const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const out = [];
    for (const c of cases) {
        try {
            const f = await F._forecastFor(c.comid, [c.lat, c.lon], c.year, c.beta);
            out.push({
                comid: c.comid, year: c.year,
                dates: Array.from(f.dates), reg: Array.from(f.reg), clf: Array.from(f.clf), op: f.op,
            });
        } catch (e) {
            out.push({comid: c.comid, error: String(e.message || e)});
        }
    }
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    process.stderr.write(String(e && e.stack ? e.stack : e) + "\n");
    process.exit(1);
});
