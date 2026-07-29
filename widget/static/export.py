"""Build the static site: snapshot the Dash app with dash2html, then graft the asset bundle in.

HOW THE SNAPSHOT WORKS (dash2html v1, read from its source). It captures the werkzeug access log, picks out GETs to `/_*` (i.e. `_dash-layout` and `_dash-dependencies`), downloads index.html, and injects a `window.fetch` monkey-patch that answers those two requests from inlined JSON. The consequence that matters: **dash-renderer still boots normally in the browser**, so `app.clientside_callback` callbacks keep working with no server. Server-side callbacks POST to `/_dash-update-component`, which the patch does not intercept -- they fail silently. Hence the callback audit below: a Python callback that ships is a control that looks alive and is not.

TWO THINGS WE DO NOT TAKE FROM THE LIBRARY:

1. Its entry point calls `app.run_server(...)`, removed in Dash 3 (this repo runs Dash 4), so `dash2html.dash2html()` raises AttributeError. We drive `app.run(...)` ourselves and call its `process_log` / `make_static` helpers, which are fine.

2. Its asset set is whatever happened to be logged, so anything first fetched by a user interaction is missed. Our ~700-file data bundle is copied in from build_bundle's manifest instead of being discovered.

Usage
-----
    python -m widget.static.export                  # build bundle if stale, then snapshot
    python -m widget.static.export --skip-bundle    # snapshot only (bundle already current)
    python -m widget.static.export --audit-only     # just report which callbacks are server-side
    python -m widget.static.export --out site/
"""

import argparse
import io
import json
import logging
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent  # widget/static
_WIDGET = _HERE.parent
_ROOT = _WIDGET.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WIDGET))  # the app uses flat imports (`from layout import ...`)

# The published site. Gitignored: it is generated, it is tens of megabytes, and committing it to
# main would grow the history by that much on every rebuild. widget/static/publish.py pushes its
# contents to an orphan gh-pages branch instead, which GitHub Pages serves.
DEFAULT_OUT = _ROOT / "static-site"
DEFAULT_PORT = 8099

# Routes we request explicitly rather than hoping a browser touches them. dash-renderer would fetch the two _dash-* endpoints itself; requesting them here is what puts them in the access log, which is how dash2html learns to inline them.
SNAPSHOT_ROUTES = ["/", "/_dash-layout", "/_dash-dependencies"]

# Callbacks that are allowed to remain server-side, because the features they drive are deferred until the lighter recipe/model lands. Anything else server-side is a bug: it would render as a live control that silently does nothing on the published site.
DEFERRED_OUTPUTS = {
    "forecast-graph.figure",  # run_forecast: NLDI + feature build + XGBoost
    "forecast-download.data",  # download_forecast: server-side matplotlib render
    "pin-basin-layer.children",  # v1 pin delineation: flowline snap + NLDI
    "pin-basin-v3-layer.children",  # v3 pin delineation: D8 raster flood-fill
}


# ──
# app under test ────────────────────────────────────────────────────────────


def _build_app():
    """    Construct the Dash app exactly as widget/app.py does.

    assets_folder is passed EXPLICITLY. Dash resolves it relative to the module that constructs the app, and this module lives in widget/static/ rather than widget/ -- so `Dash(__name__)` here finds no assets folder at all and silently omits custom.css and dashExtensions_default.js. The latter carries the rain grid's style/hover/tooltip functions, so the snapshot would build and look fine while the grid rendered unstyled and untooltipped.
    """
    from dash import Dash

    from layout import build_layout
    from components import map_panel, info_panel, forecast_panel

    app = Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(_WIDGET / "assets"))
    app.layout = build_layout()
    map_panel.register_callbacks(app)
    info_panel.register_callbacks(app)
    forecast_panel.register_callbacks(app)
    return app


class _LogCapture(logging.Handler):
    """    Collect werkzeug access lines into a buffer in dash2html's expected shape.

    dash2html swaps `sys.stderr` for a StringIO before starting the server. That is fragile: a StreamHandler binds to whatever `sys.stderr` was when it was created, so the swap only works if nothing has logged yet. Attaching a handler to the `werkzeug` logger is equivalent for process_log's purposes (it only greps for `"GET `) and does not depend on import ordering.
    """

    def __init__(self):
        super().__init__()
        self.buf = io.StringIO()

    def emit(self, record):
        self.buf.write(record.getMessage() + "\n")


def _serve(app, port):
    """Run the app in a daemon thread; return the log capture once it answers."""
    cap = _LogCapture()
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(logging.INFO)
    wlog.addHandler(cap)

    threading.Thread(
        target=lambda: app.run(debug=False, port=port, use_reloader=False),
        daemon=True,
    ).start()

    base = f"http://127.0.0.1:{port}/"
    for _ in range(100):  # ~10 s
        try:
            requests.get(base, timeout=1)
            return cap, base
        except requests.RequestException:
            time.sleep(0.1)
    raise RuntimeError(f"app did not start on port {port}")


# ──
# callback audit ────────────────────────────────────────────────────────────


def audit_callbacks(dependencies) -> dict:
    """    Split the dependency graph into clientside vs server-side callbacks.

    A server-side callback in a static build is worse than a missing one: the control renders, the user clicks it, and nothing happens with no error surfaced.
    """
    clientside, server = [], []
    for cb in dependencies:
        out = cb.get("output", "")
        (clientside if cb.get("clientside_function") else server).append(out)
    unexpected = [o for o in server if not any(d in o for d in DEFERRED_OUTPUTS)]
    return {"clientside": clientside, "server": server, "unexpected": unexpected}


def _report_audit(audit) -> None:
    n_c, n_s = len(audit["clientside"]), len(audit["server"])
    print(f"\n── callback audit ──  {n_c} clientside / {n_s} server-side")
    if audit["server"]:
        deferred = [o for o in audit["server"] if o not in audit["unexpected"]]
        if deferred:
            print(f"  deferred (expected, forecast/pin): {len(deferred)}")
            for o in sorted(deferred):
                print(f"    - {o}")
        if audit["unexpected"]:
            print(f"  NOT YET CONVERTED -- these will be dead controls on the static site: {len(audit['unexpected'])}")
            for o in sorted(audit["unexpected"]):
                print(f"    ! {o}")


# ──
# export ────────────────────────────────────────────────────────────────────


def _copy_bundle(out: Path) -> int:
    """    Copy every artifact the manifest lists into the built site.

    Not discovered from the access log: most of these are fetched lazily by a user interaction the snapshot never performs, so log-based discovery would silently ship a site with missing data.
    """
    import bundle

    man = bundle.manifest()
    src_root = bundle.DATA_DIR
    dst_root = out / "assets" / "data"

    # Orphan check: a file that exists in the bundle but is absent from the manifest never gets copied, so it 404s on the published site with nothing to point at the cause. (This is not hypothetical -- a mid-run builder failure once left grid.geojson on disk but unrecorded.)
    listed = {(src_root / rel).resolve() for rel in man.get("artifacts", {})}
    on_disk = {p.resolve() for p in src_root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if orphans := sorted(on_disk - listed):
        raise RuntimeError(
            f"{len(orphans)} bundle file(s) are not in the manifest and would be dropped, e.g. "
            f"{[str(o.relative_to(src_root)) for o in orphans[:5]]}. Re-run build_bundle."
        )

    n = 0
    for rel in man.get("artifacts", {}):
        src = (src_root / rel).resolve()
        # "../iowa_flowlines.geojson" entries live in assets/, not assets/data/
        dst = (dst_root / rel).resolve()
        if not src.exists():
            raise FileNotFoundError(f"manifest lists {rel} but {src} is missing; re-run build_bundle")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    shutil.copy2(bundle.MANIFEST, dst_root / "manifest.json")
    return n + 1


# Chunks whose absence breaks something visible, asserted after the copy. dash_leaflet's GeoJSON is the load-bearing one: it backs the hydro, basin, rain-grid, marker and pin layers, so without it the map renders with no data at all.
_REQUIRED_CHUNKS = ["async-GeoJSON", "async-graph", "async-dropdown", "async-slider", "plotly.min.js"]

_FINGERPRINT_RE = re.compile(r'splice\(1,\s*0,\s*"(v[0-9_]+m\d+)"\)')


def _fingerprints_in(directory: Path) -> set:
    """Chunk fingerprints embedded in the non-async bundles of `directory`.

    Each Dash component bundle carries a webpack shim that rewrites its own lazy-chunk URLs, hardcoding a fingerprint: `a.splice(1, 0, "v4_4_0m1783096861")`. That value is what the BROWSER will ask for, and it is not the same as the fingerprint Dash serves the eager bundles under (v4_4_0m1784603323 here) -- the running server hides the difference because check_fingerprint strips any fingerprint, but a static host cannot. A directory can legitimately hold more than one (dash_table ships bundles from two builds), so return them all and write the chunk under each.
    """
    out = set()
    for js in directory.glob("*.js"):
        if js.name.startswith("async-"):
            continue
        if m := _FINGERPRINT_RE.search(js.read_text(errors="ignore")):
            out.add(m.group(1))
    return out


def _fingerprinted(name: str, fp: str) -> str:
    """Insert `fp` after the FIRST dot-segment, matching dash.fingerprint.build_fingerprint.

    It splits on the first ".", so async-EditControl.ts.js becomes async-EditControl.<fp>.ts.js -- not ...ts.<fp>.js. Getting this backwards yields a 500 from the dev server and a 404 from a static host.
    """
    head, _, ext = name.partition(".")
    return f"{head}.{fp}.{ext}"


def _layout_packages(app) -> set:
    """PACKAGE names whose chunks the export needs, derived from the layout.

    Note the two namings differ and conflating them silently drops chunks: components report a library namespace (dash_core_components, dash_html_components), while app.registered_paths is keyed by the installed package (dash). So `dash` is kept unconditionally -- dcc and html always render -- and the walk only decides the optional third-party packages, which happen to use the same string for both (dash_leaflet, dash_extensions). plotly is kept because dcc.Graph pulls it in via dash-renderer without ever appearing as a component namespace.
    """
    from plotly.utils import PlotlyJSONEncoder

    seen = {"dash", "plotly"}

    def walk(o):
        if isinstance(o, dict):
            if ns := o.get("namespace"):
                seen.add(ns)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(json.dumps(app.layout, cls=PlotlyJSONEncoder)))
    return seen


def _copy_async_chunks(app, out: Path) -> int:
    """Copy the lazily-loaded component chunks into the export.

    dash2html discovers assets from the access log, which only ever sees the EAGER bundles: Dash lazy-loads dcc.Graph, dcc.Dropdown, dcc.Slider and every dash_leaflet layer as separate webpack chunks, requested on first render. A `requests` walk never triggers them, and even a browser walk would only capture whichever ones the walk happened to touch -- so this enumerates them from app.registered_paths rather than hoping.

    Scope: packages the layout instantiates. Within those, EVERY async chunk is copied, including ones for components we do not render (dcc's mathjax, dash_extensions' Mermaid). That is deliberate ~5 MB of dead weight: deciding a chunk is unused requires a component-type-to-chunk-name mapping Dash does not publish, and getting it wrong reproduces exactly the silent "renders but does not work" failure this function exists to fix. Size is not the binding constraint at ~13 MB against a 1 GB budget.
    """
    import importlib.util

    namespaces = _layout_packages(app)
    n = 0
    for namespace, paths in app.registered_paths.items():
        if namespace not in namespaces:
            continue
        spec = importlib.util.find_spec(namespace)
        if spec is None or not spec.submodule_search_locations:
            continue
        pkg_dir = Path(list(spec.submodule_search_locations)[0])
        for rel in sorted(paths):
            if rel.endswith(".map"):
                continue  # source maps are devtools-only
            src = pkg_dir / rel
            if not src.exists():
                continue
            rel_path = Path(rel)
            dst_dir = out / "_dash-component-suites" / namespace / rel_path.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            if rel_path.name.startswith("async-"):
                # Fingerprinted name ONLY. The webpack shim rewrites the URL whenever the loading script sits under /_dash-component-suites/, which it always does here, so the plain name is never requested and would just double the payload.
                for fp in _fingerprints_in(src.parent):
                    shutil.copy2(src, dst_dir / _fingerprinted(rel_path.name, fp))
                    n += 1
            else:
                # Non-async entries (plotly.min.js) are referenced by dash-renderer at their plain path.
                plain = dst_dir / rel_path.name
                if not plain.exists():
                    shutil.copy2(src, plain)
                    n += 1

    present = {p.name for p in (out / "_dash-component-suites").rglob("*.js")}
    if missing := [c for c in _REQUIRED_CHUNKS if not any(c in p for p in present)]:
        raise RuntimeError(f"async chunks missing from the export: {missing}; the site would render without them")
    return n


def export(out: Path, port=DEFAULT_PORT, skip_bundle=False, audit_only=False) -> None:
    from dash2html.utils import make_static, process_log

    if not skip_bundle and not audit_only:
        from widget.static import build_bundle

        build_bundle.main()

    app = _build_app()
    cap, base = _serve(app, port)

    for route in SNAPSHOT_ROUTES:
        r = requests.get(base.rstrip("/") + route, timeout=30)
        r.raise_for_status()

    deps = requests.get(base + "_dash-dependencies", timeout=30).json()
    audit = audit_callbacks(deps)
    _report_audit(audit)
    if audit_only:
        return

    json_paths, extra_res = process_log(cap.buf.getvalue())
    if not json_paths:
        raise RuntimeError(
            "no /_dash-* routes found in the access log -- the snapshot would have nothing to "
            "inline. Werkzeug logging may have been silenced."
        )
    print(f"\ninlining {sorted(json_paths)}; {len(extra_res)} other logged path(s)")

    zbuf = make_static(base, json_paths, extra_res)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zbuf) as z:
        z.extractall(out)

    n = _copy_bundle(out)
    n_chunks = _copy_async_chunks(app, out)
    # GitHub Pages runs Jekyll by default, which drops paths beginning with "_" -- that would delete the entire Dash runtime under _dash-component-suites/.
    (out / ".nojekyll").write_text("")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    largest = max((f for f in out.rglob("*") if f.is_file()), key=lambda f: f.stat().st_size)
    print(f"\nsite -> {out}")
    print(f"  {n_files} files, {total / 1e6:.2f} MB  (bundle: {n} artifacts, {n_chunks} lazy chunks)")
    print(f"  largest: {largest.relative_to(out)} ({largest.stat().st_size / 1e6:.2f} MB)")
    if total > 1e9:
        print("  WARNING: over GitHub Pages' 1 GB published-site limit")
    if largest.stat().st_size > 100e6:
        print("  WARNING: a file exceeds GitHub's 100 MB limit")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--skip-bundle", action="store_true", help="do not run build_bundle first")
    p.add_argument("--audit-only", action="store_true", help="report server-side callbacks and exit")
    a = p.parse_args()
    export(a.out, port=a.port, skip_bundle=a.skip_bundle, audit_only=a.audit_only)
