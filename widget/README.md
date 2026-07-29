# Iowa Nitrate Forecast Widget

Dash app for exploring Iowa water-quality monitoring sites and forecasting nitrate at **arbitrary ungauged points**: drop a pin, and it snaps to the nearest NHD reach, scores the light REG and CLF boosters for that reach's basin, and draws a daily predicted-nitrate curve and a violation-probability curve — with a β slider trading recall against false alarms.

**It runs with no server.** Every callback is clientside and every panel reads a precomputed bundle in `assets/data/`, so the same code path serves the local dev app and the published static site. The pin-to-forecast chain — snap, feature assembly, XGBoost tree walk — happens in the browser (`assets/clientside/forecast.js`); Python's copy of it (`model_interface.py`) survives only as the reference the parity harness checks against.

## Static build instructions

The published site is `static-site/` (gitignored), served from an orphan `gh-pages` branch. Building it is four stages, each resumable and each cached separately, in dependency order:

```bash
# 1. NLDI basin polygons, one JSON per reach. ONE-TIME, and it takes the best part of a day:
#    NLDI meters by QUOTA (800 per ~12 min window), so the run alternates bursts and waits.
python -m widget.static.fetch_basins --status      # progress, no requests
python -m widget.static.fetch_basins               # everything still missing

# 2. Per-reach cells and feature rows. Hours. Ctrl-C is safe; a restart skips what exists.
#    Pin the BLAS threads: unpinned, N workers x 8 threads oversubscribe the machine (~3x slower).
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m widget.static.build_reaches --workers 3

# 3. Pack everything the browser reads into assets/data/ + manifest.json.
python -m widget.static.build_bundle               # all stale groups
python -m widget.static.build_bundle --list        # what is in the bundle, by group
python -m widget.static.build_bundle --only models --force

# 4. Snapshot the app, graft the bundle in, publish.
python -m widget.static.check_forecast             # browser vs Python parity, a few reaches
python -m widget.static.export                     # -> static-site/
python -m widget.static.publish                    # stage + describe, pushes NOTHING
python -m widget.static.publish --push             # force-push -> origin/gh-pages
```

**Stage 2 keeps two caches, split at the cost.** `src/data/cache/reach_grids/` holds each basin's clipped cells — a D8 flow-distance field plus a cell intersection, which is the hours in this pass, and which no recipe setting touches. `src/data/cache/reach_rows/` holds the aggregation over those cells, keyed by `ROW_SCHEMA` and a recipe fingerprint. So retuning a bucket edge, decay length or weather geometry re-runs only the aggregation (~140 ms a reach), not the delineation.

**The models come from `deploy/models/`**, not from a training run directly: `build_bundle --only models` repacks whatever light pair is deployed there into flat arrays the browser can walk. Retraining means copying the new booster and its `.meta.json` into `deploy/models/` and re-running that group. The classifier's meta needs a `beta_table` (from `src.models.tune_threshold`) or the β slider has no operating point to show.

**`export` refuses to write a site that would be quietly broken.** It audits every callback and fails on any server-side one outside the two deferred Debug overlays — a server callback on a static host renders a live-looking control that silently does nothing. It then runs the feature-skew check (`_feature_skew.cjs`), which fails if the browser cannot resolve a model feature from the bundle: an unresolvable column arrives as NaN, indistinguishable from a legitimately absent distance ring, so the site would score a plausible forecast without it.

## Running locally

```bash
python widget/app.py                  # -> http://127.0.0.1:8050
python widget/app.py --refresh        # rebuild stale bundle groups first
python widget/app.py --refresh models covariates --force
```

The app reads `assets/data/`, so it shows the bundle rather than live `src/data/` — what you see locally is what the published site does. `--refresh` is how an edit to `src/data/` reaches the app.

## Setup

**Running** the app needs only the bundle and Dash. **Building** it needs the full project environment and the ~5 GB of local data (weather parquets, the D8 raster, grid_global), because that is what the builders read:

```bash
conda env create -f ../environment.yml && conda activate sustag
```

then download the data bundle into `src/data/interim/` as the root [`../README.md`](../README.md) describes.

## Structure

```
app.py                 # Dash entry point; layout + callback registration
layout.py              # page layout and shared dcc.Store definitions
bundle.py              # READ side of assets/data/ -- coverage + relative asset URLs
colors.py              # layer palette and Explore-tab control defaults
geo_utils.py           # region geometry helpers
model_interface.py     # the Python forecast: parity reference, not called by the app
                       #   forecast_virtual_site(lat, lon, ...) delineates then scores
                       #   forecast_site_data(sd, ...)          scores a basin you already have
components/
    map_panel.py       # Leaflet map, markers, selection tools, layer callbacks
    map_layout.py      # map + tools panel (Explore / Forecast / Debug)
    map_common.py      # shared constants, marker builders, clientside_consts()
    info_panel.py      # nitrate timeseries for selected sites
    forecast_panel.py  # registers the two clientside forecast callbacks
    basin_editor.py    # Debug: basin review, read-only
assets/
    custom.css
    dashExtensions_default.js   # generated by dash-extensions (map JS helpers)
    iowa_{flowlines,waterbodies}.geojson
    clientside/        # bundle.js (asset reader) + ui / layers / panels / forecast
    data/              # the precomputed bundle + manifest.json (gitignored)
static/
    fetch_basins.py    # NLDI basin polygons (stage 1)
    build_reaches.py   # per-reach cells + feature rows (stage 2)
    build_bundle.py    # the bundle and its manifest (stage 3); build_forecast.py holds the forecast groups
    export.py          # dash2html snapshot + bundle graft + the audits
    publish.py         # orphan-commit force-push to gh-pages
    check_forecast.py  # browser vs Python parity on one basin (+ _forecast_parity.cjs)
    _feature_skew.cjs  # the JS half of deploy.predict._assert_no_skew
```

## Notes

- **Where a pin lands is a model INPUT, not a label.** `dist_to_sensor` is the D8 flow distance from each cell to the pin, and the light recipes cut rings at 5/50 km (REG) and 2/5 km (CLF), so moving the pin along a reach pushes cells across ring boundaries and changes `pct_*_b<n>`, the surplus rings, the per-ring weather and the travel-time lags. Measured: scoring a reach from its midpoint rather than its outlet moves the prediction 0.15-0.36 mg/L. Every forecast is therefore computed at the reach OUTLET, which is also where its row was built — and why the map moves the marker there rather than leaving it under the cursor.
- **`check_forecast` measures implementation, not geometry.** Both sides get the same cached NLDI basin at the same outlet and neither snaps, so what it reports is the browser-vs-Python difference alone: currently ~0.012 mg/L mean against the model's own 4.4 mg/L LOFO RMSE. It does NOT cover snapping — that is `_make_basins.snap_comid` vs `forecast.js::snapComid`, and letting it run inside this harness only ever meant each side scoring a different catchment.
- **Asset URLs must stay relative** (`assets/data/...`, never `/assets/...`). The site is a GitHub *project* page under `user.github.io/<repo>/`, and a leading slash works locally and 404s only once deployed.
- **The snap set and the reach store describe the same reaches by construction.** Reaches NLDI cannot answer for are tombstoned in `failed.json` and dropped from both `fetch_basins.reach_ids` and the snap index, so a pin can never land on a COMID with no feature row. Rebuild `snap_index` if new tombstones appear.
- **The Debug tab is read-only.** Basin review shows the comparison; the controls that wrote `preferred_basin.csv` are gone, since a static site cannot write. Curating a basin selection is a local task against the Python app.
- **Two callbacks remain server-side on purpose** — the Debug pin-delineation overlays (`pin-basin-layer`, `pin-basin-v3-layer`), which call NLDI and flood-fill a D8 raster live. They are inert on the published site and listed in `export.DEFERRED_OUTPUTS`.
- **The presentation-only "bad sites" markers are excluded from the static build** — decorative, with no data behind them.
