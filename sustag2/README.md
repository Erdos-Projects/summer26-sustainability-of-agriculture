# sustag — Iowa waterborne-nitrate prediction

Predict waterborne nitrate concentration (regression) and 10 mg/L violation risk
(classification) at Iowa monitoring sites — and at arbitrary *virtual* sites — from
continuous weather and land-use data, using XGBoost over a leakage-aware (LOFO) CV design.

> **⚠ This `sustag2/` tree is a staging area for an in-progress restructure.**
> All new files/dirs from the re-grain + `src/` refactor land here. Once `sustag2/` is
> verified to reproduce the current `sustag/` project, everything else in `sustag/` is
> deleted and `sustag2/`'s contents move up one level to become the new repo root.
> Until then, `sustag/` remains the source of truth.

## Quickstart

1. Clone the repo and obtain the dependencies.

```bash
git clone https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture.git
cd sustag2
conda env create -f environment.yml
conda activate sustag
```

2. Download raw data from https://utexas.box.com/s/h4bjxgsuydcl7ya6gpwyiqepl477cdo7. Place the contents of the download in `src/data/raw/`.

3. Navigate to `src/build/` and run the command `python make_data.py`. Wait for it to finish.

You should now be setup!

## Architecture in one breath

Two grains, one boundary:

- **Global grain** — crops/surplus/weather are aggregated **once** over all ~23k Iowa IEM
  cells (keyed by `global_node_id`), on **one canonical global Voronoi** (`grid_global`).
- **Site grain** — a "site" is a **live view**: a thin membership table
  (`node_id, global_node_id, dist_to_sensor, frac_cell_in_basin`) joined to the global tables.

And the split that keeps the runtime read path lean:

- **`src/build/`** — build-time ETL/geo (rasterio, geopandas overlay, Voronoi, API clients).
  Runs occasionally; produces artifacts into `src/data/{raw,interim,processed}`.
- **`src/data/`** — the runtime read path (parquet reads + joins). Imports **none** of the
  build stack. Sorting rule: *if the read path imports a module, it lives on the data side;
  if the read path only reads a file that module produced, that module lives in `build/`.*

## Directory structure

```
sustag2/                        (staging; becomes repo root after verification)
├── README.md                   this file
├── data_inventory.md           the 8 sources (access / license / limits)
├── kpis.md                     metric definitions (mirrors src/eval/metrics.py)
├── schema.json                 SiteData + feature/target schema
├── environment.yml             reproducible env (ported from widget/)
├── Makefile                    raw → grid_global → features → models → figures
│
├── src/
│   ├── data/                   ← runtime READ path (lean; geo deps only inside site_view)
│   │   ├── raw/                immutable source snapshots        (gitignored)
│   │   ├── interim/            grid_global, crops_global, surplus_global, weather_global
│   │   ├── processed/          thin per-site views {node_id, global_node_id, dist, frac}
│   │   ├── cache/              memoized cross-site features (state-daily, climatologies)
│   │   ├── access.py           SINGLE read surface + SiteData + live joins; re-exports build_site_view
│   │   ├── site_view.py        (basin, sensor) → membership; merges build_grid + build_grid_from_basin
│   │   ├── crs.py              EQUAL_AREA_CRS = "EPSG:5070" + wgs84_to_albers
│   │   └── cdl_legend.py       CDL code→class lookup (imported by access AND build)
│   │
│   ├── build/                  ← build-time ETL/geo (heavy deps quarantined)
│   │   ├── make_data.py        orchestrator
│   │   ├── _make_water.py  _make_basins.py  _make_grid.py  _make_weather.py
│   │   ├── _make_crops.py  _make_surplus.py  _make_map_overlays.py  _make_aux.py
│   │   ├── config.py           parses pipeline_config.toml (get_config, get_region_bbox)
│   │   ├── pipeline_config.toml  build knobs (region, years, site filters, thresholds, agg_crops)
│   │   └── util/               gen_surplus_statistics.py, clip_crops.py, build_source.py
│   │
│   ├── features/               transformers.py · recipes.py · preprocessing.py
│   ├── splits/                 conflict_graph.py  (LOSO / LOFO-family splitter)
│   ├── eval/                   metrics.py · stress_tests.py
│   └── models/                 train.py · wrappers.py · tune.py
│
├── deploy/                     virtual-site inference (APP; imports src/data, not src/build)
├── widget/                     Dash app (APP; imports src/data)
│
├── notebooks/                  eda · feature_selection · pipeline_demo · modeling_baselines ·
│                               modeling_experiments · metric_evaluation · final_results
├── results/                    eda/ interpretability/ stress_tests/ final/ + model_comparison.csv
├── artifacts/                  final_model_{reg,clf}.json + .meta.json
├── tests/                      test_pipeline · test_models · test_splits · test_parity
├── presentation/               slides + summary.md
├── logs/                       (optional) provenance
└── experiments/                per-contributor scratch — promote keepers into src/
```

## Migration status

Skeleton scaffolded; modules are stubs pending port from `sustag/`. Guardrail before deleting
anything in `sustag/`: assert the new `access` getters reproduce the current per-site outputs
for all 85 sites (exact for crops/membership; float-tolerant for area-weighted surplus).
