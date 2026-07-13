# sustag — Iowa waterborne-nitrate prediction

Project completed in the Erdos Summer 2026 Data Science Bootcamp. Predict waterborne nitrate concentration (regression) and 10 mg/L violation risk
(classification) at Iowa monitoring sites — and at arbitrary *virtual* sites — from continuous weather and land-use data, using XGBoost over a leakage-aware (LOFO) CV design.

> **This `sustag2/` tree is the active, working project** — the re-grain + `src/` refactor.
> It currently lives nested under the original `sustag/` repo and will move up to become the
> repo root once the legacy `sustag/` tree is retired.

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

## Running the app & notebooks

**Interactive widget** — drop a pin and get a predicted nitrate + violation-risk forecast at that ungauged point, with a β slider for the recall / false-alarm tradeoff:

```bash
python widget/app.py        # Dash dev server -> http://127.0.0.1:8050
```

**Demo notebooks** — `notebooks/fulldemo.ipynb` is intended as a walkthrough of the pipeline (data access -> EDA -> feature engineering -> cross-site CV -> final models -> results & deployment). It is the only demo written retroactively.

It imports helpers from the sibling `demo_*.py` modules (`demo_eda`, `demo_model`, `demo_baselines`, `demo_recipes`).

The following additional notebooks (all in `notebooks/`) were written as demos for other team members at various points in the project, and may be of use to someone going through the repo.

- `clean-IWQIS-site-data.ipynb`: shows why and which of the 162 original sites were thrown away to arrive at the current 85 site list
- `example_recipes.ipynb`: a file intended for showing how to use `src.features.features` to build recipes
- `example_split.ipynb`: a file intended to show how to use `src.splits.conflict_graph` for generating CV splits
- `examples_data_access.ipynb`: a file written for showcasing the original version of the data module pre-Erdos spec, 80% it has been fixed to work post-refactor

## Repo Structure
- `experiments/`: contains one directory for each team member, mostly a grave yard. CODE HERE NOT GUARANTEED TO RUN. It was not revised after the Erdos-spec refactor.
- `logs/`: an underutilized log directory. Only `src.models.train` writes to it.
- `notebooks/`: contains example notebooks and the `fulldemo.ipynb` notebook
- `presentation/`: contains a copy of the presentation slides
- `src/`: all reusable code lives here. Has submodules
  - `src/build/`: for building data (main file `make_data.py`)
  - `src/data/`: data access module, also where data is stored
  - `src/eval/`: one file, `cook.py`, used for training and CV
  - `src/features/`: used for building feature lists, used extensively during feature engineering phase of project
  - `src/models/`: used for tuning and training models. Models stored to `src/models/models/` and manually moved to `{proj-root}/deploy/models` for deployment.
  - `src/splits/`: contains `conflict_graph.py`, used for CV splits.
  - `selftest.py`: a data consistency check, indended to be run from command line
- `widget`: the browser-based widget used throughout this project for various things. Almost entirely vibe-coded, only exception being some parts of the Basin Editor tool.

**Modifications from Erdos Spec**
- `artifacts/` has been deleted, models live in `deploy/models/`
- `tests/` has been deleted, `src/selftest.py` tests the data pipeline post construction, `make_data` runs its own tests as well.

## What's committed vs. downloaded

A fresh clone already contains most of the data: per-site **water/nitrate**, **basins**, the tiny cell-aggregated **crop / surplus / grid globals**, and the **surplus display assets**. The only large artifact not in git is **`weather_global` (~4.6 GB)** plus a few small `processed/` dirs — that's what the data download provides (extract it into `src/data/`). So everything runs against committed data except the weather layer until the bundle is in place.

The trained **models** live in **`deploy/models/`** — the deployed boosters `isaac_REG2` / `isaac_CLF2` and their `.meta.json` sidecars, committed so the widget and deploy path work on a fresh clone. (This is the replacement for the former top-level `artifacts/` directory; new training runs also write a copy under `src/models/models/`.)

A *full* rebuild from raw (`make_data.py`) additionally needs `src/build/api-keys.toml` (a USGS token) and the CDL / gTREND sources — see [`src/README.md`](src/README.md)'s "Catastrophic rebuild" for per-source steps.

## Documentation

- [`src/README.md`](src/README.md) — per-submodule (`data` / `eval` / `features` / `models` / `splits`) API, with runnable examples.
- [`data_inventory.md`](data_inventory.md) — every external data source (URL, access method, API-key needs).
- [`kpis.md`](kpis.md) — metric definitions for the CV scores.

## Status

The re-grain + `src/` refactor is functional end to end — data access, feature recipes, cross-site
CV, model training, the deploy inference path, and the widget all run on the new `global_node_id`
grain, and the raw-acquisition utilities (`clip_crops`, `build_source`, `gen_surplus_statistics`)
are ported so the dataset can be rebuilt from source. Remaining: promote `sustag2/` up to the repo
root once the legacy `sustag/` tree is retired.
