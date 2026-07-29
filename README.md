# sustag — Virtual Waterborne Nitrate Sensors in Iowa

Iowa grows a lot of corn and puts a lot of fertilizer in the ground. Nitrogen from that fertilizer becomes nitrate in Iowa's water and gives people cancer, which is bad. [Real time nitrate sensors exist](https://iwqis.iowawis.org/) to monitor the nitrate in the water supply, but they break frequently and are currently being hit by funding cuts. 

**Question: Can dangerous nitrate levels be predicted from weather and land-use data at sites that have never been seen?** Said another way, using data and science, can we deploy *virtual sensors* to make safety predictions in areas without any physical detectors in the water?

TLDR; Yes, you can.

- [Link to our beautiful presentation video](https://www.youtube.com/watch?v=O_ZCylQCXe8) (5 minutes long, results slightly outdated)
- [Link to an interactive demo of our widget/model]() (please play around with it!)

We build two separate XGBoost models, a classifier which identifies nitrate violations and a regressor which models maximum daily nitrate values. We provide a Dash widget to interact with our data and run forecasts, inspired by the [IWQIS water quality app](https://iwqis.iowawis.org/app/?iwqis=/sensors-map) (the demo site linked above is a light version of this widget running a stripped-down pair of our models).
  - [Results](#results)
  - [Quickstart](#quickstart)
  - [Running the app \& notebooks](#running-the-app--notebooks)
  - [Repo Structure](#repo-structure)
  - [What's committed vs. downloaded](#whats-committed-vs-downloaded)
  - [Documentation](#documentation)

## Results
**Goal:** drop a pin in an arbitrary location in Iowa's waterways and on any given day, make two predictions:
- (A): will the nitrate at this location exceed 10 mg/L (yes/no)?
- (B): what is maximum nitrate value this location will observe (in mg/L)?

**Training:** Our two XGBoost models are trained on data from 81 live nitrate sensors (158,215 sensor-days) along with geographic crop, nitrogen-surplus and weather data clipped to each sensor's drainage basin. We validate our models against sites with no hydrological connection to the training sites, see [Roberts et al. 2017, *Ecography* 40:913–929](https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/ecog.02881) as a precedent for example. This means our results are conservative; these models likely perform better than reported.

**Classifier Results:** On basins withheld from training, the classifier reaches 0.86 ROC AUC and 0.69 average precision against a 26% violation base rate — a 2.7× improvement over chance ranking. We ship it with a table of $F_\beta$-optimal operating points: the row at $\beta = 3.5$ shows that we catch 97% of true violation days if we're willing to tollerate 65% of our positive predictions being false alarms.

| $\beta$ | recall | fdr | precision | accuracy |
| --- | --- | --- | --- | --- | 
| 1.0 | 0.7579 | 0.4339 | 0.5661 | 0.7877 |
| 2.0 | 0.8804 | 0.5332 | 0.4668 | 0.7098 |
| 3.0 | 0.9565 | 0.6300 | 0.3700 | 0.5687 |
| **3.5** | **0.9732** | **0.6528** | 0.3472 | 0.5212 |
| 4.0 | 0.9773 | 0.6607 | 0.3393 | 0.5033 |

**Regressor Results:** On basins withheld from training, the regressor explains 40% of daily variance (R² = 0.401) with a typical error of 4.4 mg/L. That splits into two very different abilities: it tracks a basin's variation over time reasonably well (within-site R² = 0.41) but ranks one basin's overall level against another's only weakly (between-site R² = 0.20 ± 0.08) — the harder half, and the one that matters most for ungauged sites.

**Improvements:** Currently our virtual sensors are modeled according to the most drastic possible deployment scenario: we presume they have no access to no *physical* water sensors, and therefore they make predictions using only daily weather data, annual land-use data, and static geographic data. If we were to instead treat these virtual sensors as a supplement to an existing network of live sensors, as exists in Iowa and indeed the rest of the Continental United States, then we would gain access to a slew of additional features which would vastly improve our predictions. This would unlock not only live nitrate values elsewhere in the hydrological network but also other useful covariates such as water turbitidy, discharge rate, and chlorophyll fluorescence which are known to affect water-borne nitrate. Basic statistics tests demonstrate that including network-enabled features in our models would *drastically* improve our 

## Quickstart

1. Clone the repo and obtain the dependencies.

```bash
git clone https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture.git
cd summer26-sustainability-of-agriculture
conda env create -f environment.yml
conda activate sustag
```

2. Download the weather layer from https://utexas.box.com/s/xvoktu11q2s06c0rlpz39wz7itbn5sur and extract the `weather_global_*.parquet` files into **`src/data/interim/`** (~4.6 GB). This is the only large data not committed to git — everything else the models, notebooks, and widget read is already in the clone. It is technically possible to build without this step, but it will take a long time.

3. (optional) Navigate to `src/build/` and run

```
python make_data.py
```

to ensure everything is there. (NOTE: Running this will download weather data from the USGS, and if it exceeds a certain amount, you will need to obtain an API key to proceed. See the instructions in `src/build/api-key-example.toml` if this occurs.)

You should now be set up! Run the app or notebooks below.

> **Advanced — full rebuild from raw.** To regenerate all data from the original sources instead of using the committed + downloaded artifacts: download the raw sources into `src/data/raw/`, add `src/build/api-keys.toml` (a USGS token), then run `python -m src.build.make_data`. This is slow and network-heavy (re-fetches gridMET/IEM/CDL/etc.) — see [`src/README.md`](src/README.md)'s "Catastrophic rebuild". Most users never need this. 
> 
> **Note:** `weather_global` is a *built* table (gridMET + IEM interpolated onto the IEM grid), so it lives in `interim/`, not `raw/` — dropping it in `raw/weather/` will *not* be picked up, and the build will re-download.

## Running the app & notebooks

**Interactive widget** — drop a pin and get a predicted nitrate + violation-risk forecast at that ungauged point, with a β slider for the recall / false-alarm tradeoff:

```bash
python widget/app.py        # Dash dev server -> http://127.0.0.1:8050
```

**Demo notebooks** — `notebooks/fulldemo.ipynb` is intended as a walkthrough of the pipeline (data access -> EDA -> feature engineering -> cross-site CV -> final models -> results & deployment). It is the only demo written retroactively.

It imports helpers from the sibling `demo_*.py` modules (`demo_eda`, `demo_model`, `demo_baselines`, `demo_recipes`).

The following additional notebooks (all in `notebooks/`) were written as demos for other team members at various points in the project, and may be of use to someone going through the repo.

- `clean-IWQIS-site-data.ipynb`: shows why and which of the 162 original sites were thrown away to arrive at the current 83 site list (the notebook itself stops at 85; two groundwater sites were dropped afterwards — see `[site_filters].groundwater` in `src/build/pipeline_config.toml`)
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

The **trained models** live in **`deploy/models/`** — the deployed boosters `isaac_REG2` / `isaac_CLF2` and their `.meta.json` sidecars, committed so the widget and deploy path work on a fresh clone. (This is the replacement for the former top-level `artifacts/` directory; new training runs also write a copy under `src/models/models/`.)

A *full* rebuild from raw (`make_data.py`) additionally needs `src/build/api-keys.toml` (a USGS token) and the CDL / gTREND sources — see [`src/README.md`](src/README.md)'s "Catastrophic rebuild" for per-source steps.

## Documentation

- [`src/README.md`](src/README.md) — per-submodule (`data` / `eval` / `features` / `models` / `splits`) API, with runnable examples.
- [`data_inventory.md`](data_inventory.md) — every external data source (URL, access method, API-key needs).
- [`kpis.md`](kpis.md) — metric definitions for the CV scores.