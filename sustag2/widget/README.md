# Iowa Nitrate Forecast Widget

Dash app for exploring Iowa water-quality monitoring sites and forecasting nitrate at
**arbitrary ungauged points**: drop a pin, and the app auto-delineates the drainage basin (USGS
NLDI), builds the model feature set for that basin, and returns a daily predicted-nitrate curve
and violation-probability curve — with a β slider to trade recall against false alarms.

## Setup

The app reads through the `src/data` access layer, so it needs the project data and environment.
Follow the **root [`../README.md`](../README.md)** "Quickstart" once (create the conda env, download
the data bundle into `src/data/`). No separate widget data step is required.

Environment options (any one):

```bash
conda env create -f ../environment.yml && conda activate sustag   # project env (recommended)
# or, widget-only:
conda env create -f environment.yml        # widget/environment.yml
pip install -r requirements.txt            # widget/requirements.txt
```

## Running

From the repo root:

```bash
python widget/app.py        # Dash dev server -> http://127.0.0.1:8050
```

## Structure

```
app.py               # Dash entry point; builds layout + registers callbacks
layout.py            # top-level page layout and shared dcc.Store definitions
colors.py            # centralized layer palette + Explore-tab control defaults (+ DEBUG_MODE_ON)
geo_utils.py         # region geometry helpers
model_interface.py   # seam to the deploy virtual-site forecast (build_virtual_basin -> predict)
components/
    map_panel.py      # Leaflet map, site markers, selection tools, layer callbacks
    map_layout.py     # map + tools-panel layout (Explore / Forecast / Debug sections)
    map_common.py     # shared map constants, marker builders, assets
    info_panel.py     # nitrate timeseries for selected sites
    forecast_panel.py # drop-a-pin forecast figure + PNG download
    basin_editor.py   # Debug: basin-review tools
assets/
    custom.css
    iowa_flowlines.geojson    # generated at first startup
    iowa_waterbodies.geojson  # generated at first startup
```
