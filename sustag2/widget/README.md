# Iowa Nitrate Forecast Widget

Dash app for exploring Iowa water-quality monitoring sites and (eventually)
forecasting nitrate exceedance at downstream IWQIS/USGS stations.

## Prerequisites

The app reads from `data/` at the repo root. Run the data pipeline first:

```bash
cd data
python initialize_data.py
```

## Running

```bash
cd widget
python app.py
```

Then open http://127.0.0.1:8050.

## Environment setup

**Conda (recommended)**

```bash
conda env create -f widget/environment.yml
conda activate sustag-widget
```

**pip**

```bash
pip install -r widget/requirements.txt
```

## Structure

```
app.py               # Dash entry point; registers all callbacks
layout.py            # top-level page layout and shared dcc.Store definitions
geo_utils.py         # region geometry helpers
model_interface.py   # seam to the (not yet implemented) forecast model
components/
    map_panel.py     # Leaflet map, site markers, selection tools, tools panel
    info_panel.py    # nitrate timeseries for selected sites
    forecast_panel.py
assets/
    custom.css
    iowa_flowlines.geojson    # generated at first startup
    iowa_waterbodies.geojson  # generated at first startup
```
