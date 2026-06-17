# Iowa Nitrate Forecast Widget

A Dash app for exploring soil/water data for a point or area in Iowa, and
(eventually) forecasting whether nitrate concentrations at downstream IWQIS
monitoring sites will exceed 10 mg/L within a month given an average
nitrogen surplus over a selected region.

## Running

```bash
python app.py
```

Requires the packages in `requirements.txt`. `data/iwqis_utils.py` reads
IWQIS data from `../../data/IWQIS/` (relative to this directory), so run
from `isaac/widget/`.

## Structure

```
app.py                      # Dash() instance + callback registration
layout.py                   # top-level page layout
components/
    map_panel.py             # the map + region-of-interest selection
    info_panel.py            # descriptive data about the selected region/site
    forecast_panel.py        # nitrogen surplus input -> model -> results
data/
    sda_utils.py             # USDA Soil Data Access (SDA) queries
    iwqis_utils.py           # IWQIS monitoring site data
    geo_utils.py             # region geometry helpers (normalization, downstream sites)
model_interface.py           # seam to the (not-yet-implemented) forecast model
```

### `app.py`

The entry point. Creates the `Dash` app, sets `app.layout` from
`layout.build_layout()`, and calls each component module's
`register_callbacks(app)`. `suppress_callback_exceptions=True` is required
because `map_panel` references `edit-control`, a component it creates
dynamically (only once area-selection mode is chosen) rather than at
startup.

### `layout.py`

The only place that decides where each panel's output appears on the page.
It also defines the two pieces of shared state every panel reads or writes:

- `dcc.Store(id="region-geom")` — the currently selected region, as a
  GeoJSON `Point` or `Polygon`. This is the single source of truth for
  "what is the user looking at."
- `dcc.Store(id="selected-site")` — the uid of an IWQIS site the user
  clicked on the map, if any.

Panels don't assume anything about their position on the page; `map_panel`
exposes named `dl.LayerGroup` "slots" (`mapunit-layer`, `forecast-layer`)
that `info_panel` and `forecast_panel` render into, regardless of where
`layout.py` places the map.

### `components/map_panel.py`

Owns the Leaflet map (`dl.Map`) and everything about selecting a region of
interest:

- **Point mode** (default): clicking the map sets `region-geom` to a
  `Point` and drops a marker.
- **Area mode**: a second toggle chooses **Rectangle** or **Polygon**. This
  mounts a `dl.EditControl` (Leaflet's draw toolbar) configured for the
  chosen shape; finishing a shape sets `region-geom` to its `Polygon`
  geometry.

Also handles:

- Tile layer switching (street/satellite).
- IWQIS site markers — toggled on/off, and clicking one sets
  `selected-site` (consumed by `info_panel` for the timeseries graph).

`region-geom` and `selected-site` are the *only* things this module
produces for other panels to consume — it has no knowledge of soil data,
timeseries, or the forecast model.

### `components/info_panel.py`

Purely descriptive — reacts to `region-geom` and `selected-site` but knows
nothing about the forecast model.

- On `region-geom` change: if it's a `Point`, queries `data.sda_utils` for
  soil map-unit data and renders it as tables, and draws the map unit
  outline into the `mapunit-layer` slot. If it's a `Polygon` (area
  selection), currently shows a placeholder — area-based soil queries
  aren't implemented yet.
- On `selected-site` change: fetches that IWQIS site's history via
  `data.iwqis_utils` and renders a nitrate-concentration timeseries graph.

`get_location_data()` and `get_site_timeseries()` are the two functions to
edit if you want to change what descriptive data is shown.

### `components/forecast_panel.py`

The nitrogen-surplus input, "Run forecast" button, and results display.
On click:

1. `data.geo_utils.normalize_for_forecast(region_geom)` — turns a `Point`
   into a small buffered `Polygon` (area selections pass through
   unchanged), so the model always receives an area.
2. `data.geo_utils.get_downstream_sites(...)` — looks up which IWQIS sites
   are downstream of that area (currently a stub returning `[]`).
3. `model_interface.forecast_exceedance(...)` — asks the model whether each
   downstream site will exceed 10 mg/L within a month. Currently raises
   `NotImplementedError`, which is caught and shown as "Forecast model is
   not yet implemented."

### `data/sda_utils.py`

Queries the USDA Soil Data Access (SDA) REST API for soil map unit
metadata, crop yield, horizon, and restriction data given a lat/lon point,
plus helpers to fetch map unit polygons as GeoJSON.

### `data/iwqis_utils.py`

Loads IWQIS monitoring site metadata and per-site water quality
timeseries from local CSVs, plus an `aggregate_by_interval` helper for
resampling timeseries data.

### `data/geo_utils.py`

Geometry helpers shared by the panels:

- `normalize_for_forecast(region_geojson)` — Point → buffered Polygon,
  Polygon → unchanged.
- `get_downstream_sites(region_geojson, sites_df)` — **stub**. Identifying
  downstream sites requires a watershed/flow-network dataset that isn't
  integrated yet; currently returns `[]`.

### `model_interface.py`

The seam between the Dash app and the forecast model. Defines
`forecast_exceedance(region_geojson, nitrogen_surplus, target_uids)`,
which should return a `dict[uid, ForecastResult]` where `ForecastResult`
has `exceeds_threshold`, `probability`, and `eta_days`. Not implemented yet
— this is the function to fill in once the model is ready. Nothing in the
Dash code needs to change when it is; only `forecast_panel.py` calls it.

## Known gaps / TODOs

- `data/geo_utils.get_downstream_sites` always returns `[]` (no watershed
  data yet).
- `model_interface.forecast_exceedance` is unimplemented.
- `info_panel` has no area-based soil query yet — area selections show a
  placeholder.
- `info_panel.get_location_data` passes the same `horizon` DataFrame for
  all three tables (Crop/Horizons/Restrictions) — pre-existing, not yet
  fixed.
