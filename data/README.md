# Documentation for `data/`

All raw data acquisition and access for the SUSTAG project — one subfolder per data source, each with a `make_*.py` script that downloads/builds it and an `access.py` module that exposes a clean read-only API.

Run through the `examples.ipynb` to see examples.

---

## Quickstart

Copy `api-keys-example.toml` to `api-keys.toml` and fill in your keys, then run the full pipeline:

```bash
cd data/
python initialize_data.py
```

This runs `water/make_water.py`, `map_overlays/make_overlays.py` and `weather/make_weather.py` in that order. If pulled from git, the source files should be present and `initialize_data.py` should finish quickly. If not, then prepare to wait a while. The scripts should intelligently save work, so if the install is interrupted, simply run `python initialize_data.py` again.

Weather data takes the longest to download. To build it separately:
```bash
python weather/make_weather.py
```

To target specific sites only:

```python
from data.weather import make_rain
make_rain.main(site_uids=["USGS-05482500", "USGS-05412500"])
```

### Key access patterns

```python
from data import water, map_overlays, weather

# get all valid site_uid codes as a list
uids = water.get_all_water_sites()

# nitrate time series for one site, aggregated to a 1W average
water.get_site_data("USGS-05412500")
water.aggregate_by_interval("USGS-05412500", interval="1W")

# rainfall for one site, aggregated to 3-day periods per grid cell
weather.get_site_rain("USGS-05412500")
weather.aggregate_by_interval("USGS-05412500", interval="3D")

# NHD hydrography
map_overlays.get_flowlines()
map_overlays.get_waterbodies()
```

---

## water/

Nitrate concentration time series and site metadata from IWQIS and USGS-NWIS monitoring stations across Iowa.

```python
from data import water
```

| Function | Returns |
|---|---|
| `get_all_water_sites()` | List of all site UIDs |
| `get_all_iwqis_sites()` | List of IWQIS site UIDs (`WQS*`) |
| `get_all_usgs_sites()` | List of USGS site UIDs (`USGS-*`) |
| `get_site_metadata()` | DataFrame of location metadata for all sites |
| `get_site_data(uid)` | Full nitrate time series for one site |
| `get_full_data()` | Concatenated time series for all sites |
| `aggregate_by_interval(uid, interval, agg_func)` | Resampled time series |
| `get_stats(uid)` | One-row DataFrame of site statistics |
| `get_full_stats()` | Statistics for all sites |
| `get_basin(uid)` | Upstream drainage basin polygon (GeoDataFrame) |
| `get_all_basins()` | All basin polygons in one GeoDataFrame |
| `get_all_basins_union()` | Dissolved union of all basins |
| `make_site_timeseries_plot(uid, interval, agg_func)` | Plotly figure |

```python
# list all sites
water.get_all_water_sites()
# ['WQS0003', 'WQS0054', ..., 'USGS-05412500', ...]

# location metadata (lat, lon, name, network)
water.get_site_metadata()

# raw time series for one site
df = water.get_site_data("USGS-05412500")

# resample to monthly mean
water.aggregate_by_interval("USGS-05412500", interval="1MS", agg_func="mean")

# resample from a pre-loaded DataFrame
water.aggregate_by_interval(df=df, interval="1W", agg_func="max")

# site statistics (sparsity, date range, lifespan)
water.get_stats("USGS-05412500")
water.get_full_stats()

# drainage basin polygon
water.get_basin("USGS-05412500")
water.get_all_basins()
water.get_all_basins_union()

# interactive Plotly timeseries figure
fig = water.make_site_timeseries_plot("USGS-05412500", interval="1W", agg_func="mean")
fig.show()
```

---

## weather/

Daily precipitation from IEM (Iowa Environmental Mesonet) grid polygons, filtered to each monitoring site's upstream drainage basin.

Output is one parquet per site at `weather/rain/<uid>_rain.parquet`, with one row per (grid cell, day). Grid cells are the ~4 km IEM precipitation polygons that intersect the basin.

```python
from data import weather
```

| Function | Returns |
|---|---|
| `get_site_rain(uid)` | Raw per-cell daily rain DataFrame |
| `aggregate_by_interval(uid, interval, agg_func)` | Per-cell rainfall aggregated to N-day periods |
| `plot_site_rain(uid)` | Three-panel matplotlib summary figure |

```python
# raw data: one row per (grid cell, day)
df = weather.get_site_rain("USGS-05412500")
# columns: date, lon, lat, precip_in_1d, year, month, day_of_year, week

# aggregate to 3-day totals, preserving spatial structure
weather.aggregate_by_interval("USGS-05412500", interval="3D", agg_func="sum")
# columns: date, lon, lat, precip_3d  — dates in 3-day increments

# aggregate from a pre-loaded DataFrame
weather.aggregate_by_interval(df=df, interval="1W", agg_func="sum")

# summary figure (time series, grid cell map, monthly pattern)
weather.plot_site_rain("USGS-05412500")
```

---

## map_overlays/

Iowa NHD (National Hydrography Dataset) flowlines and waterbodies, simplified and stored as GeoParquet. Used by the widget's hydrography layer.

```python
from data import map_overlays
```

| Function | Returns |
|---|---|
| `get_flowlines()` | NHD flowlines, stream order ≥ 3 (GeoDataFrame, EPSG:4326) |
| `get_waterbodies()` | NHD waterbodies (GeoDataFrame, EPSG:4326) |

```python
rivers = map_overlays.get_flowlines()
lakes  = map_overlays.get_waterbodies()
```

Both return cached GeoDataFrames — the parquet is only read once per process.

---

## SDA (Soil Data Access)

Queries USDA Web Soil Survey via the Soil Data Access REST API. Returns soil composition, horizon data, mapunit polygons, and crop yield summaries for arbitrary lat/lon points.

This module is only partially implemented and not yet integrated into the main pipeline.
