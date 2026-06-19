# Documentation for `data/`

Unified data creation/access for the sustainability of agriculture project. Top level folder organized as follows:

```
data/
  basins/
  crops/
  map_overlays/
  rain/
  surplus/
  water/
  access.py
  make_data.py
```

Each of the above directories is a submodule of data organized as follows:
```
<name>/
  access.py        read-only API — import this from other code
  make_<name>.py   downloads / builds the data
  <name>_data/     processed outputs (parquet, geojson, …)
  <name>_meta/     metadata files (CSVs, manifests)
  <name>_raw/      raw source files — gitignored
```

## Quickstart

You shouldn't need api-keys since the data should be accessible on Github, but if you do, copy `api-keys-example.toml` to `api-keys.toml` and fill in your keys. Then run the full pipeline to make sure everything is there:

```bash
cd data/
python make_data.py
```

Scripts run in dependency order: `water → map_overlays → basins → rain → surplus`. Each script skips work that is already up to date, so interrupted runs can be resumed by running `make_data.py` again. Pass `--force` to rebuild everything from scratch (but don't do this).

Once you've done that, run through the `examples.ipynb`. It shouldn't take long, maybe 5-10 minutes.

## Main Entrypoint — `data/access.py`

The top-level `access.py` provides a single entry point for loading all data for a site. The two methods `get_data` and `get_site_ids` are typically all you'd ever need. It also provides `aggregate_by_interval`, convenient for aggregating rain and water data simultaneously.

Designed to be imported from elsewhere in the project as follows:
```python
import sys
sys.path.insert(0, "path/to/project/root/")
from data import get_data, get_site_ids, aggregate_by_interval
```

See the `data/examples.ipynb` notebook for details.

--- EVERYTHING BELOW THIS POINT CLAUDE GENERATED DOCUMENTATION ---

--- MIGHT BE USEFUL BUT ALSO YOU CAN JUST READ THE VARIOUS `access.py` SCRIPTS ---
### `get_site_ids() → list[str]`

Returns all valid site UID codes. Uses `water/water_meta/site_location_metadata.csv` as the ground truth, so the list reflects the filtered set of sites (sites excluded by `water/config/` filters will not appear).

```python
uids = get_site_ids()
# ['WQS0003', 'WQS0054', ..., 'USGS-05412500', ...]
```

### `get_data(site_uid, include=None) → SiteData`

Loads all available data for a site into a `SiteData` dataclass. Missing data sources are silently skipped — check `.available()` to see what loaded.

```python
site = get_data("WQS0003")
site.water    # DataFrame or None
site.rain     # DataFrame or None
site.surplus  # DataFrame or None
site.basin    # GeoDataFrame or None
site.crops    # DataFrame or None

site.has("water")     # True / False
site.available()      # ['basin', 'rain', 'surplus', 'water']
```

Pass `include=` to load only specific fields and avoid unnecessary disk reads:

```python
site = get_data("WQS0003", include=["water", "rain"])
```

---

## Submodules

### `water/`

Nitrate concentration time series from IWQIS (`WQS*`) and USGS-NWIS (`USGS-*`) monitoring stations across Iowa. Source files are downloaded from the IWQIS and USGS REST APIs by the `make_water.py` script.

```python
from data import water
```

| Function | Returns |
|---|---|
| `get_site_ids()` | `list[str]` — all site UIDs |
| `get_metadata()` | `DataFrame` — location metadata for all sites |
| `get_water(uid)` | `DataFrame` — full nitrate time series for one site |
| `get_all_water()` | `DataFrame` — concatenated time series for all sites |
| `aggregate_by_interval(uid, interval, agg_func)` | `DataFrame` — resampled time series |
| `get_stats(uid)` | `DataFrame` — statistics for one site |
| `get_all_stats()` | `DataFrame` — statistics for all sites |
| `plot_water(uid, interval, agg_func)` | Plotly figure |

```python
# site metadata (lat, lon, name, network)
water.get_metadata()

# raw nitrate time series
df = water.get_water("USGS-05412500")
# columns: datetime, value, …

# resample to monthly mean
water.aggregate_by_interval("USGS-05412500", interval="1MS", agg_func="mean")

# resample from a pre-loaded DataFrame
water.aggregate_by_interval(df=df, interval="1W", agg_func="max")

# site statistics (date range, sparsity, …)
water.get_stats("USGS-05412500")
water.get_all_stats()

# interactive Plotly figure
fig = water.plot_water("USGS-05412500", interval="1W", agg_func="mean")
fig.show()
```

---

### `basins/`

Upstream drainage basin polygons for each monitoring site, in three variants: v1 (NLDI), v2 (authenticated NLDI), and v3 (D8 flow accumulation). A per-site "preferred basin" is tracked in `basin_meta/preferred_basin.csv` and is what `get_basin()` returns.

The basin editor in the widget UI can be used to manually review and reassign preferred basins.

```python
from data import basins
```

| Function | Returns |
|---|---|
| `get_basin(uid)` | `GeoDataFrame` — preferred basin polygon for one site |
| `get_all_basins()` | `GeoDataFrame` — all preferred basins |
| `get_all_basins_union()` | `GeoDataFrame` — dissolved union of all basins |
| `get_metadata()` | `DataFrame` — preferred basin selection metadata and review flags |
| `update_basin(uid, fields, basin_geom)` | Writes a new basin selection to disk |

```python
basin = basins.get_basin("WQS0003")   # GeoDataFrame, EPSG:4326
basins.get_all_basins()
basins.get_all_basins_union()

# review metadata: basin_type, reviewed, flags, …
basins.get_metadata()
```

**Basin priority when running the pipeline:**
The `rain` and `surplus` scripts both read `basin_meta/preferred_basin.csv` and detect stale sites via a `.basin_manifest.csv` file in their data directories. Changing a site's preferred basin will trigger a rebuild of that site's rain and surplus parquets on the next `make_data.py` run.

---

### `rain/`

Daily precipitation per site, derived from IEM (Iowa Environmental Mesonet) 4 km grid polygons spatially filtered to each site's preferred basin. Output is one parquet per site with one row per (grid cell, day).

Sites whose basin overlaps the IEM data footprint by less than 75% are skipped (their basins extend too far outside Iowa for the data to be meaningful).

```python
from data import rain
```

| Function | Returns |
|---|---|
| `get_rain(uid)` | `DataFrame` — daily per-cell precipitation |
| `aggregate_by_interval(uid, interval, agg_func)` | `DataFrame` — aggregated precipitation |
| `plot_rain(uid)` | Matplotlib summary figure |

```python
df = rain.get_rain("WQS0003")
# columns: date, lon, lat, precip_in_1d, year, month, day_of_year, week

# aggregate to 3-day totals (spatial structure preserved)
rain.aggregate_by_interval("WQS0003", interval="3D", agg_func="sum")

# aggregate from a pre-loaded DataFrame
rain.aggregate_by_interval(df=df, interval="1W", agg_func="sum")
```

---

### `surplus/`

Per-site nitrogen surplus time series and heatmap images derived from the Iowa nitrogen surplus grid dataset (ISU Extension). Pixels whose Albers (x, y) position falls inside a site's preferred basin are extracted and stored as one parquet per site.

```python
from data import surplus
```

| Function | Returns |
|---|---|
| `get_surplus(uid)` | `DataFrame` — annual nitrogen surplus per grid pixel |
| `get_surplus_image(year, site_uid)` | `(PIL.Image, bounds)` — heatmap image for one site/year |
| `get_surplus_image_buffer(uid, year)` | `(data_url, bounds)` — base64 PNG data URL (cached) |
| `get_iowa_surplus_image_buffer(year)` | `(data_url, bounds)` — Iowa-wide heatmap (cached) |
| `get_stats()` | `DataFrame` — global min/max surplus statistics |

```python
df = surplus.get_surplus("WQS0003")
# columns: pixel_id, year, surplus_kgha, total_kg_N, x, y, lon, lat

# heatmap rendering (used by the widget)
url, bounds = surplus.get_surplus_image_buffer("WQS0003", 2017)
```

The image is built in Albers (EPSG:5070) space so pixels align to the regular grid. Colors use the `YlOrRd` colormap normalised to the global min/max across all sites and years. Pixels with surplus below the global minimum (functionally, values of 0) are transparent.

**Stale detection:** `make_surplus.py` tracks which basin was used to generate each parquet in `surplus_data/.basin_manifest.csv`. Reassigning a preferred basin will trigger a rebuild on the next run.

---

### `map_overlays/`

Iowa NHD (National Hydrography Dataset) flowlines and waterbodies, simplified and stored as GeoParquet. Used by the widget's hydrography layer.

```python
from data import map_overlays
```

| Function | Returns |
|---|---|
| `get_flowlines()` | `GeoDataFrame` — NHD flowlines, stream order ≥ 3 (EPSG:4326) |
| `get_waterbodies()` | `GeoDataFrame` — NHD waterbodies (EPSG:4326) |

Both return cached GeoDataFrames — the parquet is read from disk only once per process.

---

### `crops/`

Placeholder — not yet implemented. The access layer and make script exist but return empty DataFrames.

```python
from data import crops
crops.get_crops("WQS0003")   # returns empty DataFrame
```
