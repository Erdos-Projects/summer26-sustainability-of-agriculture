# Data column units

Units for each column in the major per-site data files. `<site_uid>` is the site
identifier (e.g. `USGS-05465500` or `WQS0033`). Columns that are identifiers or
calendar fields carry no physical unit and are marked `—`.

The `<site_uid>_water.parquet` schema is heterogeneous: USGS sites carry the
discharge/stage subset, IWQIS (`WQS`) sites carry the water-chemistry subset.
The table below lists every column that can appear, with its unit.

## `<site_uid>_basinx.parquet`

| column name | units |
|-------------|-------|
| site_id | — (site identifier) |
| comid | — (NHDPlus COMID identifier) |
| site_lat | degrees (WGS84) |
| site_lon | degrees (WGS84) |
| area_km2 | km^2 |
| geometry | polygon, EPSG:4326 (degrees) |

## `<site_uid>_water.parquet`

| column name | units |
|-------------|-------|
| datetime (index) | timestamp (UTC) |
| site_uid | — (site identifier) |
| temp_water | °C |
| dts_temp_water | °C |
| nitrate_con | mg/L (as N) |
| phosphate_con | mg/L (as PO4-P) |
| discharge | ft^3/s |
| stage | ft |
| spec_cond | µS/cm |
| spec_cond_v2 | µS/cm |
| ph | pH units (dimensionless) |
| ph_v2 | pH units (dimensionless) |
| diss_oxy_con | mg/L |
| diss_oxy_con_v2 | mg/L |
| diss_oxy_sat | % |
| diss_oxy_sat_v2 | % |
| chloro_con | µg/L |
| chloro_v | volts (V) |
| turbi_mean | NTU |
| turbi_mean_v2 | NTU |
| turbi_med | NTU |
| turbi_min | NTU |
| turbi_max | NTU |
| turbi_var | NTU |
| turbi_bes | NTU |
| r_exc | — (reference extinction) |
| m_exc | — (measurement extinction) |

## `<site_uid>_surplus_grid.parquet`

| column name | units |
|-------------|-------|
| node_id | — (basin-local grid node id) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| year | — (calendar year) |
| surplus_kgha | kg N/ha |
| total_kg_N | kg N |

## `<site_uid>_surplus_pixel.parquet`

| column name | units |
|-------------|-------|
| pixel_id | — (surplus raster pixel id) |
| year | — (calendar year) |
| surplus_kgha | kg N/ha |
| total_kg_N | kg N |
| x | meters (m), EPSG:5070 easting |
| y | meters (m), EPSG:5070 northing |
| lon | degrees (WGS84) |
| lat | degrees (WGS84) |

## `<site_uid>_crops_grid.parquet`

| column name | units |
|-------------|-------|
| node_id | — (basin-local grid node id) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| year | — (calendar year) |
| Alfalfa | pixel count (30 m × 30 m CDL pixels) |
| Corn | pixel count (30 m × 30 m CDL pixels) |
| Fallow | pixel count (30 m × 30 m CDL pixels) |
| Hay_Pasture | pixel count (30 m × 30 m CDL pixels) |
| Nonag | pixel count (30 m × 30 m CDL pixels) |
| Other | pixel count (30 m × 30 m CDL pixels) |
| Small_Grains | pixel count (30 m × 30 m CDL pixels) |
| Soybeans | pixel count (30 m × 30 m CDL pixels) |

# Weather pipeline (data/weather/)

Weather adds gridMET meteorology (~4 km daily, bilinearly sampled to each cell)
alongside the IEM precip. Source-of-truth is the global yearly files; per-site
files are slices of them. Temperatures are °C; all other gridMET variables are
in their native units.

`lon`/`lat` and the calendar fields (`year`, `month`, `week`, `day_of_year`) are
**not stored** in the weather files — they were redundant on every (cell, day)
row. Coordinates are constant per `global_node_id` and live in the grid files
(`<site_uid>_grid.parquet` / `global_grid.parquet`); calendar fields are
derivable from `date`. The global files cover only the cells inside the project
region bbox (`data.settings.get_region_bbox`), not the whole IEM footprint.

## `<site_uid>_weather.parquet`

One row per (cell, day) over the site's nitrate record padded ±60 days.

Large sites are stored split across `<site_uid>_weather_p1.parquet`,
`<site_uid>_weather_p2.parquet`, … (each kept under GitHub's 100 MB file limit);
`data.weather.get_weather` concatenates the ordered parts transparently, so the
schema below is what you get back either way.

| column name | units |
|-------------|-------|
| date | calendar date |
| node_id | — (basin-local cell id; joins surplus/crops) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| precip_in_1d | inches (in), per day — IEM |
| max_temp | °C (gridMET tmmx) |
| min_temp | °C (gridMET tmmn) |
| max_rel_humidity | % (gridMET rmax) |
| min_rel_humidity | % (gridMET rmin) |
| vpd | kPa (gridMET vpd, mean vapor-pressure deficit) |
| solar_rad | W/m² (gridMET srad, downwelling shortwave) |
| evapotranspiration | mm (gridMET pet, reference ET grass) |
| fuel_moisture_1000h | % (gridMET fm1000, 1000-hr dead fuel moisture) |

## `global_grid_weather_{year}.parquet`

Same columns and units as `<site_uid>_weather.parquet` **except `node_id`** (the
global files are basin-independent, keyed by `global_node_id`). One row per
(global cell, day) for every region cell in `global_grid.parquet` and every day
of the year.

## `<site_uid>_grid.parquet`  (weather_grid/)

The shared spatial aggregation grid (per-basin Voronoi cells over the IEM grid)
that surplus and crops aggregate onto; join to them on `node_id`.

| column name | units |
|-------------|-------|
| node_id | — (basin-local grid node id) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| x | meters (m), EPSG:5070 easting |
| y | meters (m), EPSG:5070 northing |
| lat | degrees (WGS84) |
| lon | degrees (WGS84) |
| cell_area | m^2 |
| dist_to_sensor | meters (m) |
| frac_cell_in_basin | fraction (0–1, dimensionless) |
| geometry | polygon, EPSG:5070 (m) |

## `global_grid.parquet`  (weather_grid/)

| column name | units |
|-------------|-------|
| global_node_id | — (canonical IEM cell id) |
| contained_in_sites | — (list of site_uids whose grid includes the cell) |
| n_sites | — (count) |
| lat | degrees (WGS84) |
| lon | degrees (WGS84) |
