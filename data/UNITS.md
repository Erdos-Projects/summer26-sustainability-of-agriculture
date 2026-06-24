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

## `<site_uid>_rain.parquet`

| column name | units |
|-------------|-------|
| date | calendar date |
| node_id | — (basin-local rain-grid node id) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| lon | degrees (WGS84) |
| lat | degrees (WGS84) |
| precip_in_1d | inches (in), per day |
| year | — (calendar year) |
| month | — (1–12) |
| day_of_year | — (1–366) |
| week | — (ISO week, 1–53) |

## `<site_uid>_rain_grid.parquet`

| column name | units |
|-------------|-------|
| node_id | — (basin-local rain-grid node id) |
| global_node_id | — (canonical IEM cell id, shared across basins) |
| x | meters (m), EPSG:5070 easting |
| y | meters (m), EPSG:5070 northing |
| lat | degrees (WGS84) |
| lon | degrees (WGS84) |
| cell_area | m^2 |
| dist_to_sensor | meters (m) |
| frac_cell_in_basin | fraction (0–1, dimensionless) |
| geometry | polygon, EPSG:5070 (m) |

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
| node_id | — (basin-local rain-grid node id) |
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
| node_id | — (basin-local rain-grid node id) |
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
