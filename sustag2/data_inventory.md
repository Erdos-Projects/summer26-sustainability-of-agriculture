# Data inventory

Sources feeding the pipeline. Acquisition lives in `src/build/` (builders + `util/`), raw
snapshots in `src/data/raw/` (gitignored — see download instructions per source). TODO: fill
license/limits precisely; regenerate this table from the builders where possible.

| Source | What | Access | Builder | License / limits |
|---|---|---|---|---|
| IWQIS | Iowa nitrate sensor time series (`WQS*`) | REST API | `_make_water.py` | TODO |
| USGS-NWIS | USGS nitrate gauges (`USGS-*`) | REST API | `_make_water.py` | public domain |
| NLDI | Drainage-basin delineation (v1/v2) | REST API | `_make_basins.py`, `site_view.py` | public |
| IEM | ~4 km precip Voronoi grid (reference-day geometry) | zip download | `_make_grid.py` | TODO |
| gridMET | Daily weather (temp, ET, humidity, solar…) | download | `_make_weather.py` | TODO |
| USDA CDL | Cropland Data Layer (crop classification raster) | CropScape API | `util/clip_crops.py`, `_make_crops.py` | public |
| N-surplus | Iowa nitrogen-surplus grid (250 m) | static parquet | `util/build_source.py`, `_make_surplus.py` | TODO |
| NHD | Flowlines & waterbodies (widget overlays) | USGS NHD | `_make_map_overlays.py` | public |
