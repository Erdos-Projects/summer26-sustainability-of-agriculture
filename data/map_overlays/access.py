"""Read-only access layer for map overlay data in data/map_overlays/.

Overlays are GeoDataFrames (EPSG:4326) built from the USGS National
Hydrography Dataset and stored as GeoParquet. Re-run make_iowa_water_overlays.py
to refresh them.
"""

import geopandas as gpd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_OVERLAYS_DIR = _THIS_DIR / "overlays_data"

_FLOWLINES = None
_WATERBODIES = None


def get_flowlines() -> gpd.GeoDataFrame:
    """Return Iowa NHD flowlines (stream order >= 3)."""
    global _FLOWLINES
    if _FLOWLINES is None:
        _FLOWLINES = gpd.read_parquet(_OVERLAYS_DIR / "iowa_flowlines.parquet")
    return _FLOWLINES


def get_waterbodies() -> gpd.GeoDataFrame:
    """Return Iowa NHD waterbodies."""
    global _WATERBODIES
    if _WATERBODIES is None:
        _WATERBODIES = gpd.read_parquet(_OVERLAYS_DIR / "iowa_waterbodies.parquet")
    return _WATERBODIES
