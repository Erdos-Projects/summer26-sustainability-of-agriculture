"""Shared access to the pipeline configuration (pipeline_config.toml).

Import from anywhere in the project:

    from data.settings import get_config, get_region_bbox, get_equal_area_crs
"""

import tomllib
from functools import lru_cache
from pathlib import Path
from pyproj import Transformer

_CONFIG_FILE = Path(__file__).resolve().parent / "pipeline_config.toml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Return the full parsed pipeline config, cached for the process.

    Uses an absolute path off this file, so it is unaffected by os.chdir()
    (the pipeline runner changes the working directory per submodule).
    """
    with open(_CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def wgs84_to_albers(min_lon, min_lat, max_lon, max_lat):
    """
    Reproject a WGS84 bounding box to CONUS Albers (EPSG:5070).

    The CDL API requires coordinates in Albers projection — passing WGS84
    lat/lon directly will cause a 500 server error.

    Uses transform_bounds with densify_pts so the curved Albers edges are
    sampled and the returned box fully encloses the WGS84 region (transforming
    only the two corners undershoots the east/south edges).

    Returns:
        (min_x, min_y, max_x, max_y) in Albers metres
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:5070", always_xy=True)
    return transformer.transform_bounds(min_lon, min_lat, max_lon, max_lat, densify_pts=21)


def get_region_bbox(albers=False) -> tuple:
    """
    Shared area of interest.
    Default return is (min_lon, min_lat, max_lon, max_lat).
    If albers == True, returns in EPGS:5070.
    """
    if albers:
        return tuple(wgs84_to_albers(*tuple(get_config()["region"]["bbox_wgs84"])))

    return tuple(get_config()["region"]["bbox_wgs84"])


def get_equal_area_crs() -> str:
    """Project-wide equal-area CRS used for area/containment math (e.g. EPSG:5070)."""
    return get_config()["crs"]["equal_area"]
