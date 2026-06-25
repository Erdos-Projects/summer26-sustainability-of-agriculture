from dataclasses import dataclass
import pandas as pd
import numpy as np
import geopandas as gpd
from .basins import get_basin
from .crops import get_crops
from .surplus import get_surplus
from .water import get_water, get_site_ids as get_ids, aggregate_by_interval as agg_water, get_location
from .weather import (
    get_weather,
    get_grid as _get_grid,
    aggregate_by_interval as agg_weather,
    get_global_grid as _get_global_grid,
)

_ALL_FIELDS = ("basin", "crops", "grid", "surplus", "water", "weather")

# The precip subset of weather matching the retired <site_uid>_rain.parquet,
# exposed via SiteData.rain for back-compat.
_RAIN_COLUMNS = ("date", "node_id", "global_node_id", "precip_in_1d")


def get_grid(site_uid: str):
    """Top-level access to a site's grid (the shared spatial aggregation grid).

    Thin pass-through to data.weather.get_grid so surplus/crop builders depend on
    the public access layer rather than importing the weather module directly.
    """
    return _get_grid(site_uid)


@dataclass
class SiteData:
    site_uid: str
    basin: gpd.GeoDataFrame | None = None
    crops: pd.DataFrame | None = None
    grid: pd.DataFrame | None = None
    surplus: pd.DataFrame | None = None
    water: pd.DataFrame | None = None
    weather: pd.DataFrame | None = None

    basin_area: float | None = None
    sensor_location: tuple[float] | None = None

    def has(self, key: str) -> bool:
        """Return True if the named field was successfully loaded."""
        return getattr(self, key) is not None

    def available(self) -> list[str]:
        """Return names of fields that were successfully loaded."""
        return [f for f in _ALL_FIELDS if getattr(self, f) is not None]

    @property
    def rain(self) -> pd.DataFrame | None:
        """Back-compat view: the old rain-parquet columns sliced from `weather`.

        Returns the subset of the weather frame matching the retired
        <site_uid>_rain.parquet schema (date, node_id, global_node_id,
        precip_in_1d), or None if weather was not loaded.
        """
        if self.weather is None:
            return None
        return self.weather[list(_RAIN_COLUMNS)].copy()

    def agg(self, interval="1D", agg_func_water="mean", agg_func_weather="sum", inplace=False):
        new_water = agg_water(df=self.water, interval=interval, agg_func=agg_func_water)
        new_weather = agg_weather(df=self.weather, interval=interval, agg_func=agg_func_weather)

        new_data = SiteData(
            site_uid=self.site_uid,
            basin=self.basin,
            crops=self.crops,
            grid=self.grid,
            surplus=self.surplus,
            water=new_water,
            weather=new_weather,
            basin_area=self.basin_area,
            sensor_location=self.sensor_location,
        )

        if inplace:
            self = new_data

        return new_data


def get_data(site_uid: str, include: list[str] | None = None) -> SiteData:
    """Load all available data for a site. Pass include= to load a subset."""
    load = set(include) if include else set(_ALL_FIELDS)

    def _try(fn, *args):
        try:
            return fn(*args)
        except (FileNotFoundError, KeyError):
            return None

    return SiteData(
        site_uid=site_uid,
        basin=_try(get_basin, site_uid) if "basin" in load else None,
        crops=_try(get_crops, site_uid) if "crops" in load else None,
        grid=_try(_get_grid, site_uid) if "grid" in load else None,
        surplus=_try(get_surplus, site_uid) if "surplus" in load else None,
        water=_try(get_water, site_uid) if "water" in load else None,
        weather=_try(get_weather, site_uid) if "weather" in load else None,
        basin_area=_try(get_basin_area, site_uid) if "grid" in load else None,
        sensor_location=_try(get_location, site_uid) if "water" in load else None,
    )


def get_basin_area(site_uid):
    grid = _get_grid(site_uid)
    cell_areas = np.array(grid.cell_area.values)
    overlaps = np.array(grid.frac_cell_in_basin.values)
    return np.dot(cell_areas, overlaps)


def get_site_ids() -> list[str]:
    """Get a list of all site_uid codes of sites in the dataset.

    This uses water/water_meta/site_location_metadata.csv as the ground truth.
    It means that this is a minimal list of sites, as some of the sites are filtered out manually in the make_water script using filters defined in water/config.

    This is intentional, it means that if the filter list is updated data isn't regenerated. If you manually get site_uids from some other metadata file, don't be surprised if there are more sites than you expect.

    Returns
    -------
    _type_
        _description_
    """
    return get_ids()


def get_global_grid():
    """Load the global grid: the inverse map from each grid cell to the sites
    whose basin contains it.

    Every cell that falls in at least one site's (preferred-basin) grid appears
    once, keyed by global_node_id — the canonical IEM grid-cell index, which is
    shared across basins (so overlapping basins reference the same cell).

    Returns
    -------
    pandas.DataFrame with columns:
        global_node_id      int64        canonical IEM cell index
        contained_in_sites  list[str]    site_uids whose grid includes the cell
        n_sites             int64        len(contained_in_sites)
        lat, lon            float64      cell centroid (WGS84), as in the grid

    Built by data.weather.make_grid.build_global_grid (refreshed with the
    per-site grids); raises FileNotFoundError if it has not been generated yet.
    """
    return _get_global_grid()


def aggregate_by_interval(site_uid: str, interval="1D", agg_func_water="mean", agg_func_weather="sum", inplace=False):
    return get_data(site_uid=site_uid).agg()
