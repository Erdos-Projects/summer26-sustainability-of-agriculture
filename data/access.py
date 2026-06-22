from dataclasses import dataclass
import pandas as pd
import numpy as np
import geopandas as gpd
from .basins import get_basin
from .crops import get_crops
from .rain import get_rain, get_rain_grid as _get_rain_grid, aggregate_by_interval as agg_rain
from .surplus import get_surplus
from .water import get_water, get_site_ids as get_ids, aggregate_by_interval as agg_water

_ALL_FIELDS = ("basin", "crops", "grid", "rain", "surplus", "water")


def get_rain_grid(site_uid: str):
    """Top-level access to a site's rain grid (the shared aggregation grid).

    Thin pass-through to data.rain.get_rain_grid so surplus/crop builders can
    depend on the public access layer rather than importing the rain module.
    """
    return _get_rain_grid(site_uid)


@dataclass
class SiteData:
    site_uid: str
    basin: gpd.GeoDataFrame | None = None
    crops: pd.DataFrame | None = None
    grid: pd.DataFrame | None = None
    rain: pd.DataFrame | None = None
    surplus: pd.DataFrame | None = None
    water: pd.DataFrame | None = None
    basin_area: float | None = None

    def has(self, key: str) -> bool:
        """Return True if the named field was successfully loaded."""
        return getattr(self, key) is not None

    def available(self) -> list[str]:
        """Return names of fields that were successfully loaded."""
        return [f for f in _ALL_FIELDS if getattr(self, f) is not None]

    def agg(self, interval="1D", agg_func_water="mean", agg_func_rain="sum", inplace=False):
        new_water = agg_water(df=self.water, interval=interval, agg_func=agg_func_water)
        new_rain = agg_rain(df=self.rain, interval=interval, agg_func=agg_func_rain)

        new_data = SiteData(
            site_uid=self.site_uid,
            basin=self.basin,
            crops=self.crops,
            grid=self.grid,
            rain=new_rain,
            surplus=self.surplus,
            water=new_water,
            basin_area=self.basin_area,
        )

        if inplace:
            self = new_data

        return new_data

    # alias of agg to match the rain and water submodules
    aggregate_by_interval = agg


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
        grid=_try(get_rain_grid, site_uid) if "grid" in load else None,
        rain=_try(get_rain, site_uid) if "rain" in load else None,
        surplus=_try(get_surplus, site_uid) if "surplus" in load else None,
        water=_try(get_water, site_uid) if "water" in load else None,
        basin_area=get_basin_area(site_uid) if "grid" in load else None,
    )


def get_basin_area(site_uid):
    grid = get_rain_grid(site_uid=site_uid)
    cell_areas = np.array(grid.cell_area.values)
    overlaps = np.array(grid.frac_cell_in_basin.values)
    return np.dot(cell_areas * overlaps)


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


def aggregate_by_interval(site_uid: str, interval="1D", agg_func_water="mean", agg_func_rain="sum", inplace=False):
    return get_data(site_uid=site_uid).agg()
