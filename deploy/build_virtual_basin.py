"""Build a virtual SiteData for an arbitrary (lat, lon) inside Iowa.

Pipeline (fixed TARGET_YEAR, default 2017) -- reuses the refactored data/ builders, so no
data-construction logic is duplicated here:

    (lat, lon)                                            the virtual sensor location (used raw)
      -> NLDI position-snap basin                     data.basins.make_basins._compute_basin1
      -> Voronoi target-cell grid (D8 flow dist + basin frac)  make_grid.build_grid_from_basin
      -> weather over [YEAR - 2mo, YEAR + 2mo]         make_basin_weather.weather_for_grid
      -> crops for YEAR (CDL raster -> grid)           make_crops.crops_grid
      -> surplus for YEAR (pixel source -> grid)       make_surplus.surplus_grid
      -> data.access.SiteData(water=None)

The returned SiteData has the same schema as a real get_data() result, so the generalized
data.features / recipes3 feature builders consume it unchanged.

Note: the raw (lat, lon) is used directly (matching how real sensors use their metadata
coordinates). An earlier design snapped the point to the containing weather-grid cell centroid
first, but that ~2 km move can cross catchment boundaries -- e.g. at a main-stem site it snapped
onto a headwater tributary and collapsed a 34,000 km^2 basin to ~1 km^2 -- so it was dropped.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data" / "surplus"))  # make_surplus's bare `import gen_surplus_statistics`

from data.access import SiteData
from data.basins import make_basins
from data.weather import make_grid, make_basin_weather
from data.crops import make_crops
from data.surplus import make_surplus

TARGET_YEAR = 2017
_BUFFER = pd.DateOffset(months=2)  # weather lead-in/out around the year for rolling/lag lookback
VIRTUAL_UID = "VIRTUAL"  # a uid absent from the state -> neighbour nitrate = full rest-of-state avg


def build_virtual_basin(lat: float, lon: float, target_year: int = TARGET_YEAR, timeout: int = 60) -> SiteData:
    """Assemble a SiteData (water=None) for a virtual sensor at (lat, lon) for `target_year`.

    Raises ValueError if the point is outside the D8 / weather-grid coverage, has no NLDI
    basin, or yields no crop / surplus / weather for the basin.
    """
    # 1) NLDI position-snap basin at the (raw) sensor location
    basin = make_basins._compute_basin1(VIRTUAL_UID, lat, lon, timeout=timeout)
    if basin is None or basin.empty:
        raise ValueError(f"No NLDI basin near ({lat:.5f}, {lon:.5f}).")

    # 2) Voronoi target-cell grid with D8 flow distance + basin-area fraction
    grid = make_grid.build_grid_from_basin(basin, lat, lon, label=VIRTUAL_UID)
    if grid is None or len(grid) == 0:
        raise ValueError(f"No grid cells intersect the basin at ({lat:.5f}, {lon:.5f}).")
    if grid["dist_to_sensor"].isna().all():
        raise ValueError(
            f"dist_to_sensor is NaN for every cell at ({lat:.5f}, {lon:.5f}) -- the sensor is "
            f"likely outside the D8 flow-direction raster (Iowa) coverage."
        )

    # 3) weather over [year - 2mo, year + 2mo] (buffer for trailing rolling/lag features)
    ystart = pd.Timestamp(f"{target_year}-01-01")
    yend = pd.Timestamp(f"{target_year}-12-31")
    wstart, wend = ystart - _BUFFER, yend + _BUFFER
    years = sorted({wstart.year, target_year, wend.year})
    weather = make_basin_weather.weather_for_grid(grid, start=wstart, end=wend, years=years)
    if weather.empty:
        raise ValueError(f"No global weather for the basin in years {years}.")

    # 4) crops + surplus for the target year (statewide sources -> this grid)
    crops = make_crops.crops_grid(grid=grid, years=[target_year])
    if crops is None or crops.empty:
        raise ValueError(f"No CDL crop data for the basin in {target_year}.")
    surplus = make_surplus.surplus_grid(grid=grid, year=target_year)
    if surplus is None or surplus.empty:
        raise ValueError(f"No nitrogen-surplus data for the basin in {target_year}.")

    # 5) assemble -- same schema as a real get_data() result (basin_area matches get_basin_area)
    basin_area = float(np.dot(grid["cell_area"].to_numpy(), grid["frac_cell_in_basin"].to_numpy()))
    return SiteData(
        site_uid=VIRTUAL_UID,
        basin=basin,
        crops=crops,
        grid=grid,
        surplus=surplus,
        water=None,
        weather=weather,
        basin_area=basin_area,
        sensor_location=(lon, lat),
    )
