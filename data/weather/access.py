"""Read-only access layer for weather data in data/weather/."""

import glob
import re
from typing import Callable
import pandas as pd
import geopandas as gpd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_GRID_DIR = _THIS_DIR / "weather_grid"
_GLOBAL_DIR = _THIS_DIR / "weather_global"
_DATA_DIR = _THIS_DIR / "weather_data"
_GLOBAL_GRID_FILE = _GRID_DIR / "global_grid.parquet"


def get_grid(site_uid: str) -> gpd.GeoDataFrame:
    """Load a site's grid (Voronoi target cells).

    Columns: node_id, global_node_id, x, y (EPSG:5070), lat, lon, cell_area,
    dist_to_sensor, frac_cell_in_basin, geometry. node_id is the basin-local join
    key for the surplus/crop aggregates; global_node_id is the canonical IEM cell
    index, shared across basins.

    Raises FileNotFoundError if it has not been generated yet (run make_grid).
    """
    path = _GRID_DIR / f"{site_uid}_grid.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No grid for {site_uid}. Run make_grid to generate {path.name}.")
    return gpd.read_parquet(path)


def get_global_grid() -> pd.DataFrame:
    """Load the global grid: each cell -> the sites whose basin contains it.

    Columns: global_node_id, contained_in_sites (list[str]), n_sites, lat, lon.
    Only cells in at least one preferred basin appear.

    Raises FileNotFoundError if not generated yet (run make_grid).
    """
    if not _GLOBAL_GRID_FILE.exists():
        raise FileNotFoundError(f"No global grid. Run make_grid to generate {_GLOBAL_GRID_FILE.name}.")
    return pd.read_parquet(_GLOBAL_GRID_FILE)


def get_weather(site_uid: str) -> pd.DataFrame:
    """Load a site's weather timeseries (one row per cell per day).

    Columns: date, node_id, global_node_id, precip_in_1d (in, IEM),
    max_temp/min_temp (degC), max/min_rel_humidity (%), vpd (kPa), solar_rad
    (W/m^2), evapotranspiration (mm, gridMET pet), fuel_moisture_1000h (%). Spans
    the site's nitrate record padded +/-60 days. Join to the surplus/crop
    aggregates on node_id.

    Large sites are stored split across {site_uid}_weather_p1.parquet,
    _p2.parquet, ... (each kept under GitHub's file-size limit); if the combined
    {site_uid}_weather.parquet is absent, those ordered parts are concatenated
    back into the full timeseries transparently.

    Raises FileNotFoundError if not generated yet (run make_weather).
    """
    path = _DATA_DIR / f"{site_uid}_weather.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        parts = sorted(
            _DATA_DIR.glob(f"{site_uid}_weather_p*.parquet"),
            key=lambda p: int(re.search(r"_p(\d+)\.parquet$", p.name).group(1)),
        )
        if not parts:
            raise FileNotFoundError(f"No weather for {site_uid}. Run make_weather to generate {path.name}.")
        df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_global_weather(year: int) -> pd.DataFrame:
    """Load the global weather file for a year (every global cell x every day).

    Same weather columns as get_weather (keyed by date, global_node_id; no
    node_id). Raises FileNotFoundError if that year has not been built.
    """
    path = _GLOBAL_DIR / f"global_grid_weather_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No global weather for {year}. Run make_global_weather to generate {path.name}.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_weather_years() -> list[int]:
    """Years for which a global weather file exists, sorted."""
    years = []
    for p in glob.glob(str(_GLOBAL_DIR / "global_grid_weather_*.parquet")):
        m = re.search(r"global_grid_weather_(\d{4})\.parquet$", p)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def aggregate_by_interval(
    site_uid: str | None = None,
    df: pd.DataFrame | None = None,
    value_col: str = "precip_in_1d",
    interval: str = "3D",
    agg_func: str | Callable = "sum",
) -> pd.DataFrame:
    """Aggregate a site's weather by a temporal interval, per grid cell.

    Spatial structure is preserved: the output has one row per (grid cell,
    period), where each period covers `interval` days and `date` marks the start
    of the period. Cells are keyed by node_id/global_node_id (weather files no
    longer carry lon/lat — join the grid for coordinates).

    Parameters
    ----------
    site_uid : str, optional
        Site identifier; used to load weather when df is not provided.
    df : DataFrame, optional
        Pre-loaded weather DataFrame (output of get_weather). Takes precedence
        over site_uid if both are supplied.
    value_col : str
        Column to aggregate. Defaults to 'precip_in_1d'.
    interval : str
        Pandas offset alias, e.g. '1D', '3D', '1W', '1MS', '3MS', '1YS'.
    agg_func : str or callable
        Aggregation function. Defaults to 'sum' (natural for precipitation).

    Returns
    -------
    DataFrame with columns: <cell keys>, date, precip_<interval>.
    """
    if df is None:
        if site_uid is None:
            raise ValueError("provide either df or site_uid")
        df = get_weather(site_uid)

    keys = [k for k in ("node_id", "global_node_id") if k in df.columns]
    result = (
        df.set_index("date")
        .groupby(keys)[value_col]
        .resample(interval)
        .agg(agg_func)
        .reset_index()
        .rename(columns={value_col: f"precip_{interval.lower()}"})
    )

    return result
