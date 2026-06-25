"""Build the global yearly weather files.

For every cell in the *entire* IEM precipitation grid (~23,182 cells covering
Iowa plus a buffer into neighbouring states) and every day of a year, combine IEM
precipitation with gridMET meteorology into one dense table, written to
weather_global/global_grid_weather_{year}.parquet. Covering the whole IEM grid
(not just cells inside a basin) lets downstream models simulate anywhere in the
region, including outside the monitored basins.

Two sources, joined on (date, global_node_id):
  * precip_in_1d  — IEM daily precip (inches). Because global_node_id IS the IEM
    grid row index, a cell's precip is a direct array lookup vals[global_node_id]
    into the day's RAINFALL array (geometry-free read of the cached IEM zip).
  * gridMET vars  — one get_bygeom over the region bbox per year, bilinearly
    interpolated to each cell centroid. Temps are converted K -> degC.

Output columns
--------------
date, global_node_id, precip_in_1d (in), max_temp (degC), min_temp (degC),
max_rel_humidity (%), min_rel_humidity (%), vpd (kPa), solar_rad (W/m^2),
evapotranspiration (mm, gridMET pet), fuel_moisture_1000h (%).

lon/lat (constant per global_node_id; see weather_grid) and calendar fields
(year/month/week/day_of_year, derivable from date) are intentionally omitted.

Usage
-----
    python make_global_weather.py                  # all years, skip existing
    python make_global_weather.py --force          # rebuild all
    python make_global_weather.py --year 2012 --year 2013
"""

import os
import sys
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

_THIS_DIR = Path(__file__).resolve().parent
_GRID_DIR = _THIS_DIR / "weather_grid"
_GLOBAL_GRID_FILE = _GRID_DIR / "global_grid.parquet"
_GLOBAL_DIR = _THIS_DIR / "weather_global"
_GRIDMET_CACHE = _THIS_DIR / "weather_raw" / "gridMET_raw"

# Point pygridmet's NetCDF cache at weather_raw/gridMET_raw/ (it uses the parent
# of HYRIVER_CACHE_NAME). Must be set before importing pygridmet.
_GRIDMET_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HYRIVER_CACHE_NAME", str(_GRIDMET_CACHE / "hyriver.sqlite"))

import pygridmet  # noqa: E402
import pygridmet.pygridmet as _pgm  # noqa: E402

# gridMET is NaN over water / outside the CONUS land mask, and pygridmet treats
# ANY NaN in the grid as a failed download — it deletes the file, re-downloads,
# finds the same NaN, and after a few retries raises "did NOT process your
# request". So any tile clipping a lake (e.g. Lake Michigan on the eastern edge
# of the IEM footprint) fails *deterministically*. We accept NaN: we only sample
# land cells, and near-shore cells are nearest-filled in _sample_gridmet.
_pgm._check_nans = lambda clm, urls, clm_files, long2abbr: (False, urls)

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data.weather.make_grid import _download_shapefile, _parse_day_gdf, IEM_GRID_DATE, _ALBERS  # noqa: E402
from data.settings import get_region_bbox  # noqa: E402

ALL_YEARS = list(range(2008, 2027))

# gridMET variable code -> output column name (pr excluded; IEM supplies precip).
# Only the variables kept for nitrate modeling are fetched.
_GRIDMET_RENAME = {
    "tmmx": "max_temp",
    "tmmn": "min_temp",
    "rmax": "max_rel_humidity",
    "rmin": "min_rel_humidity",
    "vpd": "vpd",
    "srad": "solar_rad",
    "pet": "evapotranspiration",
    "fm1000": "fuel_moisture_1000h",
}
_GRIDMET_VARS = list(_GRIDMET_RENAME)
_TEMP_COLS = ["max_temp", "min_temp"]  # K -> degC

_COLUMN_ORDER = [
    "date",
    "global_node_id",
    "precip_in_1d",
    "max_temp",
    "min_temp",
    "max_rel_humidity",
    "min_rel_humidity",
    "vpd",
    "solar_rad",
    "evapotranspiration",
    "fuel_moisture_1000h",
]


def _parse_day_values(zip_path: Path, expected_n: int) -> np.ndarray | None:
    """RAINFALL (inches) for every IEM cell, in record order (geometry-free read).

    Positionally aligned to the canonical IEM grid, so vals[global_node_id] is a
    cell's precip. Returns None on a missing/odd day.
    """
    try:
        df = gpd.read_file(f"/vsizip/{zip_path}", ignore_geometry=True)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df.columns = [c.lower() for c in df.columns]
    if "rainfall" not in df.columns or len(df) != expected_n:
        return None
    return df["rainfall"].to_numpy()


def _iem_cells() -> pd.DataFrame:
    """The entire IEM grid: global_node_id (row index), lon, lat (centroid, WGS84).

    global_node_id is the cell's row index in the canonical IEM grid, so it equals
    the per-site grids' global_node_id and indexes the daily RAINFALL array.
    Centroids are computed exactly as make_grid does (projected -> WGS84), so they
    match the per-site grid coordinates for shared cells.
    """
    day = _parse_day_gdf(_download_shapefile(IEM_GRID_DATE), IEM_GRID_DATE).to_crs(_ALBERS)
    ll = day.geometry.centroid.to_crs("EPSG:4326")
    return pd.DataFrame(
        {
            "global_node_id": np.arange(len(day), dtype="int64"),
            "lon": ll.x.to_numpy(),
            "lat": ll.y.to_numpy(),
        }
    )


def _get_bygeom_retry(bbox: tuple, dates: tuple, attempts: int = 6) -> xr.Dataset:
    """get_bygeom with patient exponential-backoff retries.

    The gridMET NCSS server intermittently rejects valid requests under load
    (same generic 'did NOT process your request' message it uses for oversize
    requests), so we retry with growing waits to ride out transient spikes.
    Already-downloaded variables are cached, so a retry only re-fetches failures.
    """
    import time
    from pygridmet.exceptions import InputRangeError

    for k in range(attempts):
        try:
            return pygridmet.get_bygeom(bbox, dates, variables=_GRIDMET_VARS)
        except InputRangeError:
            raise  # dates outside gridMET's available range — retrying can't help
        except Exception as e:
            if k == attempts - 1:
                raise
            wait = min(30 * 2**k, 300)  # 30, 60, 120, 240, 300 s
            print(f"    gridMET fetch failed ({type(e).__name__}); retry {k + 1}/{attempts - 1} in {wait}s")
            time.sleep(wait)


# The gridMET NCSS endpoint rejects spatial subsets much larger than ~Iowa
# (~12k grid points OK, the full IEM footprint ~38k is refused), so requests are
# tiled. Each cell is sampled from exactly one tile; the tile is fetched with a
# small buffer so cells near a tile edge still have neighbours for interpolation.
_MAX_TILE_DEG = 5.0
_TILE_BUFFER = 0.1  # deg, > 1 gridMET cell (~0.042 deg)


def _sample_gridmet(ds: xr.Dataset, cells: pd.DataFrame) -> pd.DataFrame:
    """Bilinearly sample a gridMET dataset at the given cell centroids -> long df."""
    ds = ds.sortby("lat").sortby("lon")  # monotonic coords for interp
    lon_da = xr.DataArray(cells["lon"].to_numpy(), dims="cell")
    lat_da = xr.DataArray(cells["lat"].to_numpy(), dims="cell")
    samp = ds.interp(lon=lon_da, lat=lat_da, method="linear")  # bilinear -> (time, cell)
    # near-shore cells whose bilinear neighbours include a water-NaN: fill from the
    # nearest grid cell (cells truly over water stay NaN — they're lakes).
    samp = samp.fillna(ds.interp(lon=lon_da, lat=lat_da, method="nearest"))
    samp = samp.assign_coords(cell=("cell", cells["global_node_id"].to_numpy()))

    gm = samp.to_dataframe().reset_index()
    gm = gm.drop(columns=[c for c in ("spatial_ref", "lat", "lon") if c in gm.columns])
    gm = gm.rename(columns={"time": "date", "cell": "global_node_id", **_GRIDMET_RENAME})
    gm["date"] = pd.to_datetime(gm["date"]).dt.normalize()
    for c in _TEMP_COLS:
        gm[c] = gm[c] - 273.15
    return gm


def _gridmet_range(cells: pd.DataFrame, bbox: tuple, start: str, end: str) -> pd.DataFrame:
    """gridMET vars for every (day, cell) in [start, end], bilinearly sampled.

    Tiles the bbox so each NCSS request stays under the server's size cap; every
    cell is assigned to exactly one tile (by centroid) and sampled from that
    tile's (buffered) fetch.
    """
    minx, miny, maxx, maxy = bbox
    nx = max(1, math.ceil((maxx - minx) / _MAX_TILE_DEG))
    ny = max(1, math.ceil((maxy - miny) / _MAX_TILE_DEG))
    xs = np.linspace(minx, maxx, nx + 1)
    ys = np.linspace(miny, maxy, ny + 1)
    ix = np.clip(np.searchsorted(xs, cells["lon"].to_numpy(), side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, cells["lat"].to_numpy(), side="right") - 1, 0, ny - 1)
    tile = ix * ny + iy

    out = []
    for t in np.unique(tile):
        i, j = divmod(int(t), ny)
        sub = cells[tile == t]
        fetch = (xs[i] - _TILE_BUFFER, ys[j] - _TILE_BUFFER, xs[i + 1] + _TILE_BUFFER, ys[j + 1] + _TILE_BUFFER)
        ds = _get_bygeom_retry(fetch, (start, end))
        out.append(_sample_gridmet(ds, sub))
    return pd.concat(out, ignore_index=True)


def _assemble(gm: pd.DataFrame, prec: pd.DataFrame) -> pd.DataFrame:
    """Join gridMET + IEM precip, order columns.

    lon/lat (constant per global_node_id, in the grid) and calendar fields
    (derivable from date) are intentionally not stored — see weather_grid for
    coordinates and recompute calendar fields from `date` if needed.
    """
    df = gm.merge(prec, on=["date", "global_node_id"], how="left")
    df["date"] = df["date"].astype("datetime64[ms]")
    df = df[_COLUMN_ORDER].sort_values(["date", "global_node_id"], ignore_index=True)

    # shrink: float32 weather values + int32 cell id
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df["global_node_id"] = df["global_node_id"].astype("int32")
    return df


def _iem_precip_year(year: int, cells: pd.DataFrame, dates: pd.DatetimeIndex, n_ref: int) -> pd.DataFrame:
    """IEM precip (inches) for every (day, cell) in `year` via vals[global_node_id]."""
    gids = cells["global_node_id"].to_numpy()
    precip = np.full((len(dates), len(gids)), np.nan)
    for i, d in enumerate(dates):
        zp = _download_shapefile(d.date())
        if zp is None:
            continue
        vals = _parse_day_values(zp, n_ref)
        if vals is None:
            continue
        precip[i] = vals[gids]
    return pd.DataFrame(
        {
            "date": np.repeat(dates.values, len(gids)),
            "global_node_id": np.tile(gids, len(dates)),
            "precip_in_1d": precip.ravel(),
        }
    )


def build_year(year: int, cells: pd.DataFrame, bbox: tuple, n_ref: int) -> None:
    """Build global_grid_weather_{year}.parquet, streaming one month at a time.

    Each month is fetched + assembled + appended independently, so peak memory is
    ~one month rather than a full year. The year is written to a .tmp file and
    renamed on success, so an interrupted run leaves no partial output (and the
    cached gridMET .nc files make the re-run cheap).
    """
    # gridMET publishes with a lag, so the most recent weeks aren't available yet.
    # Requesting them returns a deterministic InputRangeError (not a transient
    # outage), so cap every request at one month ago and never touch the current,
    # still-incomplete month.
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=1)
    out = _GLOBAL_DIR / f"global_grid_weather_{year}.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    writer = None
    total = 0
    try:
        for month in range(1, 13):
            m_start = pd.Timestamp(year=year, month=month, day=1)
            if m_start > cutoff:  # month is newer than gridMET's available range
                break
            m_end = min(m_start + pd.offsets.MonthEnd(0), cutoff)
            gm = _gridmet_range(cells, bbox, m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            if gm.empty:
                continue
            dates = pd.DatetimeIndex(sorted(gm["date"].unique()))
            prec = _iem_precip_year(year, cells, dates, n_ref)
            df = _assemble(gm, prec)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(df)
            print(f"    {year}-{month:02d}: {len(df):,} rows (running {total:,})")
        if writer is not None:
            writer.close()
            writer = None
    except BaseException:  # outage/interrupt mid-year -> discard the partial file
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise

    if total == 0:
        if tmp.exists():
            tmp.unlink()
        print(f"  {year}: no data")
        return
    tmp.replace(out)
    print(f"  {year}: {total:,} rows ({len(cells):,} cells x {total // len(cells)} days) -> {out.name}")


def build_global_weather(years: list[int] | None = None, force: bool = False) -> None:
    """Build/refresh the yearly global weather files (IEM cells in the region bbox)."""
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

    all_cells = _iem_cells()  # the full IEM grid (global_node_id == IEM row index)
    n_ref = len(all_cells)  # full record count: validates each daily IEM shapefile

    # Restrict the global weather to the project region (settings.py bbox), not the
    # entire IEM footprint. global_node_id is preserved as the canonical IEM row
    # index, so vals[global_node_id] precip lookups and per-site joins still work.
    min_lon, min_lat, max_lon, max_lat = get_region_bbox()
    in_region = (
        all_cells.lon.between(min_lon, max_lon) & all_cells.lat.between(min_lat, max_lat)
    )
    cells = all_cells[in_region].reset_index(drop=True)

    buf = 0.1  # deg, > 1 gridMET cell so edge cells interpolate cleanly
    bbox = (min_lon - buf, min_lat - buf, max_lon + buf, max_lat + buf)
    print(
        f"Global weather: {len(cells):,} of {n_ref:,} IEM cells in region "
        f"| bbox {tuple(round(b, 2) for b in bbox)}"
    )

    years = years or ALL_YEARS
    failed = []
    for year in years:
        out = _GLOBAL_DIR / f"global_grid_weather_{year}.parquet"
        if out.exists() and not force:
            print(f"  {year}: exists, skipping")
            continue
        try:
            build_year(year, cells, bbox, n_ref)
        except Exception as e:  # a gridMET outage window — keep finished years, move on
            failed.append(year)
            print(f"  {year}: FAILED ({type(e).__name__}: {str(e)[:90]}) — re-run to fill it in")
    if failed:
        print(f"\n{len(failed)} year(s) failed (likely gridMET outages): {failed}. Re-run to complete them.")


def main(api_keys=None, force: bool = False, years: list[int] | None = None):
    build_global_weather(years=years, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing years.")
    parser.add_argument("--year", action="append", type=int, help="Limit to these years (repeatable).")
    args = parser.parse_args()
    main(force=args.force, years=args.year)
