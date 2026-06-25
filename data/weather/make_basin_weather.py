"""Build per-site weather files from the global yearly weather files.

For each site with a grid, slice the global_grid_weather_{year}.parquet files to
the site's cells over its date range (padded +/-60 days), join the basin-local
node_id, and write weather_data/{site_uid}_weather.parquet.

The date range is the site's nitrate observation span (from site_statistics.csv)
padded by 60 days on each end (clamped to the available weather years), so each
site carries a little weather lead-in/out around its water record.

Columns: date, node_id, global_node_id, precip_in_1d, max_temp, min_temp,
max_rel_humidity, min_rel_humidity, vpd, solar_rad, evapotranspiration,
fuel_moisture_1000h. node_id joins to the surplus/crop aggregates; global_node_id
joins across basins. lon/lat (see weather_grid) and calendar fields (derivable
from date) are not stored.

Usage
-----
    python make_basin_weather.py                 # all sites, skip existing
    python make_basin_weather.py --force         # rebuild all
    python make_basin_weather.py --site WQS0012
"""

import re
import sys
import glob
import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd

_THIS_DIR = Path(__file__).resolve().parent
_GRID_DIR = _THIS_DIR / "weather_grid"
_GLOBAL_DIR = _THIS_DIR / "weather_global"
_DATA_DIR = _THIS_DIR / "weather_data"
_STATS_FILE = _THIS_DIR.parent / "water" / "water_meta" / "site_statistics.csv"

sys.path.insert(0, str(_THIS_DIR.parents[1]))

_PAD = pd.Timedelta(days=60)

_COLUMN_ORDER = [
    "date", "node_id", "global_node_id", "precip_in_1d",
    "max_temp", "min_temp", "max_rel_humidity", "min_rel_humidity",
    "vpd", "solar_rad", "evapotranspiration", "fuel_moisture_1000h",
]


def _available_years() -> list[int]:
    """Years for which a global weather file exists, sorted."""
    years = []
    for p in glob.glob(str(_GLOBAL_DIR / "global_grid_weather_*.parquet")):
        m = re.search(r"global_grid_weather_(\d{4})\.parquet$", p)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _site_date_range(site_uid: str, stats: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Padded [start, end] for a site, or None if it has no stats row."""
    row = stats[stats["site_uid"] == site_uid]
    if row.empty:
        return None
    start = pd.to_datetime(row.iloc[0]["start_date"]).tz_localize(None).normalize() - _PAD
    end = pd.to_datetime(row.iloc[0]["last_date"]).tz_localize(None).normalize() + _PAD
    return start, end


def build_site_weather(site_uid: str, stats: pd.DataFrame, avail_years: list[int]) -> bool:
    """Write one {site_uid}_weather.parquet. Returns True if written."""
    grid_path = _GRID_DIR / f"{site_uid}_grid.parquet"
    if not grid_path.exists():
        print(f"  [SKIP] {site_uid}: no grid")
        return False
    rng = _site_date_range(site_uid, stats)
    if rng is None:
        print(f"  [SKIP] {site_uid}: no site_statistics row")
        return False
    start, end = rng

    grid = gpd.read_parquet(grid_path)[["node_id", "global_node_id"]]
    cells = grid["global_node_id"].tolist()
    gid2node = dict(zip(grid["global_node_id"], grid["node_id"]))

    years = [y for y in avail_years if start.year <= y <= end.year]
    frames = []
    for y in years:
        gf = _GLOBAL_DIR / f"global_grid_weather_{y}.parquet"
        # predicate pushdown: read only this site's cells, not the whole year
        df = pd.read_parquet(gf, filters=[("global_node_id", "in", cells)])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        frames.append(df)

    if not frames or sum(len(f) for f in frames) == 0:
        print(f"  [SKIP] {site_uid}: no weather in {start.date()}..{end.date()} (years {years})")
        return False

    df = pd.concat(frames, ignore_index=True)
    df["node_id"] = df["global_node_id"].map(gid2node)
    df = df[_COLUMN_ORDER].sort_values(["date", "node_id"], ignore_index=True)

    # shrink: float32 weather values + int32 node ids, zstd-compressed
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df[["node_id", "global_node_id"]] = df[["node_id", "global_node_id"]].astype("int32")

    # writing the combined file supersedes any earlier split parts; drop them so
    # we never end up with both the combined file and its (now stale) _p* parts.
    for stale in _DATA_DIR.glob(f"{site_uid}_weather_p*.parquet"):
        stale.unlink()

    out = _DATA_DIR / f"{site_uid}_weather.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  {site_uid}: {len(df):,} rows, {df['node_id'].nunique()} cells, "
          f"{start.date()}..{end.date()} -> {out.name}")
    return True


def build_basin_weather(site_uids: list[str] | None = None, force: bool = False) -> None:
    """Build/refresh per-site weather files from the global yearly files."""
    from data import basins

    avail_years = _available_years()
    if not avail_years:
        raise FileNotFoundError("No global_grid_weather_*.parquet found — run make_global_weather first.")
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(_STATS_FILE)

    sites = site_uids or basins.get_metadata()["site_uid"].tolist()
    print(f"Basin weather: {len(sites)} site(s) | weather years {avail_years[0]}-{avail_years[-1]}")
    for uid in sites:
        # a site is "built" if its combined file OR its split parts exist; without
        # the parts check, previously-split large sites get rebuilt as one big file.
        combined = _DATA_DIR / f"{uid}_weather.parquet"
        has_parts = any(_DATA_DIR.glob(f"{uid}_weather_p*.parquet"))
        if (combined.exists() or has_parts) and not force:
            continue
        build_site_weather(uid, stats, avail_years)


def main(api_keys=None, force: bool = False, site_uids: list[str] | None = None):
    build_basin_weather(site_uids=site_uids, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing site files.")
    parser.add_argument("--site", action="append", help="Limit to these site UIDs (repeatable).")
    args = parser.parse_args()
    main(force=args.force, site_uids=args.site)
