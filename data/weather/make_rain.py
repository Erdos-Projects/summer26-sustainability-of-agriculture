"""
Per-site drainage-basin rainfall time series builder.

Source : IEM (Iowa Environmental Mesonet) daily precipitation grid polygons
         https://mesonet.agron.iastate.edu/rainfall/

For each monitoring site that has a basin file in data/water/basins/, this
script downloads IEM daily precipitation polygons covering the site's active
date range, spatially filters the Iowa-region grid to cells that intersect the
site's drainage basin, and writes one row per (grid cell, day) to a parquet
file at data/weather/rain/<uid>_rain.parquet.

Output schema
-------------
date            datetime64[ms]   calendar date of the observation
lon             float64          centroid longitude of the IEM grid cell (WGS84)
lat             float64          centroid latitude  of the IEM grid cell (WGS84)
precip_1d_in    float64          daily precipitation in inches for that cell
year            int32
month           int32
day_of_year     int32
week            int64

The _1d suffix on precip_in_1d leaves room for rolling-window columns
(precip_in_3d, precip_in_7d, ...) to be added in downstream processing.

Loop structure — date-first
---------------------------
All basins are loaded upfront and a date -> [uid] index is built.  For each
unique calendar day, the IEM zip is downloaded and parsed exactly once, then
gpd.sjoin filters the grid to each site's basin.  This avoids re-parsing the
same zip N_sites times when date ranges overlap heavily.

Memory management: each site's accumulated rows are written to disk and freed
as soon as its last required date has been processed.  At most one site's full
date range resides in memory at once (in the worst case where all sites share
the same last date).

Coverage check
--------------
Before a site is queued, the fraction of its basin area that falls within the
IEM data footprint is computed.  Sites below COVERAGE_THRESHOLD are skipped:
their basins are mostly outside the IEM region and would produce misleadingly
sparse data (e.g. sites whose basins extend into Montana).

CRS notes
---------
IEM shapefiles are served as EPSG:4269 (NAD83) despite the URL requesting
EPSG:4326. The two datums are identical for practical purposes (sub-metre
difference). The parse step re-labels the CRS as EPSG:4326 so it matches the
basin files without a coordinate transformation.

Centroids are computed in geographic coordinates (EPSG:4326).  For ~4 km grid
cells at Iowa's latitude the error vs. a metric-CRS centroid is roughly 10 m —
negligible given the 4 km spatial resolution of the rainfall data.
"""

import re
import warnings
import requests
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
from pathlib import Path
from tqdm import tqdm

# ── paths (all relative to this script's directory: data/weather/) ────────────
_THIS_DIR = Path(__file__).resolve().parent
_BASIN_DIR = _THIS_DIR.parents[1] / "data/water/basins"
_STATS_FILE = _THIS_DIR.parents[1] / "data/water/metadata/site_statistics.csv"
_RAW_DIR = _THIS_DIR / "rain_raw"  # cached IEM daily zips, shared across sites
_RAIN_DIR = _THIS_DIR / "rain"  # output: one parquet per site

# ── IEM API ───────────────────────────────────────────────────────────────────
# polygon geometry gives actual grid-cell extents so we can spatially filter
# to the basin and recover centroid lon/lat for each kept cell
_SHP_URL = (
    "https://mesonet.agron.iastate.edu/rainfall/dshape.php?"
    "month={month}&day={day}&year={year}"
    "&geometry=polygon"
    "&duration=day"
    "&epsg=4326"
)

# ── IEM footprint ─────────────────────────────────────────────────────────────
# Empirically measured from a sample download: lon -97.7->-87.4, lat 38.8->45.3.
# The IEM daily download covers Iowa plus roughly 1-2 degrees into neighbouring
# states (Nebraska, Minnesota, Wisconsin, Illinois, Missouri).
# Used only for the coverage check — not as a spatial clip.
_IEM_FOOTPRINT = gpd.GeoDataFrame.from_features(
    [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-97.7, 38.8], [-87.4, 38.8], [-87.4, 45.3], [-97.7, 45.3], [-97.7, 38.8]]],
            },
            "properties": {},
        }
    ],
    crs="EPSG:4326",
)

# Sites whose basin overlaps the IEM footprint by less than this fraction are
# skipped.  Tune upward to be more aggressive about excluding out-of-region sites.
COVERAGE_THRESHOLD = 0.75


# ── helpers ───────────────────────────────────────────────────────────────────


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _iowa_coverage(basin: gpd.GeoDataFrame) -> float:
    """Fraction of basin area that falls within the IEM data footprint."""
    clipped = gpd.overlay(basin[["geometry"]], _IEM_FOOTPRINT, how="intersection")
    if clipped.empty:
        return 0.0
    crs = "EPSG:26915"  # NAD83 UTM Zone 15N — metric CRS for Iowa
    return float(clipped.to_crs(crs).geometry.area.sum() / basin.to_crs(crs).geometry.area.sum())


def _download_shapefile(d: date) -> Path | None:
    """
    Fetch the IEM daily precipitation shapefile for date d into _RAW_DIR.
    Returns the local zip path, or None on network failure or missing-data days.
    Skips the network request entirely if the file is already cached.

    IEM returns an HTML error page (not a zip) for days with no data —
    detected by checking for the PK zip magic bytes.
    """
    path = _RAW_DIR / f"iowa_rain_{d.isoformat()}.zip"
    if path.exists():
        return path

    url = _SHP_URL.format(month=d.month, day=d.day, year=d.year)
    try:
        resp = requests.get(url, timeout=(10, 60))
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):
            return None
        path.write_bytes(resp.content)
        return path
    except Exception as e:
        print(f"  [WARN] {d}: {e}")
        return None


def _parse_day_gdf(zip_path: Path, d: date) -> gpd.GeoDataFrame | None:
    """
    Parse an IEM daily zip into a GeoDataFrame of precipitation polygons.

    Returns a GeoDataFrame in EPSG:4326 with columns:
        date (datetime.date), precip_in_1d (float), geometry (Polygon)

    The IEM API serves data labelled EPSG:4269 (NAD83) despite the epsg=4326
    URL parameter.  The datums are sub-metre apart so we re-label without
    transforming coordinates.

    The only data column in IEM polygon shapefiles is 'RAINFALL' (lowercased
    to 'rainfall' here), renamed to precip_in_1d for schema consistency.
    """
    try:
        gdf = gpd.read_file(f"/vsizip/{zip_path}")
    except Exception as e:
        print(f"  [WARN] parse {zip_path.name}: {e}")
        return None

    if gdf is None or gdf.empty:
        return None

    gdf.columns = [c.lower() for c in gdf.columns]

    if "rainfall" not in gdf.columns:
        print(f"  [WARN] unexpected columns in {zip_path.name}: {list(gdf.columns)}")
        return None

    # Re-label CRS: EPSG:4269 ≈ EPSG:4326, no coordinate transformation needed
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.rename(columns={"rainfall": "precip_in_1d"})
    gdf["date"] = d
    return gdf[["date", "precip_in_1d", "geometry"]].copy()


def _basin_filter(day_gdf: gpd.GeoDataFrame, basin: gpd.GeoDataFrame) -> pd.DataFrame | None:
    """
    Return one row per IEM grid cell that intersects the basin polygon.

    Uses gpd.sjoin with predicate='intersects' (faster than gpd.overlay since
    we only need membership, not clipped geometries).  Centroid lon/lat are
    extracted from the original grid-cell polygon geometry.

    The centroid warning for geographic CRS is suppressed: at ~4 km grid
    spacing the error vs. a metric-CRS centroid is ~10 m, negligible here.

    Returns a plain DataFrame with columns: date, lon, lat, precip_in_1d.
    Returns None if no cells intersect the basin.
    """
    inside = gpd.sjoin(
        day_gdf,
        basin[["geometry"]].to_crs(day_gdf.crs),
        how="inner",
        predicate="intersects",
    )

    if inside.empty:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inside["lon"] = inside.geometry.centroid.x
        inside["lat"] = inside.geometry.centroid.y

    return inside[["date", "lon", "lat", "precip_in_1d"]].reset_index(drop=True)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append calendar columns used as ML features."""
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year.astype("int32")
    df["month"] = dt.dt.month.astype("int32")
    df["day_of_year"] = dt.dt.day_of_year.astype("int32")
    df["week"] = dt.dt.isocalendar().week.astype("int64")
    return df


# ── precheck / main ───────────────────────────────────────────────────────────


def _precheck(stats: pd.DataFrame) -> bool:
    """Return True if every site in stats already has a rain parquet.

    Prints a summary of present and missing files.  When all files are
    present the caller should skip the main loop entirely.
    """
    all_uids = stats["site_uid"].tolist()
    missing  = [uid for uid in all_uids if not (_RAIN_DIR / f"{uid}_rain.parquet").exists()]
    present  = len(all_uids) - len(missing)

    print(f"Precheck: {present}/{len(all_uids)} rain parquets present in {_RAIN_DIR.name}/")

    if not missing:
        print("All parquets accounted for — nothing to do.")
        return True

    print(f"  {len(missing)} missing: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    return False


def main(_api_keys=None, site_uids: list[str] | None = None, force: bool = False):
    """Build per-site rainfall parquets.

    Parameters
    ----------
    site_uids : list of str, optional
        If provided, only process these site UIDs.  Useful for targeted
        re-runs or testing individual sites without triggering the full
        pipeline.  Unrecognised UIDs (no matching basin file) are warned
        and skipped.  If None, all WQS*/USGS-* basin files are processed.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _RAIN_DIR.mkdir(parents=True, exist_ok=True)

    stats = pd.read_csv(_STATS_FILE)

    if not force and site_uids is None and _precheck(stats):
        return

    # Only process sites whose uid matches WQS* or USGS-* — derived aggregate
    # basin files (e.g. union basins) do not match either pattern and are skipped.
    basin_files = [
        p
        for p in sorted(_BASIN_DIR.glob("*_basin.parquet"))
        if re.match(r"^(WQS|USGS-)", p.stem.replace("_basin", ""))
    ]

    if site_uids is not None:
        uid_set = set(site_uids)
        found   = {p.stem.replace("_basin", "") for p in basin_files}
        for uid in sorted(uid_set - found):
            print(f"  [WARN] {uid}: no basin file found, skipping")
        basin_files = [p for p in basin_files if p.stem.replace("_basin", "") in uid_set]
        print(f"Processing {len(basin_files)} of {len(site_uids)} requested sites.")
    else:
        print(f"Found {len(basin_files)} basin files matching WQS*/USGS-* pattern.")

    # ── Phase 1: load basins and build date → [uid] index ─────────────────────
    # Basins are small (single polygon each) so holding all 90 in memory is fine.
    # The date index enables date-first processing: each daily zip is parsed once
    # and dispatched to all sites that need that date.

    basins = {}  # uid -> GeoDataFrame (single polygon row)
    frames = {}  # uid -> list of per-day DataFrames (freed after write)
    end_dates = {}  # uid -> last date needed (triggers write + free)
    date_to_uids = {}  # date -> [uid, ...]

    for basin_path in basin_files:
        uid = basin_path.stem.replace("_basin", "")

        if not force and (_RAIN_DIR / f"{uid}_rain.parquet").exists():
            continue

        row = stats[stats["site_uid"] == uid]
        if row.empty:
            print(f"  [SKIP] {uid}: no entry in site_statistics.csv")
            continue

        basin = gpd.read_parquet(basin_path)

        frac = _iowa_coverage(basin)
        if frac < COVERAGE_THRESHOLD:
            print(f"  [SKIP] {uid}: {frac:.0%} of basin within IEM footprint " f"(threshold {COVERAGE_THRESHOLD:.0%})")
            continue

        # start_date / last_date are full ISO strings with timezone offset,
        # e.g. "2012-05-30 21:40:00+00:00" — pd.to_datetime handles the tz
        start = pd.to_datetime(row.iloc[0]["start_date"]).date()
        end = pd.to_datetime(row.iloc[0]["last_date"]).date()

        basins[uid] = basin
        frames[uid] = []
        end_dates[uid] = end

        for d in _date_range(start, end):
            date_to_uids.setdefault(d, []).append(uid)

    # ── Phase 2: date-first download → parse → filter loop ────────────────────

    all_dates = sorted(date_to_uids)
    print(f"{len(all_dates)} unique dates across {len(basins)} sites.")

    for d in tqdm(all_dates, desc="days", unit="day"):
        zip_path = _download_shapefile(d)
        if zip_path is None:
            continue

        day_gdf = _parse_day_gdf(zip_path, d)
        if day_gdf is None:
            continue

        for uid in date_to_uids[d]:
            chunk = _basin_filter(day_gdf, basins[uid])
            if chunk is not None:
                frames[uid].append(chunk)

            # Write and free memory as soon as a site's last date is reached.
            # Dates are processed in chronological order so this fires exactly once.
            if d == end_dates[uid] and frames[uid]:
                df = pd.concat(frames[uid], ignore_index=True)
                df["date"] = pd.to_datetime(df["date"])
                df = _add_time_features(df)
                out = _RAIN_DIR / f"{uid}_rain.parquet"
                df.to_parquet(out, index=False)
                print(f"\n  {uid}: {len(df):,} rows -> {out.name}")
                del frames[uid]
                del basins[uid]

    # ── Phase 3: write any sites not yet flushed ──────────────────────────────
    # Fires for sites whose last date is shared with other sites — only the
    # last uid processed in date_to_uids[end_date] triggered a write above.

    for uid, frame_list in frames.items():
        if not frame_list:
            continue
        df = pd.concat(frame_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = _add_time_features(df)
        out = _RAIN_DIR / f"{uid}_rain.parquet"
        df.to_parquet(out, index=False)
        print(f"  {uid}: {len(df):,} rows -> {out.name}")


if __name__ == "__main__":
    main()
