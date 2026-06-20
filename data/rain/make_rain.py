"""
Per-site drainage-basin rainfall time series builder.

Source : IEM (Iowa Environmental Mesonet) daily precipitation grid polygons
         https://mesonet.agron.iastate.edu/rainfall/

For each monitoring site that has a basin file in data/basins/basins1/, this
script downloads IEM daily precipitation polygons covering the site's active
date range, spatially filters the Iowa-region grid to cells that intersect the
site's drainage basin, and writes one row per (grid cell, day) to a parquet
file at data/rain/rain/<uid>_rain.parquet.

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

import sys
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
from pathlib import Path
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, cKDTree
from tqdm import tqdm

# ── paths (all relative to this script's directory: data/rain/) ───────────────
_THIS_DIR = Path(__file__).resolve().parent
_STATS_FILE = _THIS_DIR.parent / "water" / "water_meta" / "site_statistics.csv"
_RAW_DIR = _THIS_DIR / "rain_raw"    # cached IEM daily zips, shared across sites
_RAIN_DIR = _THIS_DIR / "rain_data"  # output: one parquet per site
_GRID_DIR = _THIS_DIR / "rain_grid"  # output: one Voronoi-cell geoparquet per site
_MANIFEST_FILE = _RAIN_DIR / ".basin_manifest.csv"
_GRID_MANIFEST_FILE = _GRID_DIR / ".basin_manifest.csv"

# Geometry of the rain grid only — a single IEM day supplies the cell polygons.
IEM_GRID_DATE = date(2018, 6, 15)

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import basins
from data.settings import get_config, get_equal_area_crs

_ALBERS = get_equal_area_crs()  # equal-area CRS for the Voronoi/area math (EPSG:5070)

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
COVERAGE_THRESHOLD = get_config()["rain"]["coverage_threshold"]


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


# ── manifest / precheck ───────────────────────────────────────────────────────


def _stale_sites(preferred_meta: pd.DataFrame) -> list[str]:
    """Return UIDs that need a (re)build.

    A site is stale if its rain parquet is missing or if the basin recorded in
    .basin_manifest.csv differs from the current entry in preferred_basin.csv
    (i.e. the preferred basin was reassigned since the last run).
    """
    manifest = {}
    if _MANIFEST_FILE.exists():
        mdf = pd.read_csv(_MANIFEST_FILE)
        manifest = dict(zip(mdf["site_uid"], mdf["basin_name"]))

    stale = []
    for _, row in preferred_meta.iterrows():
        uid = row["site_uid"]
        parquet_missing = not (_RAIN_DIR / f"{uid}_rain.parquet").exists()
        basin_changed   = manifest.get(uid) != row["basin_name"]
        if parquet_missing or basin_changed:
            stale.append(uid)
    return stale


def _write_manifest(preferred_meta: pd.DataFrame) -> None:
    """Record site_uid → basin_name for every rain parquet that currently exists."""
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if (_RAIN_DIR / f"{row['site_uid']}_rain.parquet").exists()
    ]
    pd.DataFrame(rows).to_csv(_MANIFEST_FILE, index=False)


# ── Rain grid: padded-halo Voronoi target cells ──────────────────────────────
def _finite_voronoi(points: np.ndarray) -> dict[int, Polygon]:
    """Map point index -> its Voronoi cell polygon, skipping infinite/empty cells."""
    vor = Voronoi(points)
    polys = {}
    for pidx, ridx in enumerate(vor.point_region):
        verts = vor.regions[ridx]
        if not verts or -1 in verts:
            continue  # hull (infinite) or degenerate cell
        pts = vor.vertices[verts]
        c = pts.mean(axis=0)  # order vertices CCW so the polygon is valid
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys[pidx] = Polygon(pts)
    return polys


def build_rain_grid(site_uid: str) -> gpd.GeoDataFrame:
    """Build the rain grid (Voronoi target cells) for one basin.

    Returns a GeoDataFrame with columns node_id, x, y (EPSG:5070), lat, lon,
    cell_area, geometry. Cells are the IEM nodes whose cell intersects the basin;
    each is bounded by real neighbours via a padded halo so edge cells stay
    finite (infinite hull cells are dropped). The same node_id keys the surplus
    and crop aggregates downstream.
    """
    basin = basins.get_basin(site_uid).to_crs(_ALBERS)
    basin_poly = basin.geometry.union_all()
    minx, miny, maxx, maxy = basin.total_bounds

    # A single IEM day supplies the cell polygons; centroids give the node coords.
    day = _parse_day_gdf(_download_shapefile(IEM_GRID_DATE), IEM_GRID_DATE).to_crs(_ALBERS)
    centroids = day.geometry.centroid  # projected CRS -> accurate centroids
    lonlat = centroids.to_crs("EPSG:4326")
    grid = gpd.GeoDataFrame(
        {
            "x": centroids.x.to_numpy(),
            "y": centroids.y.to_numpy(),
            "lon": lonlat.x.to_numpy(),
            "lat": lonlat.y.to_numpy(),
        },
        geometry=day.geometry.values,
        crs=_ALBERS,
    )

    # Nodes whose cell touches the basin are the targets; max spacing sets the pad.
    grid["is_target"] = grid.geometry.intersects(basin_poly)
    tgt_xy = grid.loc[grid["is_target"], ["x", "y"]].to_numpy()
    if len(tgt_xy) < 2:
        # With 0/1 target node the k=2 query has no neighbour (returns inf -> pad
        # inf). Fall back to the grid-wide median spacing so a tiny basin works.
        all_xy = grid[["x", "y"]].to_numpy()
        pad = 2 * float(np.median(cKDTree(all_xy).query(all_xy, k=2)[0][:, 1]))
    else:
        pad = 2 * float(cKDTree(tgt_xy).query(tgt_xy, k=2)[0][:, 1].max())

    # Halo = every node within the padded bbox (a superset of the targets).
    px0, py0, px1, py1 = minx - pad, miny - pad, maxx + pad, maxy + pad
    halo = grid[grid["x"].between(px0, px1) & grid["y"].between(py0, py1)].reset_index(drop=True)
    polys = _finite_voronoi(halo[["x", "y"]].to_numpy())

    rows = []
    for i in halo.index[halo["is_target"]]:
        if i not in polys:
            print(f"  [warn] {site_uid}: a target node has an infinite cell (pad too small) — skipped")
            continue
        rows.append(
            {
                "node_id": len(rows),
                "x": halo.at[i, "x"],
                "y": halo.at[i, "y"],
                "lat": halo.at[i, "lat"],
                "lon": halo.at[i, "lon"],
                "cell_area": polys[i].area,
                "geometry": polys[i],
            }
        )
    return gpd.GeoDataFrame(rows, crs=_ALBERS)


def _grid_stale_sites(preferred_meta: pd.DataFrame) -> list[str]:
    """UIDs whose rain grid is missing or whose basin changed since last build."""
    manifest = {}
    if _GRID_MANIFEST_FILE.exists():
        mdf = pd.read_csv(_GRID_MANIFEST_FILE)
        manifest = dict(zip(mdf["site_uid"], mdf["basin_name"]))
    return [
        row["site_uid"]
        for _, row in preferred_meta.iterrows()
        if not (_GRID_DIR / f"{row['site_uid']}_rain_grid.parquet").exists()
        or manifest.get(row["site_uid"]) != row["basin_name"]
    ]


def _write_grid_manifest(preferred_meta: pd.DataFrame) -> None:
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if (_GRID_DIR / f"{row['site_uid']}_rain_grid.parquet").exists()
    ]
    pd.DataFrame(rows).to_csv(_GRID_MANIFEST_FILE, index=False)


def build_grids(site_uids: list[str] | None = None, force: bool = False) -> None:
    """Build/refresh the per-site rain grids. Depends only on basin geometry and
    the IEM grid (independent of the rain time series)."""
    _GRID_DIR.mkdir(parents=True, exist_ok=True)
    preferred_meta = basins.get_metadata()

    to_process = preferred_meta["site_uid"].tolist() if force else _grid_stale_sites(preferred_meta)
    if site_uids is not None:
        to_process = [u for u in to_process if u in set(site_uids)]

    if not to_process:
        print("Rain grids up to date.")
        return

    print(f"Building rain grids for {len(to_process)} site(s)...")
    for uid in to_process:
        try:
            grid = build_rain_grid(uid)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [SKIP] {uid}: {e}")
            continue
        if grid.empty:
            print(f"  [SKIP] {uid}: no rain cells intersect basin")
            continue
        out = _GRID_DIR / f"{uid}_rain_grid.parquet"
        grid.to_parquet(out)
        print(f"  {uid}: {len(grid)} cells -> {out.name}")
    _write_grid_manifest(preferred_meta)


def main(_api_keys=None, site_uids: list[str] | None = None, force: bool = False):
    """Build per-site rainfall parquets.

    Parameters
    ----------
    site_uids : list of str, optional
        If provided, only process these site UIDs.  Useful for targeted
        re-runs or testing individual sites without triggering the full
        pipeline.  Unrecognised UIDs are warned and skipped.  If None, all
        sites in preferred_basin.csv are candidates.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _RAIN_DIR.mkdir(parents=True, exist_ok=True)

    # The rain grid (Voronoi cells) is independent of the time series and feeds
    # the surplus/crop aggregates; build it first on its own staleness check.
    build_grids(site_uids=site_uids, force=force)

    preferred_meta = basins.get_metadata()
    stats = pd.read_csv(_STATS_FILE)

    if force:
        to_process = preferred_meta["site_uid"].tolist()
    else:
        to_process = _stale_sites(preferred_meta)

    if site_uids is not None:
        uid_set    = set(site_uids)
        known      = set(preferred_meta["site_uid"])
        for uid in sorted(uid_set - known):
            print(f"  [WARN] {uid}: not in preferred_basin.csv, skipping")
        to_process = [u for u in to_process if u in uid_set]
        print(f"Processing {len(to_process)} of {len(site_uids)} requested sites.")
    else:
        n_total = len(preferred_meta)
        n_skip  = n_total - len(to_process)
        print(f"Precheck: {n_skip}/{n_total} sites up to date, {len(to_process)} to build.")

    if not to_process:
        print("Nothing to do.")
        return

    # ── Phase 1: load basins and build date → [uid] index ─────────────────────

    basin_map  = {}  # uid -> GeoDataFrame
    frames     = {}  # uid -> list of per-day DataFrames (freed after write)
    end_dates  = {}  # uid -> last date needed (triggers write + free)
    date_to_uids = {}  # date -> [uid, ...]

    for uid in to_process:
        row = stats[stats["site_uid"] == uid]
        if row.empty:
            print(f"  [SKIP] {uid}: no entry in site_statistics.csv")
            continue

        try:
            basin = basins.get_basin(uid)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [SKIP] {uid}: {e}")
            continue

        frac = _iowa_coverage(basin)
        if frac < COVERAGE_THRESHOLD:
            print(f"  [SKIP] {uid}: {frac:.0%} of basin within IEM footprint "
                  f"(threshold {COVERAGE_THRESHOLD:.0%})")
            continue

        # start_date / last_date are full ISO strings with timezone offset,
        # e.g. "2012-05-30 21:40:00+00:00" — pd.to_datetime handles the tz
        start = pd.to_datetime(row.iloc[0]["start_date"]).date()
        end   = pd.to_datetime(row.iloc[0]["last_date"]).date()

        basin_map[uid]  = basin
        frames[uid]     = []
        end_dates[uid]  = end

        for d in _date_range(start, end):
            date_to_uids.setdefault(d, []).append(uid)

    # ── Phase 2: date-first download → parse → filter loop ────────────────────

    all_dates = sorted(date_to_uids)
    print(f"{len(all_dates)} unique dates across {len(basin_map)} sites.")

    for d in tqdm(all_dates, desc="days", unit="day"):
        zip_path = _download_shapefile(d)
        if zip_path is None:
            continue

        day_gdf = _parse_day_gdf(zip_path, d)
        if day_gdf is None:
            continue

        for uid in date_to_uids[d]:
            chunk = _basin_filter(day_gdf, basin_map[uid])
            if chunk is not None:
                frames[uid].append(chunk)

            # Write and free memory as soon as a site's last date is reached.
            if d == end_dates[uid] and frames[uid]:
                df = pd.concat(frames[uid], ignore_index=True)
                df["date"] = pd.to_datetime(df["date"])
                df = _add_time_features(df)
                out = _RAIN_DIR / f"{uid}_rain.parquet"
                df.to_parquet(out, index=False)
                print(f"\n  {uid}: {len(df):,} rows -> {out.name}")
                del frames[uid]
                del basin_map[uid]

    # ── Phase 3: write any sites not yet flushed ──────────────────────────────

    for uid, frame_list in frames.items():
        if not frame_list:
            continue
        df = pd.concat(frame_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = _add_time_features(df)
        out = _RAIN_DIR / f"{uid}_rain.parquet"
        df.to_parquet(out, index=False)
        print(f"  {uid}: {len(df):,} rows -> {out.name}")

    # ── Phase 4: update manifest ───────────────────────────────────────────────
    _write_manifest(preferred_meta)


if __name__ == "__main__":
    main()
