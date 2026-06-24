"""
Per-site drainage-basin rainfall time series builder.

Source : IEM (Iowa Environmental Mesonet) daily precipitation grid polygons
         https://mesonet.agron.iastate.edu/rainfall/

For each monitoring site that has a preferred basin (data/basins/basin_data/),
this script downloads IEM daily precipitation polygons covering the site's
active date range, selects the grid cells that fall in the site's drainage
basin, and writes one row per (grid cell, day) to a parquet file at
data/rain/rain_data/<uid>_rain.parquet.

Output schema
-------------
date            datetime64[ms]   calendar date of the observation
node_id         int64            rain-grid node this cell maps to (basin-local join key)
global_node_id  int64            canonical IEM cell index (shared across basins)
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
A date -> [uid] index is built upfront.  For each unique calendar day the IEM
zip is downloaded and parsed exactly once, then each active site's cells are
selected by plain array indexing.  This avoids re-parsing the same zip N_sites
times when date ranges overlap heavily.

The per-day basin filter is *not* a spatial join.  The IEM grid is static — the
daily shapefile holds the same cells in the same record order every day — so a
day's RAINFALL array is positionally aligned to a reference day.  _build_cell_index
matches each site's rain-grid nodes to their reference-cell row once (a single
KDTree query per site); per day the filter is then `values[rows]`, which also
hands each row its node_id for free.  Because the grid nodes are exactly the IEM
cells intersecting the basin, this yields the same cells the old gpd.sjoin did.

Memory management: each site's accumulated rows are written to disk and freed
as soon as its last required date has been processed.  At most one site's full
date range resides in memory at once (in the worst case where all sites share
the same last date).

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
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from pyproj import Geod
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, cKDTree
from tqdm import tqdm

# ── paths (all relative to this script's directory: data/rain/) ───────────────
_THIS_DIR = Path(__file__).resolve().parent
_STATS_FILE = _THIS_DIR.parent / "water" / "water_meta" / "site_statistics.csv"
_RAW_DIR = _THIS_DIR / "rain_raw"  # cached IEM daily zips, shared across sites
_RAIN_DIR = _THIS_DIR / "rain_data"  # output: one parquet per site
_GRID_DIR = _THIS_DIR / "rain_grid"  # output: one Voronoi-cell geoparquet per site
_MANIFEST_FILE = _RAIN_DIR / ".basin_manifest.csv"
_GRID_MANIFEST_FILE = _GRID_DIR / ".basin_manifest.csv"
_GLOBAL_GRID_FILE = _GRID_DIR / "global_rain_grid.parquet"  # cell -> sites containing it

# Geometry of the rain grid only — a single IEM day supplies the cell polygons.
IEM_GRID_DATE = date(2018, 6, 15)

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import basins
from data.settings import get_config, get_equal_area_crs

_ALBERS = get_equal_area_crs()  # equal-area CRS for the Voronoi/area math (EPSG:5070)

# ── IEM API ───────────────────────────────────────────────────────────────────
# polygon geometry gives actual grid-cell extents, used once to build the
# Voronoi rain grid (build_rain_grid); per-day rainfall is read geometry-free
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


def _parse_day_values(zip_path: Path, expected_n: int) -> np.ndarray | None:
    """Read just the RAINFALL column (inches) from an IEM daily zip, in record order.

    The IEM grid is static — every daily shapefile holds the same cells in the
    same record order — so the returned array is positionally aligned to the
    reference grid used by _build_cell_index, and geometry can be skipped
    entirely (~5x faster than parsing polygons).  Returns None on a parse error,
    a missing-data day, or if the cell count does not match the reference (which
    would break the positional alignment).
    """
    try:
        df = gpd.read_file(f"/vsizip/{zip_path}", ignore_geometry=True)
    except Exception as e:
        print(f"  [WARN] parse {zip_path.name}: {e}")
        return None
    if df is None or len(df) == 0:
        return None
    df.columns = [c.lower() for c in df.columns]
    if "rainfall" not in df.columns:
        print(f"  [WARN] unexpected columns in {zip_path.name}: {list(df.columns)}")
        return None
    if len(df) != expected_n:
        print(f"  [WARN] {zip_path.name}: {len(df)} cells != reference {expected_n}; skipping day")
        return None
    return df["rainfall"].to_numpy()


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
        basin_changed = manifest.get(uid) != row["basin_name"]
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

    Returns a GeoDataFrame with columns node_id, global_node_id, x, y (EPSG:5070),
    lat, lon, cell_area, dist_to_sensor, frac_cell_in_basin, geometry. Cells are
    the IEM nodes whose cell intersects the basin; each is bounded by real
    neighbours via a padded halo so edge cells stay finite (infinite hull cells
    are dropped). node_id is the basin-local index (0..N-1) that keys the surplus
    and crop aggregates downstream; global_node_id is the cell's row index in the
    canonical IEM grid — identical across basins, so overlapping basins share it
    for shared cells. dist_to_sensor is the metres-of-flow from the node centre to
    the monitoring sensor (see dist_to_sensor / _grid_node_distances);
    frac_cell_in_basin is the fraction of the cell's area inside the basin, in
    [0, 1] (see _grid_basin_fractions).
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
    # Keep the original IEM-grid row index — a global, basin-independent cell id.
    px0, py0, px1, py1 = minx - pad, miny - pad, maxx + pad, maxy + pad
    halo = grid[grid["x"].between(px0, px1) & grid["y"].between(py0, py1)].reset_index(names="global_node_id")
    polys = _finite_voronoi(halo[["x", "y"]].to_numpy())

    rows = []
    for i in halo.index[halo["is_target"]]:
        if i not in polys:
            print(f"  [warn] {site_uid}: a target node has an infinite cell (pad too small) — skipped")
            continue
        rows.append(
            {
                "node_id": len(rows),
                "global_node_id": int(halo.at[i, "global_node_id"]),
                "x": halo.at[i, "x"],
                "y": halo.at[i, "y"],
                "lat": halo.at[i, "lat"],
                "lon": halo.at[i, "lon"],
                "cell_area": polys[i].area,
                "geometry": polys[i],
            }
        )
    grid = gpd.GeoDataFrame(rows, crs=_ALBERS)

    # Per-node columns computed here from the in-memory grid, since the parquet
    # does not exist yet: flow distance (metres) to the monitoring sensor, and
    # the fraction of each cell's area falling inside the basin.
    if len(grid):
        try:
            grid["dist_to_sensor"] = _grid_node_distances(site_uid, grid)
        except Exception as e:
            print(f"  [warn] {site_uid}: dist_to_sensor unavailable ({e}); column set to NaN")
            grid["dist_to_sensor"] = np.nan
        grid["frac_cell_in_basin"] = _grid_basin_fractions(site_uid, grid)
    return grid


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


def build_global_grid() -> pd.DataFrame | None:
    """Build the global rain grid: every cell that falls in at least one
    preferred-basin rain grid, with the sites whose grid contains it.

    Inverts the per-site rain grids (which key on global_node_id, the canonical
    IEM cell index shared across basins). Written to global_rain_grid.parquet.

    Columns
    -------
    global_node_id      int64        canonical IEM cell index
    contained_in_sites  list[str]    sites whose rain grid includes this cell
    n_sites             int64        len(contained_in_sites)
    lat, lon            float64      cell centroid (WGS84), as in rain_grid

    Only cells contained in some basin appear. Requires rain grids built with a
    global_node_id column; raises KeyError if a grid predates it.
    """
    preferred_meta = basins.get_metadata()
    site_list = preferred_meta["site_uid"].tolist()

    frames = []
    for uid in site_list:
        path = _GRID_DIR / f"{uid}_rain_grid.parquet"
        if not path.exists():
            continue
        g = gpd.read_parquet(path)
        if "global_node_id" not in g.columns:
            raise KeyError(
                f"{path.name} has no global_node_id column — rebuild the rain grids "
                "(build_grids(force=True)) before building the global grid."
            )
        frames.append(
            pd.DataFrame(
                {
                    "global_node_id": g["global_node_id"].to_numpy(),
                    "site_uid": uid,
                    "lat": g["lat"].to_numpy(),
                    "lon": g["lon"].to_numpy(),
                }
            )
        )

    if not frames:
        print("No rain grids found — run build_grids first; skipping global grid.")
        return None

    cells = pd.concat(frames, ignore_index=True)
    # lat/lon are identical across basins for a given cell, so "first" is exact.
    out = (
        cells.groupby("global_node_id")
        .agg(
            contained_in_sites=("site_uid", lambda s: sorted(s.unique())),
            n_sites=("site_uid", "nunique"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
        .sort_values("global_node_id", ignore_index=True)
    )
    out.to_parquet(_GLOBAL_GRID_FILE, index=False)
    print(f"  global rain grid: {len(out):,} cells across {len(frames)} sites -> {_GLOBAL_GRID_FILE.name}")
    return out


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
        if not _GLOBAL_GRID_FILE.exists():
            build_global_grid()
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
    build_global_grid()  # refresh the cell -> sites inverse map


# ── Flow distance: rain-grid node → sensor, along the D8 drainage network ──────
# "As water flows" distance, reusing the 500 m IWQIS D8 flow-direction raster
# that make_basins uses for basin3.  Each raster cell stores its downslope
# neighbour as a numeric-keypad direction (7 8 9 / 4 5 6 / 1 2 3; 5 = sink,
# 0 = nodata), so every cell drains to exactly one neighbour and the cells that
# flow to a given outlet form a tree rooted there.  Walking *up* that tree once
# (following inflow edges, exactly as basins._bfs does) yields the flow distance
# from every upstream cell to the outlet in a single pass; we cache the field
# per site and sample it at each node centre.
#
# Outlet placement is the subtle part.  A registered sensor (lat, lon) routinely
# sits a cell or two off the digitised main stem — on a minor tributary or a
# nodata cell — which would root the tree on a tiny sub-catchment.  basins'
# inflow-count snap is too weak (it picks any locally well-connected cell).  We
# instead snap to the nearest cell carrying main-stem *flow accumulation*
# (upstream drainage area), which reliably lands on the channel within ~1 km of
# the sensor.  Distance is then measured to that outlet; the residual sensor↔
# outlet offset (typically < 1.5 km) is negligible against basin-scale paths.

_GEOD = Geod(ellps="WGS84")
_FLOW_FIELD_CACHE: dict[str, np.ndarray] = {}  # site_uid -> (_H, _W) metres-to-outlet
_NODE_DIST_CACHE: dict[str, pd.Series] = {}  # site_uid -> node_id -> metres-to-sensor
_GRID_CACHE: dict[str, gpd.GeoDataFrame] = {}  # site_uid -> rain grid (node_id-indexed)
_ACCUM: np.ndarray | None = None  # (_H, _W) upstream-cell count (grid-wide)

# Keypad direction code -> downstream (dcol, drow). The inverse of the inflow
# offsets in basins._NEIGHBOR_CHECKS: a cell with code c flows to this neighbour.
_D8_STEP = {7: (-1, -1), 8: (0, -1), 9: (1, -1), 4: (-1, 0), 6: (1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1)}

_OUTLET_SNAP_RADIUS = 10  # cells (~5 km) searched for the main-stem outlet cell
_OUTLET_ACC_FRAC = 0.5  # min fraction of the window-max accumulation to count as main stem
_NODE_SNAP_RADIUS = 4  # cells (~2 km) to recover a node straddling the basin divide


def _sensor_lonlat(site_uid: str) -> tuple[float, float]:
    """(lon, lat) of the monitoring sensor for site_uid, from water metadata.

    Reads site_location_metadata.csv (via the water access layer); the row's
    latitude/longitude is the sensor position for that site.
    """
    from data import water

    row = water.get_metadata().query("site_uid == @site_uid")
    if row.empty:
        raise KeyError(f"No location metadata for {site_uid}.")
    return float(row.iloc[0]["longitude"]), float(row.iloc[0]["latitude"])


def _get_grid(site_uid: str) -> gpd.GeoDataFrame:
    """Rain grid for site_uid, indexed by node_id (cached)."""
    grid = _GRID_CACHE.get(site_uid)
    if grid is None:
        path = _GRID_DIR / f"{site_uid}_rain_grid.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No rain grid for {site_uid}. Run build_grids to generate {path.name}.")
        grid = gpd.read_parquet(path).set_index("node_id")
        _GRID_CACHE[site_uid] = grid
    return grid


def _flow_accumulation(direction: np.ndarray, mb) -> np.ndarray:
    """Upstream-cell count for every D8 cell (computed once, cached grid-wide).

    Kahn topological sweep: each cell flows to exactly one downstream neighbour,
    so the raster is a forest. Starting from sources (no inflow) and pushing
    counts downstream gives each cell's total upstream area in O(N).
    """
    global _ACCUM
    if _ACCUM is not None:
        return _ACCUM

    H, W = mb._H, mb._W
    down = np.full((H, W, 2), -1, np.int32)
    indeg = np.zeros((H, W), np.int32)
    for code, (dc, dr) in _D8_STEP.items():
        rs, cs = np.where(direction == code)
        nc, nr = cs + dc, rs + dr
        ok = (nc >= 0) & (nc < W) & (nr >= 0) & (nr < H)
        down[rs[ok], cs[ok], 0] = nc[ok]
        down[rs[ok], cs[ok], 1] = nr[ok]
        np.add.at(indeg, (nr[ok], nc[ok]), 1)

    acc = np.ones((H, W), np.int64)
    q = deque(zip(*np.where(indeg == 0)))
    while q:
        r, c = q.popleft()
        dc, dr = down[r, c]
        if dc < 0:
            continue
        acc[dr, dc] += acc[r, c]
        indeg[dr, dc] -= 1
        if indeg[dr, dc] == 0:
            q.append((dr, dc))

    _ACCUM = acc
    return acc


def _snap_outlet(direction: np.ndarray, col: int, row: int, mb) -> tuple[int, int]:
    """Snap a sensor pixel to the nearest main-stem cell (pour-point snapping).

    Within a small window, take the cells whose flow accumulation is at least
    _OUTLET_ACC_FRAC of the window maximum (i.e. on the main channel) and return
    the one closest to the sensor — landing on the channel without overshooting
    downstream onto extra drainage area below the sensor.
    """
    H, W = mb._H, mb._W
    acc = _flow_accumulation(direction, mb)
    R = _OUTLET_SNAP_RADIUS
    c0, c1 = max(0, col - R), min(W, col + R + 1)
    r0, r1 = max(0, row - R), min(H, row + R + 1)
    sub = acc[r0:r1, c0:c1]
    rs, cs = np.where(sub >= _OUTLET_ACC_FRAC * sub.max())
    cc, rr = cs + c0, rs + r0
    j = int(np.argmin((cc - col) ** 2 + (rr - row) ** 2))
    return int(cc[j]), int(rr[j])


def _build_flow_field(direction: np.ndarray, col: int, row: int, mb) -> np.ndarray:
    """Metres-to-outlet for every cell draining to (col, row) on the D8 grid.

    Breadth-first walk up the inflow tree from the outlet, accumulating the
    geodesic distance between successive cell centres.  Cells that do not drain
    to the outlet stay NaN.  Diagonal vs. cardinal steps are handled implicitly
    by measuring the true centre-to-centre distance of each move.
    """
    H, W, transform = mb._H, mb._W, mb._TRANSFORM
    dist = np.full((H, W), np.nan)
    dist[row, col] = 0.0
    q = deque([(col, row)])
    while q:
        cx, cy = q.popleft()
        clon, clat = transform * (cx + 0.5, cy + 0.5)  # current cell centre
        base = dist[cy, cx]
        for dx, dy, expected in mb._NEIGHBOR_CHECKS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and np.isnan(dist[ny, nx]) and direction[ny, nx] == expected:
                nlon, nlat = transform * (nx + 0.5, ny + 0.5)
                _, _, seg = _GEOD.inv(clon, clat, nlon, nlat)
                dist[ny, nx] = base + seg
                q.append((nx, ny))
    return dist


def _flow_distance_field(site_uid: str) -> np.ndarray:
    """Per-cell flow distance (metres) to the sensor's outlet (cached per site)."""
    if site_uid in _FLOW_FIELD_CACHE:
        return _FLOW_FIELD_CACHE[site_uid]

    from data.basins import make_basins as mb

    direction = mb._load_direction_array()
    lon, lat = _sensor_lonlat(site_uid)
    col, row = mb._ll_to_image_pixel(lat, lon)
    if not (0 <= col < mb._W and 0 <= row < mb._H):
        raise ValueError(f"{site_uid}: sensor falls outside the D8 raster extent.")

    ocol, orow = _snap_outlet(direction, col, row, mb)
    field = _build_flow_field(direction, ocol, orow, mb)
    _FLOW_FIELD_CACHE[site_uid] = field
    return field


def _sample_field(field: np.ndarray, col: int, row: int, mb) -> float:
    """Flow distance at a node pixel, recovering basin-divide straddlers.

    If the node centre falls outside the D8 catchment (coarse 500 m basin vs.
    finer NLDI basin), snap to the nearest in-catchment cell within a small
    radius — the node's ~4 km cell still overlaps the catchment.  Returns NaN
    only when the node sits well outside any drained cell.
    """
    H, W = mb._H, mb._W
    if 0 <= col < W and 0 <= row < H and np.isfinite(field[row, col]):
        return float(field[row, col])

    R = _NODE_SNAP_RADIUS
    c0, c1 = max(0, col - R), min(W, col + R + 1)
    r0, r1 = max(0, row - R), min(H, row + R + 1)
    sub = field[r0:r1, c0:c1]
    fin = np.isfinite(sub)
    if not fin.any():
        return float("nan")
    rs, cs = np.where(fin)
    j = int(np.argmin((cs + c0 - col) ** 2 + (rs + r0 - row) ** 2))
    return float(sub[rs[j], cs[j]])


def _grid_node_distances(site_uid: str, grid: gpd.GeoDataFrame) -> np.ndarray:
    """Flow distance (metres) to the sensor for each row of an in-memory grid.

    `grid` must carry the build_rain_grid columns lat, lon, x, y; the result is
    an array aligned to its row order.  Two passes:
      1. Route each node through the D8 distance field (with the per-cell
         straddler recovery in _sample_field).
      2. Any node still NaN — its 500 m cell drains to a neighbouring outlet on
         the coarse raster — is filled from the nearest resolved node B:
             dist(A) = dist(B) + |centre(A) - centre(B)|
         where the second term is the straight-line node-centre distance in
         EPSG:5070 metres.  This guarantees a finite, monotone-ish estimate for
         divide-straddling edge nodes the raster cannot route directly.
    """
    from data.basins import make_basins as mb

    field = _flow_distance_field(site_uid)
    dist = np.array(
        [
            _sample_field(field, *mb._ll_to_image_pixel(lat, lon), mb)
            for lat, lon in zip(grid["lat"].to_numpy(), grid["lon"].to_numpy())
        ]
    )

    nan = np.isnan(dist)
    if nan.any() and (~nan).any():
        xy = grid[["x", "y"]].to_numpy()  # EPSG:5070 metres
        gap, idx = cKDTree(xy[~nan]).query(xy[nan])
        dist[nan] = dist[~nan][idx] + gap

    return dist


def _grid_basin_fractions(site_uid: str, grid: gpd.GeoDataFrame) -> np.ndarray:
    """Fraction of each cell's area that lies inside the basin, in [0, 1].

    `grid` must carry Voronoi cell geometry in the equal-area CRS (_ALBERS), as
    build_rain_grid produces — so the ratio of intersection area to cell area is
    an unbiased areal fraction.  Interior cells return ~1.0; cells straddling the
    basin divide return a partial fraction.  Result is aligned to grid row order.
    """
    cells = grid.geometry
    if cells.crs is not None and cells.crs != _ALBERS:
        cells = cells.to_crs(_ALBERS)
    basin_poly = basins.get_basin(site_uid).to_crs(_ALBERS).geometry.union_all()
    inside = cells.intersection(basin_poly).area
    return np.clip((inside / cells.area).to_numpy(), 0.0, 1.0)


def _node_distances(site_uid: str) -> pd.Series:
    """Per-node flow distance to the sensor, node_id-indexed (cached per site).

    Reads the saved rain grid; for building it from an in-memory grid use
    _grid_node_distances directly.
    """
    if site_uid in _NODE_DIST_CACHE:
        return _NODE_DIST_CACHE[site_uid]

    grid = _get_grid(site_uid)
    series = pd.Series(_grid_node_distances(site_uid, grid), index=grid.index, name="dist_to_sensor")
    _NODE_DIST_CACHE[site_uid] = series
    return series


def dist_to_sensor(site_uid: str, node_id: int) -> float:
    """Flow distance in metres from a rain-grid node centre to the sensor.

    Distance is measured "as water flows": down the 500 m D8 drainage network
    from the cell containing the node centre to the sensor's outlet.  Every node
    in a site's basin is upstream of the sensor by construction, so the routed
    distance is always >= the straight-line distance.  Nodes the coarse raster
    cannot route are filled from the nearest resolved node (see _node_distances),
    so a value is returned for every node unless the whole site fails to route.

    Parameters
    ----------
    site_uid : str
        Monitoring site identifier (must have a rain grid; run build_grids).
    node_id : int
        Node identifier from the site's rain grid (data/rain/rain_grid/).
    """
    series = _node_distances(site_uid)
    if node_id not in series.index:
        raise KeyError(f"{site_uid}: no node_id {node_id} in rain grid.")
    return float(series.loc[node_id])


def _build_cell_index(site_uids: list[str]) -> tuple[dict, int]:
    """Map each site's rain-grid nodes to IEM reference-cell row indices.

    The IEM daily shapefile is identical in cell count and record order every
    day, so a day's RAINFALL array (from _parse_day_values) is positionally
    aligned to a reference day's cells.  For each site we match its grid-node
    centroids to the nearest reference cell once; per day the basin filter then
    reduces to `values[rows]` rather than a geometric sjoin, and node_id comes
    along for free.

    Returns (index, n_ref) where index[uid] = {rows, node_id, lon, lat} (arrays
    aligned to node order) and n_ref is the reference cell count used to validate
    each day's parse.
    """
    ref = _parse_day_gdf(_download_shapefile(IEM_GRID_DATE), IEM_GRID_DATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_xy = np.column_stack([ref.geometry.centroid.x.to_numpy(), ref.geometry.centroid.y.to_numpy()])
    tree = cKDTree(ref_xy)

    index = {}
    for uid in site_uids:
        grid = _get_grid(uid)
        lon, lat = grid["lon"].to_numpy(), grid["lat"].to_numpy()
        dist, rows = tree.query(np.column_stack([lon, lat]))
        if dist.size and dist.max() > 0.005:  # ~500 m: a node should land on its own IEM cell
            print(f"  [WARN] {uid}: max node→cell match {dist.max():.4f}° — grid/IEM mismatch?")
        # global_node_id = the canonical IEM row index; equals `rows` by construction,
        # but read it from the grid when present so the timeseries matches the grid file.
        gid = grid["global_node_id"].to_numpy() if "global_node_id" in grid.columns else rows
        index[uid] = {"rows": rows, "node_id": grid.index.to_numpy(), "global_node_id": gid, "lon": lon, "lat": lat}
    return index, len(ref)


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
        uid_set = set(site_uids)
        known = set(preferred_meta["site_uid"])
        for uid in sorted(uid_set - known):
            print(f"  [WARN] {uid}: not in preferred_basin.csv, skipping")
        to_process = [u for u in to_process if u in uid_set]
        print(f"Processing {len(to_process)} of {len(site_uids)} requested sites.")
    else:
        n_total = len(preferred_meta)
        n_skip = n_total - len(to_process)
        print(f"Precheck: {n_skip}/{n_total} sites up to date, {len(to_process)} to build.")

    if not to_process:
        print("Nothing to do.")
        return

    # ── Phase 1: coverage check + date → [uid] index ──────────────────────────

    kept = []  # uids that passed the coverage check
    end_dates = {}  # uid -> last date needed (triggers write + free)
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
            print(f"  [SKIP] {uid}: {frac:.0%} of basin within IEM footprint " f"(threshold {COVERAGE_THRESHOLD:.0%})")
            continue

        # start_date / last_date are full ISO strings with timezone offset,
        # e.g. "2012-05-30 21:40:00+00:00" — pd.to_datetime handles the tz
        start = pd.to_datetime(row.iloc[0]["start_date"]).date()
        end = pd.to_datetime(row.iloc[0]["last_date"]).date()

        kept.append(uid)
        end_dates[uid] = end
        for d in _date_range(start, end):
            date_to_uids.setdefault(d, []).append(uid)

    if not kept:
        print("No sites passed the coverage check.")
        return

    # Precompute the per-site IEM-cell mapping — replaces the per-(day, site)
    # spatial join with array indexing.
    cell_index, n_ref = _build_cell_index(kept)
    vals_accum = {uid: [] for uid in kept}  # uid -> list of per-day precip arrays
    date_accum = {uid: [] for uid in kept}  # uid -> list of dates (parallel)

    def _flush(uid: str) -> None:
        """Assemble a site's accumulated days into a parquet and free its memory."""
        n_days = len(date_accum[uid])
        if not n_days:
            return
        ci = cell_index[uid]
        n_nodes = len(ci["node_id"])
        df = pd.DataFrame(
            {
                "date": np.repeat(pd.to_datetime(date_accum[uid]).values, n_nodes),
                "lon": np.tile(ci["lon"], n_days),
                "lat": np.tile(ci["lat"], n_days),
                "precip_in_1d": np.concatenate(vals_accum[uid]),
                "node_id": np.tile(ci["node_id"], n_days),
                "global_node_id": np.tile(ci["global_node_id"], n_days),
            }
        )
        df = _add_time_features(df)
        df["date"] = df["date"].astype("datetime64[ms]")
        df = df[
            ["date", "node_id", "global_node_id", "lon", "lat", "precip_in_1d", "year", "month", "day_of_year", "week"]
        ]
        out = _RAIN_DIR / f"{uid}_rain.parquet"
        df.to_parquet(out, index=False)
        print(f"\n  {uid}: {len(df):,} rows -> {out.name}")
        vals_accum[uid] = []
        date_accum[uid] = []

    # ── Phase 2: date-first download → parse → index loop ─────────────────────

    all_dates = sorted(date_to_uids)
    print(f"{len(all_dates)} unique dates across {len(kept)} sites.")

    for d in tqdm(all_dates, desc="days", unit="day"):
        zip_path = _download_shapefile(d)
        if zip_path is None:
            continue

        vals = _parse_day_values(zip_path, n_ref)
        if vals is None:
            continue

        for uid in date_to_uids[d]:
            vals_accum[uid].append(vals[cell_index[uid]["rows"]])
            date_accum[uid].append(d)
            # Write and free memory as soon as a site's last date is reached.
            if d == end_dates[uid]:
                _flush(uid)

    # ── Phase 3: write any sites not yet flushed (e.g. last date had no data) ──

    for uid in kept:
        _flush(uid)

    # ── Phase 4: update manifest ───────────────────────────────────────────────
    _write_manifest(preferred_meta)


if __name__ == "__main__":
    main()
