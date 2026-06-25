"""Build the shared spatial grid for the weather pipeline.

Ported from rain/make_rain.py (grid half). Produces, under weather_grid/:

    {site_uid}_grid.parquet   per-site Voronoi target cells, columns
        node_id            int64        basin-local index (0..N-1), join key for
                                        the surplus/crop aggregates
        global_node_id     int64        canonical IEM cell index, shared across
                                        basins (overlapping basins share it)
        x, y               float64      cell centroid, EPSG:5070 (metres)
        lat, lon           float64      cell centroid, WGS84
        cell_area          float64      Voronoi cell area, m^2 (EPSG:5070)
        dist_to_sensor     float64      metres of flow from the cell to the sensor
        frac_cell_in_basin float64      fraction of the cell inside the basin [0,1]
        geometry           polygon      Voronoi cell, EPSG:5070

    global_grid.parquet       inverse map: every cell in >=1 basin -> the sites
        global_node_id, contained_in_sites (list[str]), n_sites, lat, lon

Cells come from the static IEM ~4 km precipitation grid (a single reference day
supplies the polygons); each basin keeps the IEM cells whose polygon intersects
it. The IEM reference-day zip is cached in weather_raw/.

Usage
-----
    python make_grid.py            # build stale grids, skip existing
    python make_grid.py --force    # rebuild all
"""

import sys
import argparse
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import deque
from datetime import date
from pathlib import Path
from pyproj import Geod
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, cKDTree

# ── paths (relative to this script's directory: data/weather/) ────────────────
_THIS_DIR = Path(__file__).resolve().parent
_GRID_DIR = _THIS_DIR / "weather_grid"  # output: per-site grids + global grid
_RAW_DIR = _THIS_DIR / "weather_raw" / "IEM_raw"  # cached IEM daily zips
_GLOBAL_GRID_FILE = _GRID_DIR / "global_grid.parquet"
_GRID_MANIFEST_FILE = _GRID_DIR / ".basin_manifest.csv"

# Geometry of the grid only — a single IEM day supplies the cell polygons.
IEM_GRID_DATE = date(2018, 6, 15)

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import basins
from data.settings import get_equal_area_crs

_ALBERS = get_equal_area_crs()  # equal-area CRS for the Voronoi/area math (EPSG:5070)

# ── IEM API (polygon geometry gives the cell extents used to build the grid) ───
_SHP_URL = (
    "https://mesonet.agron.iastate.edu/rainfall/dshape.php?"
    "month={month}&day={day}&year={year}"
    "&geometry=polygon"
    "&duration=day"
    "&epsg=4326"
)


# ── IEM reference-day geometry ────────────────────────────────────────────────


def _download_shapefile(d: date) -> Path | None:
    """Fetch the IEM daily precipitation shapefile for date d into weather_raw/.

    Returns the local zip path, or None on failure. Skips the network request if
    the file is already cached (here, or in the sibling rain/rain_raw/ cache, so
    a prior rain build's downloads are reused).
    """
    path = _RAW_DIR / f"iowa_rain_{d.isoformat()}.zip"
    if path.exists():
        return path

    url = _SHP_URL.format(month=d.month, day=d.day, year=d.year)
    try:
        resp = requests.get(url, timeout=(10, 60))
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):  # IEM serves HTML on no-data days
            return None
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return path
    except Exception as e:
        print(f"  [WARN] {d}: {e}")
        return None


def _parse_day_gdf(zip_path: Path, d: date) -> gpd.GeoDataFrame | None:
    """Parse an IEM daily zip into precipitation polygons (EPSG:4326).

    Columns: date, precip_in_1d, geometry. The API serves EPSG:4269 (NAD83)
    despite epsg=4326; the datums are sub-metre apart so we re-label.
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
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.rename(columns={"rainfall": "precip_in_1d"})
    gdf["date"] = d
    return gdf[["date", "precip_in_1d", "geometry"]].copy()


# ── Voronoi grid ──────────────────────────────────────────────────────────────


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


def build_grid(site_uid: str) -> gpd.GeoDataFrame:
    """Build the Voronoi target-cell grid for one basin (see module docstring)."""
    basin = basins.get_basin(site_uid).to_crs(_ALBERS)
    basin_poly = basin.geometry.union_all()
    minx, miny, maxx, maxy = basin.total_bounds

    # A single IEM day supplies the cell polygons; centroids give the node coords.
    day = _parse_day_gdf(_download_shapefile(IEM_GRID_DATE), IEM_GRID_DATE).to_crs(_ALBERS)
    centroids = day.geometry.centroid
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
        all_xy = grid[["x", "y"]].to_numpy()
        pad = 2 * float(np.median(cKDTree(all_xy).query(all_xy, k=2)[0][:, 1]))
    else:
        pad = 2 * float(cKDTree(tgt_xy).query(tgt_xy, k=2)[0][:, 1].max())

    # Halo = every node within the padded bbox; keep the original IEM row index
    # (global_node_id) so cells line up across basins.
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

    # Per-node columns from the in-memory grid (no parquet yet): flow distance to
    # the sensor and the fraction of each cell inside the basin.
    if len(grid):
        try:
            grid["dist_to_sensor"] = _grid_node_distances(site_uid, grid)
        except Exception as e:
            print(f"  [warn] {site_uid}: dist_to_sensor unavailable ({e}); column set to NaN")
            grid["dist_to_sensor"] = np.nan
        grid["frac_cell_in_basin"] = _grid_basin_fractions(site_uid, grid)
    return grid


# ── Flow distance: grid node → sensor, along the D8 drainage network ───────────
# Reuses the 500 m IWQIS D8 flow-direction raster that make_basins uses for
# basin3. Each cell stores its downslope neighbour as a numeric-keypad direction
# (7 8 9 / 4 5 6 / 1 2 3; 5 = sink, 0 = nodata), so the cells flowing to a given
# outlet form a tree; walking up that tree from the sensor's pour point gives the
# flow distance to every upstream cell in one pass. The sensor pixel is snapped
# to the nearest main-stem cell by flow accumulation before routing.

_GEOD = Geod(ellps="WGS84")
_FLOW_FIELD_CACHE: dict[str, np.ndarray] = {}  # site_uid -> (_H, _W) metres-to-outlet
_ACCUM: np.ndarray | None = None  # (_H, _W) upstream-cell count (grid-wide)

# Keypad direction code -> downstream (dcol, drow); inverse of basins._NEIGHBOR_CHECKS.
_D8_STEP = {7: (-1, -1), 8: (0, -1), 9: (1, -1), 4: (-1, 0), 6: (1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1)}

_OUTLET_SNAP_RADIUS = 10  # cells (~5 km) searched for the main-stem outlet cell
_OUTLET_ACC_FRAC = 0.5  # min fraction of window-max accumulation to count as main stem
_NODE_SNAP_RADIUS = 4  # cells (~2 km) to recover a node straddling the basin divide


def _sensor_lonlat(site_uid: str) -> tuple[float, float]:
    """(lon, lat) of the monitoring sensor for site_uid, from water metadata."""
    from data import water

    row = water.get_metadata().query("site_uid == @site_uid")
    if row.empty:
        raise KeyError(f"No location metadata for {site_uid}.")
    return float(row.iloc[0]["longitude"]), float(row.iloc[0]["latitude"])


def _flow_accumulation(direction: np.ndarray, mb) -> np.ndarray:
    """Upstream-cell count for every D8 cell (computed once, cached grid-wide)."""
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
    """Snap a sensor pixel to the nearest main-stem cell (pour-point snapping)."""
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
    """Metres-to-outlet for every cell draining to (col, row) on the D8 grid."""
    H, W, transform = mb._H, mb._W, mb._TRANSFORM
    dist = np.full((H, W), np.nan)
    dist[row, col] = 0.0
    q = deque([(col, row)])
    while q:
        cx, cy = q.popleft()
        clon, clat = transform * (cx + 0.5, cy + 0.5)
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
    """Flow distance at a node pixel, recovering basin-divide straddlers."""
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

    Pass 1 routes each node through the D8 distance field (with the per-cell
    straddler recovery in _sample_field). Pass 2 fills any node still NaN from
    the nearest resolved node B: dist(A) = dist(B) + |centre(A) - centre(B)|
    (straight-line node-centre distance, EPSG:5070 metres).
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
        xy = grid[["x", "y"]].to_numpy()
        gap, idx = cKDTree(xy[~nan]).query(xy[nan])
        dist[nan] = dist[~nan][idx] + gap

    return dist


def _grid_basin_fractions(site_uid: str, grid: gpd.GeoDataFrame) -> np.ndarray:
    """Fraction of each cell's area inside the basin, in [0, 1] (row-aligned)."""
    cells = grid.geometry
    if cells.crs is not None and cells.crs != _ALBERS:
        cells = cells.to_crs(_ALBERS)
    basin_poly = basins.get_basin(site_uid).to_crs(_ALBERS).geometry.union_all()
    inside = cells.intersection(basin_poly).area
    return np.clip((inside / cells.area).to_numpy(), 0.0, 1.0)


# ── global grid + orchestration ───────────────────────────────────────────────


def _grid_stale_sites(preferred_meta: pd.DataFrame) -> list[str]:
    """UIDs whose grid is missing or whose basin changed since last build."""
    manifest = {}
    if _GRID_MANIFEST_FILE.exists():
        mdf = pd.read_csv(_GRID_MANIFEST_FILE)
        manifest = dict(zip(mdf["site_uid"], mdf["basin_name"]))
    return [
        row["site_uid"]
        for _, row in preferred_meta.iterrows()
        if not (_GRID_DIR / f"{row['site_uid']}_grid.parquet").exists()
        or manifest.get(row["site_uid"]) != row["basin_name"]
    ]


def _write_grid_manifest(preferred_meta: pd.DataFrame) -> None:
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if (_GRID_DIR / f"{row['site_uid']}_grid.parquet").exists()
    ]
    pd.DataFrame(rows).to_csv(_GRID_MANIFEST_FILE, index=False)


def build_global_grid() -> pd.DataFrame | None:
    """Build global_grid.parquet: every cell in >=1 basin -> the sites containing it.

    Inverts the per-site grids (keyed on global_node_id). Columns: global_node_id,
    contained_in_sites (list[str]), n_sites, lat, lon. Only cells in some basin
    appear.
    """
    preferred_meta = basins.get_metadata()
    frames = []
    for uid in preferred_meta["site_uid"].tolist():
        path = _GRID_DIR / f"{uid}_grid.parquet"
        if not path.exists():
            continue
        g = gpd.read_parquet(path)
        if "global_node_id" not in g.columns:
            raise KeyError(f"{path.name} has no global_node_id — rebuild grids (build_grids(force=True)).")
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
        print("No grids found — run build_grids first; skipping global grid.")
        return None

    cells = pd.concat(frames, ignore_index=True)
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
    print(f"  global grid: {len(out):,} cells across {len(frames)} sites -> {_GLOBAL_GRID_FILE.name}")
    return out


def build_grids(site_uids: list[str] | None = None, force: bool = False) -> None:
    """Build/refresh the per-site grids + the global grid. Depends only on basin
    geometry and the IEM grid."""
    _GRID_DIR.mkdir(parents=True, exist_ok=True)
    preferred_meta = basins.get_metadata()

    to_process = preferred_meta["site_uid"].tolist() if force else _grid_stale_sites(preferred_meta)
    if site_uids is not None:
        to_process = [u for u in to_process if u in set(site_uids)]

    if not to_process:
        print("Grids up to date.")
        if not _GLOBAL_GRID_FILE.exists():
            build_global_grid()
        return

    print(f"Building grids for {len(to_process)} site(s)...")
    for uid in to_process:
        try:
            grid = build_grid(uid)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [SKIP] {uid}: {e}")
            continue
        if grid.empty:
            print(f"  [SKIP] {uid}: no cells intersect basin")
            continue
        out = _GRID_DIR / f"{uid}_grid.parquet"
        grid.to_parquet(out)
        print(f"  {uid}: {len(grid)} cells -> {out.name}")
    _write_grid_manifest(preferred_meta)
    build_global_grid()


def main(api_keys=None, site_uids: list[str] | None = None, force: bool = False):
    """Build the weather grids (per-site + global)."""
    build_grids(site_uids=site_uids, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild all grids.")
    parser.add_argument("--site", action="append", help="Limit to these site UIDs (repeatable).")
    args = parser.parse_args()
    main(force=args.force, site_uids=args.site)
