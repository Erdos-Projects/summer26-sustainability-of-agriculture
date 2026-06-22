"""Build per-site nitrogen surplus parquets from the Iowa grid dataset.

Inputs  (surplus_source/)
    iowa_grid_lookup.parquet    pixel_id → x, y (EPSG:5070), lon, lat (EPSG:4326)
    surplus{1,2,3}.parquet      pixel_id, year, surplus_kgha, total_kg_N

Intermediate (surplus_raw/  — gitignored)
    iowa_nitrogen_surplus.parquet   full 53M-row merged table
    images/iowa_surplus_{year}.png  Iowa-wide heatmaps + .json bounds sidecars

Outputs (surplus_data/)
    {site_uid}_surplus_pixel.parquet  rows whose (x, y) fall inside site_uid's preferred basin

Usage
-----
    python make_surplus.py           # process all sites, skip existing
    python make_surplus.py --force   # rewrite all
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import colormaps
from pathlib import Path
from PIL import Image
import shapely
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

_THIS_DIR = Path(__file__).resolve().parent
_SOURCE_DIR = _THIS_DIR / "surplus_source"
_RAW_DIR = _THIS_DIR / "surplus_raw"
_IMAGES_DIR = _RAW_DIR / "images"  # Iowa surplus PNGs + JSON bounds sidecars
_DATA_DIR = _THIS_DIR / "surplus_data"
_PIXEL_DIR = _DATA_DIR / "pixel"  # per-site pixel-level surplus parquets
_GRID_AGG_DIR = _DATA_DIR / "grid"  # per-site surplus aggregated onto the rain grid
_MERGED_FILE = _RAW_DIR / "iowa_nitrogen_surplus.parquet"
_MANIFEST_FILE = _DATA_DIR / ".basin_manifest.csv"
_GRID_INDEX = None  # cached (transform, height, width) for the Iowa raster
_SURPLUS_M = 250  # surplus pixel size (EPSG:5070 metres)


def _get_grid_index():
    """Build (and cache) the affine transform + shape used to rasterize basin polygons.

    Assumes grid_lookup's pixel_id is a row-major flatten index over (height, width) —
    the same convention write_iowa_surplus_images() relies on.
    """
    global _GRID_INDEX
    if _GRID_INDEX is not None:
        return _GRID_INDEX

    grid = pd.read_parquet(_SOURCE_DIR / "iowa_grid_lookup.parquet").sort_values("pixel_id")
    width = grid["x"].nunique()
    height = len(grid) // width
    if height * width != len(grid):
        raise ValueError("Grid lookup doesn't reshape into a clean rectangle.")

    # check the validitity of the grid assumption
    # form all x/y easting/northing values into two grids
    # x's should be equal across columns, y's equal across rows
    xs = grid["x"].to_numpy().reshape(height, width)
    ys = grid["y"].to_numpy().reshape(height, width)

    # get reference x coord difference and y coord difference
    res_x = xs[0, 1] - xs[0, 0]
    res_y = ys[0, 0] - ys[1, 0]  # row 0 assumed north (max y)

    assert np.allclose(np.diff(xs), res_x), "x spacing not uniform — pixel_id ordering assumption is wrong"
    assert np.allclose(np.diff(ys.transpose()), -res_y), "y spacing not uniform — pixel_id ordering assumption is wrong"
    # if this passes then the grid assumption is good

    transform = from_origin(xs[0, 0] - res_x / 2, ys[0, 0] + res_y / 2, res_x, res_y)
    _GRID_INDEX = (transform, height, width)
    return _GRID_INDEX


import gen_surplus_statistics as stats

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import basins
from data.access import get_rain_grid  # top-level access; no dependency on the rain module
from data.settings import get_config, get_equal_area_crs

EQUAL_AREA_CRS = get_equal_area_crs()
_surplus_cfg = get_config()["surplus"]
_YEAR_START, _YEAR_END = _surplus_cfg["year_start"], _surplus_cfg["year_end"]


def build_merged() -> pd.DataFrame:
    """Concatenate the surplus chunks (surplus*.parquet), join the grid lookup,
    write the full merged table to surplus_raw/. The chunk count is whatever
    build_source.py wrote (globbed, not hardcoded)."""
    chunk_files = sorted(_SOURCE_DIR.glob("surplus[0-9]*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No surplus chunks (surplus*.parquet) in {_SOURCE_DIR}. Run build_source.py.")
    surplus = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)

    lookup_file = _SOURCE_DIR / "iowa_grid_lookup.parquet"  # written by build_source.py
    lookup = pd.read_parquet(lookup_file)

    merged = surplus.merge(lookup, on="pixel_id")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(_MERGED_FILE, index=False)
    print(f"Wrote {len(merged):,} rows from {len(chunk_files)} chunks + {lookup_file.name} → {_MERGED_FILE}")
    return merged


def _merged_stale() -> bool:
    """True if the merged cache is missing or older than any source file."""
    if not _MERGED_FILE.exists():
        return True
    m_mtime = _MERGED_FILE.stat().st_mtime
    sources = list(_SOURCE_DIR.glob("surplus[0-9]*.parquet"))
    sources.append(_SOURCE_DIR / "iowa_grid_lookup.parquet")
    return any(s.exists() and s.stat().st_mtime > m_mtime for s in sources)


def _build_site_surplus(site_uid, merged: pd.DataFrame, basin):
    t0 = time.perf_counter()

    poly = basin.to_crs(EQUAL_AREA_CRS).geometry.union_all()

    transform, height, width = _get_grid_index()
    mask = geometry_mask([poly], out_shape=(height, width), transform=transform, invert=True)
    inside_pixel_ids = np.flatnonzero(mask.ravel())

    inside = merged[merged["pixel_id"].isin(inside_pixel_ids)]

    elapsed = time.perf_counter() - t0
    return inside, elapsed


def _aggregate_surplus_grid(merged: pd.DataFrame, grid) -> pd.DataFrame | None:
    """Area-weighted surplus per (node_id, year) on a site's rain grid.

    Each 250 m surplus cell is a square; its overlap area with each rain Voronoi
    cell weights its contribution. Assumes every rain cell is fully covered by
    surplus pixels (true once the source tifs span the whole region), so values
    are normalised by the full cell area — no partial-coverage bookkeeping:
        total_kg_N   = Σ area(D∩R)·surplus_kgha(D) / 1e4   (kg, the cell's N sum)
        surplus_kgha = total_kg_N / cell_area_ha           (intensive mean)
    Returns None if no surplus pixels overlap the grid.
    """
    minx, miny, maxx, maxy = grid.total_bounds
    h = _SURPLUS_M / 2
    m = merged[merged["x"].between(minx - h, maxx + h) & merged["y"].between(miny - h, maxy + h)]
    if m.empty:
        return None

    # 250 m square per unique surplus pixel (geometry is static across years).
    upx = m.drop_duplicates("pixel_id")[["pixel_id", "x", "y"]].reset_index(drop=True)
    squares = gpd.GeoDataFrame(
        {"pixel_id": upx["pixel_id"]},
        geometry=[shapely.box(x - h, y - h, x + h, y + h) for x, y in zip(upx["x"], upx["y"])],
        crs=grid.crs,
    )

    inter = gpd.overlay(squares, grid[["node_id", "geometry"]], how="intersection")
    if inter.empty:
        return None
    inter["area_DR"] = inter.geometry.area

    j = inter.merge(m[["pixel_id", "year", "surplus_kgha"]], on="pixel_id")
    j["w_density"] = j["area_DR"] * j["surplus_kgha"]  # area-weighted surplus per overlap
    g = (
        j.groupby(["node_id", "year"])
        .agg(w_density=("w_density", "sum"))
        .reset_index()
        .merge(grid[["node_id", "cell_area"]], on="node_id")
    )
    g["total_kg_N"] = g["w_density"] / 1e4  # Σ area·density (m²·kg/ha) → kg
    g["surplus_kgha"] = g["w_density"] / g["cell_area"]  # mean over the (full) cell
    return g[["node_id", "year", "surplus_kgha", "total_kg_N"]]


def write_site_surplus_pixel(site_uid: str, merged: pd.DataFrame, force: bool = False) -> bool:
    """Write a site's pixel-level surplus to surplus_data/pixel/.

    These are the surplus cells whose centre falls inside the basin polygon
    (intensive surplus_kgha + extensive total_kg_N per pixel per year). No rain
    grid involved. Returns True if the parquet was written."""
    out = _PIXEL_DIR / f"{site_uid}_surplus_pixel.parquet"
    if out.exists() and not force:
        return False

    try:
        basin = basins.get_basin(site_uid)
    except (KeyError, FileNotFoundError) as e:
        print(f"  {site_uid}: no basin — {e}")
        return False

    inside, elapsed = _build_site_surplus(site_uid=site_uid, merged=merged, basin=basin)
    if inside.empty:
        print(f"  {site_uid}: no grid points inside basin polygon")
        return False

    _PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    inside.to_parquet(out, index=False)
    n_pixels = len(inside) // inside["year"].nunique()
    print(
        f"  {site_uid}: pixel {n_pixels} px × {inside['year'].nunique()} years = {len(inside):,} rows   ({elapsed:.0f} sec)"
    )
    return True


def write_site_surplus_grid(site_uid: str, merged: pd.DataFrame, force: bool = False) -> bool:
    """Write a site's surplus aggregated onto the rain grid to surplus_data/grid/.

    Area-weights each 250 m surplus cell onto the rain Voronoi cells (see
    _aggregate_surplus_grid). Needs the rain grid from make_rain. Independent of
    the pixel-level output. Returns True if the parquet was written."""
    out = _GRID_AGG_DIR / f"{site_uid}_surplus_grid.parquet"
    if out.exists() and not force:
        return False

    try:
        grid = get_rain_grid(site_uid)
    except FileNotFoundError:
        print(f"  {site_uid}: no rain grid — run make_rain.py first; skipping surplus_grid")
        return False

    t0 = time.perf_counter()
    agg = _aggregate_surplus_grid(merged, grid)
    if agg is None:
        print(f"  {site_uid}: no surplus overlaps the rain grid")
        return False

    _GRID_AGG_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out, index=False)
    elapsed = time.perf_counter() - t0
    print(f"  {site_uid}: grid {len(agg):,} rows, {agg['node_id'].nunique()} nodes   ({elapsed:.1f}s)")
    return True


def write_iowa_surplus_images(merged: pd.DataFrame, force: bool = False) -> None:
    """Generate and save Iowa-wide surplus PNG + JSON bounds for each year.

    Used for displaying full surplus data overlay in the widget.

    Uses pixel_id as a flat row-major index into the (height, width) grid array
    so no unique() call is needed — fully vectorized, no OOM risk.
    Bounds are stored in lon/lat for correct Leaflet geographic alignment.
    Skips years whose PNG and JSON sidecar already exist unless force=True.
    """
    from data.surplus.access import _min_surplus, _max_surplus

    grid = pd.read_parquet(_SOURCE_DIR / "iowa_grid_lookup.parquet")
    width = grid["x"].nunique()
    height = len(grid) // width
    if height * width != len(grid):
        raise ValueError("Grid lookup doesn't reshape into a clean rectangle.")

    cmap = colormaps["YlOrRd"]
    lo, hi = _min_surplus(), _max_surplus()
    rng = hi - lo if hi != lo else 1.0

    bounds = [
        [float(grid["lat"].min()), float(grid["lon"].min())],
        [float(grid["lat"].max()), float(grid["lon"].max())],
    ]

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(merged["year"].unique())
    skipped = 0
    for year in years:
        img_path = _IMAGES_DIR / f"iowa_surplus_{year}.png"
        bounds_path = _IMAGES_DIR / f"iowa_surplus_{year}.json"
        if not force and img_path.exists() and bounds_path.exists():
            skipped += 1
            continue

        panel = merged[merged["year"] == year]

        flat = np.full(height * width, np.nan, dtype="float32")
        flat[panel["pixel_id"].to_numpy()] = panel["surplus_kgha"].to_numpy()
        img_arr = flat.reshape(height, width)

        t = np.clip((img_arr - lo) / rng, 0.0, 1.0)
        rgba = (cmap(t) * 255).astype(np.uint8)
        rgba[np.isnan(img_arr), 3] = 0  # transparent where no data

        Image.fromarray(rgba, mode="RGBA").save(img_path, format="PNG")
        with open(bounds_path, "w") as f:
            json.dump({"bounds": bounds}, f)
        print(f"  iowa {year}: saved")

    if skipped:
        print(f"  iowa images: {skipped}/{len(years)} already up to date, skipped.")


def _stale_sites(preferred_meta: pd.DataFrame) -> list[str]:
    """Return UIDs that need a (re)build.

    A site is stale if its surplus parquet is missing or if the basin recorded
    in .basin_manifest.csv differs from the current entry in preferred_basin.csv.
    """
    manifest = {}
    if _MANIFEST_FILE.exists():
        mdf = pd.read_csv(_MANIFEST_FILE)
        manifest = dict(zip(mdf["site_uid"], mdf["basin_name"]))

    return [
        row["site_uid"]
        for _, row in preferred_meta.iterrows()
        if not (_PIXEL_DIR / f"{row['site_uid']}_surplus_pixel.parquet").exists()
        or not (_GRID_AGG_DIR / f"{row['site_uid']}_surplus_grid.parquet").exists()
        or manifest.get(row["site_uid"]) != row["basin_name"]
    ]


def _write_manifest(preferred_meta: pd.DataFrame) -> None:
    """Record site_uid → basin_name for every surplus parquet that currently exists."""
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if (_PIXEL_DIR / f"{row['site_uid']}_surplus_pixel.parquet").exists()
    ]
    pd.DataFrame(rows).to_csv(_MANIFEST_FILE, index=False)


def main(api_keys=None, force: bool = False) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    preferred_meta = basins.get_metadata()

    # If build_source.py regenerated the source after the last merge, the cached
    # merge AND every per-site output built from it are stale — treat like --force.
    rebuild_all = force or _merged_stale()
    if rebuild_all and not force:
        print("Source newer than merged cache — rebuilding merge and all sites.")

    to_process = preferred_meta["site_uid"].tolist() if rebuild_all else _stale_sites(preferred_meta)

    n_total = len(preferred_meta)
    n_skip = n_total - len(to_process)
    print(f"Precheck: {n_skip}/{n_total} sites up to date, {len(to_process)} to build.")

    iowa_years_missing = [
        y
        for y in range(_YEAR_START, _YEAR_END + 1)
        if not (_IMAGES_DIR / f"iowa_surplus_{y}.png").exists() or not (_IMAGES_DIR / f"iowa_surplus_{y}.json").exists()
    ]
    needs_merged = bool(to_process) or bool(iowa_years_missing) or rebuild_all

    if not needs_merged:
        print("Nothing to do.")
        return

    if _MERGED_FILE.exists() and not rebuild_all:
        print(f"Loading merged dataset from {_MERGED_FILE.name}...")
        merged = pd.read_parquet(_MERGED_FILE)
    else:
        print("Building merged surplus dataset...")
        merged = build_merged()
    print(f"{len(merged):,} rows loaded.\n")

    if to_process:
        written = 0
        print(f"Processing {len(to_process)} sites...")
        for uid in to_process:
            # Two independent outputs; each method's own guard decides what to
            # (re)build, so a present pixel parquet doesn't block a missing grid.
            wrote_pixel = write_site_surplus_pixel(uid, merged, force=rebuild_all)
            wrote_grid = write_site_surplus_grid(uid, merged, force=rebuild_all)
            if wrote_pixel or wrote_grid:
                written += 1
        print(f"\nSites done: {written} updated.")
        _write_manifest(preferred_meta)
        stats.gen_surplus_statistics()

    print("\nGenerating Iowa surplus images...")
    write_iowa_surplus_images(merged, force=rebuild_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite existing parquets.")
    args = parser.parse_args()
    main(force=args.force)
