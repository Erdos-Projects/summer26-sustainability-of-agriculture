"""Aggregate CDL crops onto the rain grid, per site.

For each site this reads the shared rain grid (the Voronoi target cells built by
make_rain), assigns every CDL pixel to the cell that contains it, relabels the
pixel's CDL code through a remapping function, and counts pixels per class.

Output (crops_data/grid/)
    {site_uid}_crops_grid.parquet   one row per (node_id, year); columns node_id,
                                    global_node_id, year, then one integer
                                    pixel-count column per class produced by the
                                    remap function. Join to data.get_rain_grid on
                                    node_id for coordinates; global_node_id is the
                                    canonical IEM cell index, shared across basins.

The remap is any Callable[[int], str] (default cdl_legend.cdl_to_class). The
output columns are the full set of class labels the remap can emit (probed over
all 256 CDL byte values), so every site/year shares the same columns.

Assignment is done by rasterizing the rain cells onto the CDL clip grid (a pixel
joins the cell containing its centre) and counting with np.bincount — no
per-pixel geometry, point GeoDataFrame, or spatial join.

Source
    crops_raw/clipped/cdl_clip_{year}.tif — build with clip_crops.py first.
    rain grid                              — build with make_rain.py first.

Usage
-----
    python make_crops.py                  # all stale sites, all years
    python make_crops.py --force          # rebuild everything
    python make_crops.py --site WQS0039   # one site (repeatable)
    python make_crops.py --year 2016      # one year (repeatable)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, intersection, Window

_THIS_DIR = Path(__file__).resolve().parent
_RAW_DIR = _THIS_DIR / "crops_raw"
_CLIP_DIR = _RAW_DIR / "clipped"  # regional CDL clips (built by clip_crops.py)
_DATA_DIR = _THIS_DIR / "crops_data"
_GRID_AGG_DIR = _DATA_DIR / "grid"  # per-site crops aggregated onto the rain grid
_MANIFEST_FILE = _GRID_AGG_DIR / ".basin_manifest.csv"

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import basins, get_rain_grid  # top-level access; no rain/surplus module imports
from data.crops.cdl_legend import cdl_to_class
from data.settings import get_config

_crops_cfg = get_config()["crops"]
YEARS = list(range(_crops_cfg["year_start"], _crops_cfg["year_end"] + 1))


def _grid_path(site_uid: str) -> Path:
    return _GRID_AGG_DIR / f"{site_uid}_crops_grid.parquet"


def _clip_path(year: int) -> Path:
    return _CLIP_DIR / f"cdl_clip_{year}.tif"


def _build_luts(remap) -> tuple[list[str], np.ndarray]:
    """From a code→class remap, build the sorted class list and a 256-entry
    code→class-index lookup (for vectorized counting)."""
    labels = [remap(c) for c in range(256)]
    classes = sorted(set(labels))
    idx = {c: i for i, c in enumerate(classes)}
    code_to_classidx = np.array([idx[labels[c]] for c in range(256)], dtype=np.int32)
    return classes, code_to_classidx


def _read_window(clip_path: Path, bounds):
    """Read the CDL band over `bounds` (in the clip CRS). Returns (band, transform,
    crs) or (None, None, None) if the bounds fall outside the clip."""
    minx, miny, maxx, maxy = bounds
    with rasterio.open(clip_path) as s:
        window = from_bounds(minx, miny, maxx, maxy, s.transform).round_offsets().round_lengths()
        window = intersection(window, Window(0, 0, s.width, s.height))
        if window.width <= 0 or window.height <= 0:
            return None, None, None
        return s.read(1, window=window), s.window_transform(window), s.crs


def aggregate_site_crops(grid, years, classes, code_to_classidx) -> pd.DataFrame | None:
    """Pixel counts per (node_id, year), one column per class.

    For each year: read the CDL window over the grid's extent, rasterize the rain
    cells onto that window (pixel center -> containing cell), and count pixels per
    (cell, class) with np.bincount. Returns None if no year yields pixels.
    """
    n_nodes, n_classes = len(grid), len(classes)
    node_ids = grid["node_id"].to_numpy()
    global_ids = grid["global_node_id"].to_numpy()  # canonical IEM cell index, shared across basins
    positions = np.arange(n_nodes)  # burn position+1 so 0 = "outside any cell"
    bounds = grid.total_bounds

    frames = []
    for year in years:
        clip = _clip_path(year)
        if not clip.exists():
            print(f"    {year}: no clip in crops_raw/clipped/ — skipping")
            continue
        band, transform, crs = _read_window(clip, bounds)
        if band is None:
            continue

        geoms = grid.geometry if crs == grid.crs else grid.to_crs(crs).geometry
        node_raster = rasterize(
            ((geom, int(pos) + 1) for geom, pos in zip(geoms, positions)),
            out_shape=band.shape,
            transform=transform,
            fill=0,
            dtype=np.int32,
        )

        mask = (node_raster > 0) & (band != 0)  # inside a cell and not background
        if not mask.any():
            continue
        pos = node_raster[mask] - 1  # 0..n_nodes-1 (grid row position)
        cls = code_to_classidx[band[mask]]
        counts = np.bincount(pos * n_classes + cls, minlength=n_nodes * n_classes).reshape(n_nodes, n_classes)

        nz = counts.sum(axis=1) > 0  # nodes with any pixel this year
        df = pd.DataFrame(counts[nz], columns=classes)
        df.insert(0, "year", year)
        df.insert(0, "global_node_id", global_ids[nz])
        df.insert(0, "node_id", node_ids[nz])
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def write_site_crops(
    site_uid: str, years: list, classes: list[str], code_to_classidx: np.ndarray, force: bool = False
) -> bool:
    out = _grid_path(site_uid)
    if out.exists() and not force:
        return False
    try:
        grid = get_rain_grid(site_uid)
    except FileNotFoundError as e:
        print(f"  {site_uid}: {e}")
        return False

    t0 = time.perf_counter()
    agg = aggregate_site_crops(grid, years, classes, code_to_classidx)
    if agg is None:
        print(f"  {site_uid}: no crop data in any requested year")
        return False

    _GRID_AGG_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out, index=False)
    elapsed = time.perf_counter() - t0
    print(
        f"  {site_uid}: {len(agg):,} rows, {agg['node_id'].nunique()} nodes, {agg['year'].nunique()} years   ({elapsed:.1f}s)"
    )
    return True


def _stale_sites(preferred_meta: pd.DataFrame) -> list[str]:
    manifest = {}
    if _MANIFEST_FILE.exists():
        mdf = pd.read_csv(_MANIFEST_FILE)
        manifest = dict(zip(mdf["site_uid"], mdf["basin_name"]))
    return [
        row["site_uid"]
        for _, row in preferred_meta.iterrows()
        if not _grid_path(row["site_uid"]).exists() or manifest.get(row["site_uid"]) != row["basin_name"]
    ]


def _write_manifest(preferred_meta: pd.DataFrame) -> None:
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if _grid_path(row["site_uid"]).exists()
    ]
    pd.DataFrame(rows).to_csv(_MANIFEST_FILE, index=False)


def main(api_keys=None, force: bool = False, sites: list = None, remap=cdl_to_class) -> None:
    _GRID_AGG_DIR.mkdir(parents=True, exist_ok=True)
    classes, code_to_classidx = _build_luts(remap)
    tally_years = list(YEARS)

    preferred_meta = basins.get_metadata()
    all_sites = preferred_meta["site_uid"].tolist()
    n_sites = len(all_sites)

    # ── status report ───────────────────────────────────────────────────────
    clips_found = sum(_clip_path(y).exists() for y in YEARS)
    sites_with_data = sum(_grid_path(u).exists() for u in all_sites)
    print(f"{clips_found}/{len(YEARS)} clipped source files found")
    print(f"{sites_with_data}/{n_sites} sites have data in crops_data/grid")
    if clips_found != len(YEARS):
        print("To rebuild source, run clip_crops.py --download")

    if sites:
        to_process = sites
    elif force:
        to_process = all_sites
    else:
        to_process = _stale_sites(preferred_meta)

    print(f"Processing {len(to_process)} site(s) over {tally_years[0]}–{tally_years[-1]}; classes: {classes}")
    written = 0
    for uid in to_process:
        if write_site_crops(uid, tally_years, classes, code_to_classidx, force=True):
            written += 1
    print(f"\nbuilt {written}/{n_sites} site crop files")
    _write_manifest(preferred_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Rewrite all parquets.")
    parser.add_argument(
        "--site",
        action="append",
        metavar="SITE_UID",
        help="Process only specific site(s) (repeatable). Bypasses the staleness check.",
    )
    args = parser.parse_args()
    main(force=args.force, sites=args.site)
