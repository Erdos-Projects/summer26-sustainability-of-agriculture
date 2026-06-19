"""Build per-site nitrogen surplus parquets from the Iowa grid dataset.

Inputs  (surplus_source/)
    iowa_grid_lookup.parquet    pixel_id → x, y (EPSG:5070), lon, lat (EPSG:4326)
    surplus{1,2,3}.parquet      pixel_id, year, surplus_kgha, total_kg_N

Intermediate (surplus_raw/  — gitignored)
    iowa_nitrogen_surplus.parquet   full 53M-row merged table

Outputs (surplus_data/)
    {site_uid}_surplus.parquet  rows whose (x, y) fall inside site_uid's preferred basin

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

_THIS_DIR = Path(__file__).resolve().parent
_TOP_DATA = _THIS_DIR.parent
_SOURCE_DIR = _THIS_DIR / "surplus_source"
_RAW_DIR = _THIS_DIR / "surplus_raw"
_DATA_DIR = _THIS_DIR / "surplus_data"
_MERGED_FILE = _RAW_DIR / "iowa_nitrogen_surplus.parquet"
_MANIFEST_FILE = _DATA_DIR / ".basin_manifest.csv"

import gen_surplus_statistics as stats

sys.path.insert(0, str(_TOP_DATA.parent))
from data import basins

EQUAL_AREA_CRS = "EPSG:5070"


def build_merged() -> pd.DataFrame:
    """Concatenate the three surplus chunks, join the grid lookup, write to surplus_raw/."""
    chunks = [pd.read_parquet(_SOURCE_DIR / f"surplus{i}.parquet") for i in range(1, 4)]
    surplus = pd.concat(chunks, ignore_index=True)
    lookup = pd.read_parquet(_SOURCE_DIR / "iowa_grid_lookup.parquet")
    merged = surplus.merge(lookup, on="pixel_id")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(_MERGED_FILE, index=False)
    print(f"Wrote {len(merged):,} rows → {_MERGED_FILE}")
    return merged


def write_site_surplus(site_uid: str, merged: pd.DataFrame, force: bool = False) -> bool:
    """Intersect merged grid with site_uid's preferred basin; save result.

    Returns True if a file was written, False if skipped or empty.
    """
    out = _DATA_DIR / f"{site_uid}_surplus.parquet"
    if out.exists() and not force:
        return False
    t0 = time.perf_counter()

    try:
        basin = basins.get_basin(site_uid)
    except (KeyError, FileNotFoundError) as e:
        print(f"  {site_uid}: no basin — {e}")
        return False

    poly = basin.to_crs(EQUAL_AREA_CRS).geometry.union_all()

    # Bounding-box pre-filter on Cartesian x, y (fast column comparison)
    minx, miny, maxx, maxy = poly.bounds
    candidates = merged[(merged["x"] >= minx) & (merged["x"] <= maxx) & (merged["y"] >= miny) & (merged["y"] <= maxy)]

    if candidates.empty:
        print(f"  {site_uid}: no grid points in bounding box")
        return False

    # Spatial check on unique pixels only (pixel_id deduplicates across years)
    unique_pixels = candidates.drop_duplicates("pixel_id")
    pts = gpd.GeoSeries(
        gpd.points_from_xy(unique_pixels["x"], unique_pixels["y"]),
        index=unique_pixels.index,
        crs=EQUAL_AREA_CRS,
    )
    inside_ids = unique_pixels.loc[pts.within(poly), "pixel_id"]
    inside = candidates[candidates["pixel_id"].isin(inside_ids)]

    if inside.empty:
        print(f"  {site_uid}: no grid points inside basin polygon")
        return False

    inside.to_parquet(out, index=False)
    n_pixels = len(inside) // inside["year"].nunique()
    elapsed = time.perf_counter() - t0
    print(
        f"  {site_uid}: {n_pixels} pixels × {inside['year'].nunique()} years = {len(inside):,} rows   ({elapsed:.0f} sec)"
    )
    return True


def write_iowa_surplus_images(merged: pd.DataFrame, force: bool = False) -> None:
    """Generate and save Iowa-wide surplus PNG + JSON bounds for each year.

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

    years = sorted(merged["year"].unique())
    skipped = 0
    for year in years:
        img_path = _RAW_DIR / f"iowa_surplus_{year}.png"
        bounds_path = _RAW_DIR / f"iowa_surplus_{year}.json"
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
        if not (_DATA_DIR / f"{row['site_uid']}_surplus.parquet").exists()
        or manifest.get(row["site_uid"]) != row["basin_name"]
    ]


def _write_manifest(preferred_meta: pd.DataFrame) -> None:
    """Record site_uid → basin_name for every surplus parquet that currently exists."""
    rows = [
        {"site_uid": row["site_uid"], "basin_name": row["basin_name"]}
        for _, row in preferred_meta.iterrows()
        if (_DATA_DIR / f"{row['site_uid']}_surplus.parquet").exists()
    ]
    pd.DataFrame(rows).to_csv(_MANIFEST_FILE, index=False)


def main(api_keys=None, force: bool = False) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    preferred_meta = basins.get_metadata()

    to_process = preferred_meta["site_uid"].tolist() if force else _stale_sites(preferred_meta)

    n_total = len(preferred_meta)
    n_skip = n_total - len(to_process)
    print(f"Precheck: {n_skip}/{n_total} sites up to date, {len(to_process)} to build.")

    iowa_years_missing = [
        y
        for y in range(2000, 2018)
        if not (_RAW_DIR / f"iowa_surplus_{y}.png").exists() or not (_RAW_DIR / f"iowa_surplus_{y}.json").exists()
    ]
    needs_merged = bool(to_process) or bool(iowa_years_missing) or force

    if not needs_merged:
        print("Nothing to do.")
        return

    if _MERGED_FILE.exists() and not force:
        print(f"Loading merged dataset from {_MERGED_FILE.name}...")
        merged = pd.read_parquet(_MERGED_FILE)
    else:
        print("Building merged surplus dataset...")
        merged = build_merged()
    print(f"{len(merged):,} rows loaded.\n")

    if to_process:
        written = failed = 0
        print(f"Processing {len(to_process)} sites...")
        for uid in to_process:
            ok = write_site_surplus(uid, merged, force=True)
            if ok:
                written += 1
            else:
                failed += 1
        print(f"\nSites done: {written} written, {failed} failed/empty.")
        _write_manifest(preferred_meta)
        stats.gen_surplus_statistics()

    print("\nGenerating Iowa surplus images...")
    write_iowa_surplus_images(merged, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite existing parquets.")
    args = parser.parse_args()
    main(force=args.force)
