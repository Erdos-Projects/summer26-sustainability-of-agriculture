"""Build the surplus source files (grid lookup + per-year panel) from GeoTIFFs.

Reads single-band surplus rasters (Surplus_N_{year}.tif, EPSG:5070, 250 m),
clips them to the project region box (data/pipeline_config.toml [region]), and
writes:

    iowa_grid_lookup.parquet  pixel_id -> x, y (raster CRS), lon, lat (EPSG:4326).
                              One row per cell in the cropped extent (valid or
                              not), so pixel_ids stay stable across years as the
                              nodata footprint shifts. (Name kept for the existing
                              consumers in make_surplus.py / erin/map_script.py.)
    surplus{1..N}.parquet     pixel_id, year, surplus_kgha, total_kg_N. The
                              per-year panel split into N git-friendly chunks;
                              coordinates live in the lookup. build_merged() globs
                              and concatenates them back.

Usage
-----
    python build_source.py                          # clip to the region box
    python build_source.py --tif-dir DIR --out-dir DIR
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from pyproj import Transformer

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data.settings import get_region_bbox

DEFAULT_TIF_DIR = _THIS_DIR / "surplus_source/surplus"  # put Surplus_N_*.tif here (override with --tif-dir)
DEFAULT_OUT_DIR = _THIS_DIR / "surplus_source"
GLOB = "Surplus_N_*.tif"
N_CHUNKS = 18  # split the panel into this many parquets to keep each git-friendly


def _region_mask(raster_crs) -> list:
    """The project region box (config [region]) as mask geometries in raster_crs."""
    min_lon, min_lat, max_lon, max_lat = get_region_bbox()
    region = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326").to_crs(raster_crs)
    return list(region.geometry)


def build_source(tif_dir=DEFAULT_TIF_DIR, out_dir=DEFAULT_OUT_DIR):
    """Clip surplus rasters to the region box; write the grid lookup + per-year panel.

    Returns (grid_df, panel).
    """
    # get the tif files
    tif_dir, out_dir = Path(tif_dir), Path(out_dir)
    tif_files = sorted(tif_dir.glob(GLOB))
    if not tif_files:
        raise FileNotFoundError(f"No files matching {GLOB!r} in {tif_dir}.")
    print(f"Found {len(tif_files)} files: {[f.name for f in tif_files]}")

    # get the raster CRS type, e.g. EPSG:5070
    # to_latlon used to put get the lat, lon coords from EPSG:5070 in the DataFrames
    with rasterio.open(tif_files[0]) as src0:
        raster_crs = src0.crs
    mask_geoms = _region_mask(raster_crs)
    to_latlon = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    grid_df = None  # static lookup pixel_id -> x, y, lon, lat (built from the first file)
    ref_shape = None  # (height, width) consistency check across years
    pixel_area_ha = None
    all_years = []

    for tif_path in tif_files:
        # assumes files named things like Surplus_N_2000.tif
        year = int(tif_path.stem.split("_")[-1])
        print(f"Processing {year}...")

        # get all the data out of the GeoTIFF
        with rasterio.open(tif_path) as src:
            out_image, out_transform = rio_mask(src, mask_geoms, crop=True, nodata=src.nodata, filled=True)
            band = out_image[0]  # the only band actually
            nodata = src.nodata  # number which marks "no data" in the tiff
            res_x, res_y = src.res  # width and height of one pixel in crs units
            # res_x and res_y should both be 250m here

        height, width = band.shape

        # this should only pass once
        # used to ensure grid is consistent across years
        if ref_shape is None:
            # Lock the reference grid and build the coordinate lookup for EVERY cell
            # in the cropped extent (not just this year's valid cells), so pixel_ids
            # line up across years even as the nodata footprint shifts.
            ref_shape = (height, width)
            pixel_area_ha = (res_x * res_y) / 10_000  # 250 m -> 6.25 ha

            rr, cc = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            rr, cc = rr.ravel(), cc.ravel()
            full_pixel_id = (rr.astype(np.int64) * width + cc).astype(np.int32)

            xs, ys = rasterio.transform.xy(out_transform, rr, cc)
            xs, ys = np.array(xs), np.array(ys)
            lons, lats = to_latlon.transform(xs, ys)

            grid_df = pd.DataFrame(
                {
                    "pixel_id": full_pixel_id,
                    "x": xs.astype("float32"),
                    "y": ys.astype("float32"),
                    "lon": np.asarray(lons, dtype="float32"),
                    "lat": np.asarray(lats, dtype="float32"),
                }
            )
        else:
            assert (height, width) == ref_shape, (
                f"{tif_path.name} cropped to {(height, width)} vs {ref_shape}; pixel_ids "
                f"would not line up. All rasters must share one grid origin/extent."
            )

        # builds a mask containing all the actual entires,
        # i.e. those with values != nodata
        if nodata is not None and not np.isnan(nodata):
            valid = band != nodata
        else:
            valid = ~np.isnan(band)
        rows, cols = np.where(valid)

        df = pd.DataFrame(
            {
                "pixel_id": (rows.astype(np.int64) * width + cols).astype(np.int32),
                "year": np.int16(year),
                "surplus_kgha": band[rows, cols].astype("float32"),
            }
        )
        # also store the total surplus of Nitrogen
        df["total_kg_N"] = (df["surplus_kgha"] * pixel_area_ha).astype("float32")
        all_years.append(df)

    panel = pd.concat(all_years, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(out_dir / "iowa_grid_lookup.parquet", index=False)

    # Split the panel into N_CHUNKS parquets so each file stays git-friendly;
    # build_merged() globs surplus*.parquet and concatenates them back. Split by
    # row position and slice with iloc — np.array_split(df) would coerce each
    # chunk to a bare ndarray (losing columns/dtypes, and with no .to_parquet).
    for i, pos in enumerate(np.array_split(np.arange(len(panel)), N_CHUNKS), start=1):
        panel.iloc[pos].to_parquet(out_dir / f"surplus{i}.parquet", index=False, compression="snappy")

    print(f"\nGrid cells (lookup): {len(grid_df):,}  -> {out_dir / 'iowa_grid_lookup.parquet'}")
    print(f"Panel rows (pixel x year): {len(panel):,}  -> surplus1..{N_CHUNKS}.parquet in {out_dir}")
    return grid_df, panel


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tif-dir", default=str(DEFAULT_TIF_DIR), help="Directory of Surplus_N_*.tif files.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the parquets.")
    args = p.parse_args()
    build_source(tif_dir=args.tif_dir, out_dir=args.out_dir)
