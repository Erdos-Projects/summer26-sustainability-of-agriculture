"""
download_nhd.py
===============
One-off script to download NHD medium-resolution flowlines and waterbodies
for Iowa from the USGS National Hydrography Dataset and save them locally.

The saved files are read by the app at startup instead of hitting USGS each
time. Re-run this script only if you want to refresh the data.

Dependencies (already in the conda env): pynhd, geopandas
"""

import sys
from pathlib import Path

import geopandas as gpd
from pynhd import NHD

THIS_DIR = Path(__file__).resolve().parent
SAVE_DIR = THIS_DIR / "overlays_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(THIS_DIR.parents[1]))
from data.settings import get_config

# ──────────────────────────────────────────────────────────────────────────────

# Minimum Strahler stream order to include in the flowlines layer (from config).
# 1–2 = tiny headwater streams (very large file, not very visible)
# 3   = a good balance: visible streams without overwhelming detail
# 4+  = only major rivers (small file, sparser coverage)
MIN_STREAM_ORDER = get_config()["map_overlays"]["min_stream_order"]


def iowa_geometry():
    print("Fetching Iowa boundary from Census TIGER …")
    states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip")
    return states.loc[states["NAME"] == "Iowa", "geometry"].iloc[0]


def download_flowlines(iowa_geom):
    print("Downloading NHD flowlines (medium resolution) …")
    nhd = NHD("flowline_mr")
    flowlines = nhd.bygeom(iowa_geom, geo_crs="EPSG:4326")
    print(f"  {len(flowlines):,} features before stream-order filter")

    if "streamorde" in flowlines.columns:
        flowlines = flowlines[flowlines["streamorde"] >= MIN_STREAM_ORDER].copy()
        print(f"  {len(flowlines):,} features after keeping streamorde >= {MIN_STREAM_ORDER}")
    else:
        print("  'streamorde' column not found — saving unfiltered")

    return flowlines


def download_waterbodies(iowa_geom):
    print("Downloading NHD waterbodies (medium resolution) …")
    nhd = NHD("waterbody_mr")
    waterbodies = nhd.bygeom(iowa_geom, geo_crs="EPSG:4326")
    print(f"  {len(waterbodies):,} features")
    return waterbodies


def main(api_key, force: bool = False):
    flowlines_path = SAVE_DIR / "iowa_flowlines.parquet"
    waterbodies_path = SAVE_DIR / "iowa_waterbodies.parquet"

    if flowlines_path.exists() and waterbodies_path.exists() and not force:
        print("Overlay files already exist; skipping download (pass force=True to overwrite).")
        return

    iowa_geom = iowa_geometry()

    if not flowlines_path.exists() or force:
        flowlines = download_flowlines(iowa_geom)
        flowlines.to_parquet(flowlines_path)
        print(f"Saved flowlines → {flowlines_path}  ({flowlines_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Flowlines already exist, skipping.")

    if not waterbodies_path.exists() or force:
        waterbodies = download_waterbodies(iowa_geom)
        waterbodies.to_parquet(waterbodies_path)
        print(f"Saved waterbodies → {waterbodies_path}  ({waterbodies_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Waterbodies already exist, skipping.")


if __name__ == "__main__":
    main(None)
