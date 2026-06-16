"""
download_nhd.py
===============
One-off script to download NHD medium-resolution flowlines and waterbodies
for Iowa from the USGS National Hydrography Dataset and save them locally.

The saved files are read by the app at startup instead of hitting USGS each
time. Re-run this script only if you want to refresh the data.

Dependencies (already in the conda env): pynhd, geopandas
"""

from pathlib import Path

import geopandas as gpd
from pynhd import NHD

# ── CHANGE THIS to wherever you want the files saved ──────────────────────────
SAVE_DIR = Path("data/")
# ──────────────────────────────────────────────────────────────────────────────

# Minimum Strahler stream order to include in the flowlines layer.
# 1–2 = tiny headwater streams (very large file, not very visible)
# 3   = a good balance: visible streams without overwhelming detail
# 4+  = only major rivers (small file, sparser coverage)
MIN_STREAM_ORDER = 3


def iowa_geometry():
    print("Fetching Iowa boundary from Census TIGER …")
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    )
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


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    iowa_geom = iowa_geometry()

    flowlines = download_flowlines(iowa_geom)
    flowlines_path = SAVE_DIR / "iowa_flowlines.parquet"
    flowlines.to_parquet(flowlines_path)
    print(f"Saved flowlines → {flowlines_path}  ({flowlines_path.stat().st_size / 1e6:.1f} MB)")

    waterbodies = download_waterbodies(iowa_geom)
    waterbodies_path = SAVE_DIR / "iowa_waterbodies.parquet"
    waterbodies.to_parquet(waterbodies_path)
    print(f"Saved waterbodies → {waterbodies_path}  ({waterbodies_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. Update FLOWLINES_PATH / WATERBODIES_PATH in map_panel.py to point here.")


if __name__ == "__main__":
    main()
