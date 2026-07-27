"""Download NHD medium-resolution flowlines + waterbodies for Iowa (widget map overlays).

Faithful port of data/map_overlays/make_map_overlays.py. One-off network build (pynhd / USGS NHD
+ Census TIGER boundary); the app reads the saved GeoParquet at startup rather than hitting USGS.

TWO consumers, not one: besides the widget basemap, _make_basins.py snaps every sensor to its
nearest NHD reach (basin1, the default delineation) and checks large-river proximity (flag_river)
against this layer. That is why make_data.py runs this builder BEFORE basins, and why the basin
builder raises rather than falling back when the layer is missing.

pynhd's bygeom returns features INTERSECTING the geometry (not clipped to it), so reaches that
cross the state line come through whole -- a border sensor still finds its own reach.

Output: src/data/processed/map_overlays/iowa_{flowlines,waterbodies}.parquet (EPSG:4326).
"""

import sys
from pathlib import Path

import geopandas as gpd

_THIS_DIR = Path(__file__).resolve().parent           # src/build
_SRC = _THIS_DIR.parent                               # src
sys.path.insert(0, str(_SRC.parent))                  # repo root on path

from src.build.config import get_config

_SAVE_DIR = _SRC / "data" / "processed" / "map_overlays"

# Minimum Strahler stream order to include in the flowlines layer (3 = a good balance).
# NOTE: NHD returns the column as `StreamOrde`, so the lowercase test below never fires and the
# FULL layer is kept. That is the behaviour basin1 needs -- its snap must be able to reach a small
# headwater sensor's own reach, which an order >= 3 layer would not contain. Left in place so the
# knob stays discoverable, but do not "fix" the casing without re-checking the snap.
MIN_STREAM_ORDER = get_config()["map_overlays"]["min_stream_order"]


def iowa_geometry():
    print("Fetching Iowa boundary from Census TIGER ...")
    states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip")
    return states.loc[states["NAME"] == "Iowa", "geometry"].iloc[0]


def download_flowlines(iowa_geom):
    from pynhd import NHD
    print("Downloading NHD flowlines (medium resolution) ...")
    flowlines = NHD("flowline_mr").bygeom(iowa_geom, geo_crs="EPSG:4326")
    if "streamorde" in flowlines.columns:
        flowlines = flowlines[flowlines["streamorde"] >= MIN_STREAM_ORDER].copy()
        print(f"  {len(flowlines):,} features (streamorde >= {MIN_STREAM_ORDER})")
    else:
        print(f"  {len(flowlines):,} features (unfiltered -- see MIN_STREAM_ORDER)")
    return flowlines


def download_waterbodies(iowa_geom):
    from pynhd import NHD
    print("Downloading NHD waterbodies (medium resolution) ...")
    return NHD("waterbody_mr").bygeom(iowa_geom, geo_crs="EPSG:4326")


def main(api_keys=None, force: bool = False) -> None:
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    flowlines_path = _SAVE_DIR / "iowa_flowlines.parquet"
    waterbodies_path = _SAVE_DIR / "iowa_waterbodies.parquet"

    if flowlines_path.exists() and waterbodies_path.exists() and not force:
        print("Overlay files already exist; skipping download (pass force=True to overwrite).")
        return

    iowa_geom = iowa_geometry()
    if not flowlines_path.exists() or force:
        download_flowlines(iowa_geom).to_parquet(flowlines_path)
        print(f"Saved flowlines -> {flowlines_path}")
    if not waterbodies_path.exists() or force:
        download_waterbodies(iowa_geom).to_parquet(waterbodies_path)
        print(f"Saved waterbodies -> {waterbodies_path}")


if __name__ == "__main__":
    main()
