import math
import os
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
EQUAL_AREA_CRS = "EPSG:5070"  # NAD83 / Conus Albers -- the standard equal-area CRS for CONUS

THIS_DIR = Path(__file__).resolve().parent  # the directory in which this file is located
TARGET_DIR = THIS_DIR / "basins"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

LOC_METADATA = THIS_DIR / "site_location_metadata.csv"


def delineate_basin(
    lat: float,
    lon: float,
    site_id=None,
    simplified: bool = True,
    compute_area: bool = True,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Delineate the upstream drainage basin for a single (lat, lon) point.

    Two NLDI calls:
      1. point-in-polygon lookup -> the NHDPlus COMID of the catchment the point is in
      2. upstream accumulation   -> the basin polygon draining to that catchment outlet

    Parameters
    ----------
    lat, lon : float
        Coordinates in decimal degrees (WGS84 / EPSG:4326).
    site_id : optional
        Label carried onto the output row; defaults to ``comid_<COMID>``.
    simplified : bool
        Ask NLDI for a geometry-simplified polygon (smaller, faster to draw).
    compute_area : bool
        Add an ``area_km2`` column (equal-area, EPSG:5070). One extra reprojection.
    timeout : int
        Per-request timeout in seconds.

    Returns
    -------
    GeoDataFrame  (EPSG:4326), one row, columns:
        site_id, comid, site_lat, site_lon, [area_km2,] geometry

    Raises
    ------
    ValueError if the point does not resolve to an NHDPlus catchment (e.g. it is
    outside CONUS or off-network).
    """
    # WKT/GeoJSON coordinate order is lon-then-lat -- easy to flip by accident.
    pos = requests.get(
        f"{NLDI_BASE}/comid/position",
        params={"coords": f"POINT({lon} {lat})", "f": "json"},
        timeout=timeout,
    )
    pos.raise_for_status()
    feats = pos.json().get("features", [])
    if not feats:
        raise ValueError(
            f"No NHDPlus catchment for (lat={lat}, lon={lon}); " "point may be outside CONUS or off-network."
        )
    comid = feats[0]["properties"]["comid"]

    resp = requests.get(
        f"{NLDI_BASE}/comid/{comid}/basin",
        params={"f": "json", "simplified": str(simplified).lower()},
        timeout=timeout,
    )
    resp.raise_for_status()
    basin = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    if basin.empty:
        raise ValueError(f"NLDI returned an empty basin for COMID {comid}.")

    basin = basin[["geometry"]].copy()
    basin["site_id"] = str(site_id) if site_id is not None else f"comid_{comid}"
    basin["comid"] = comid
    basin["site_lat"] = lat
    basin["site_lon"] = lon
    if compute_area:
        basin = add_area_km2(basin)

    cols = ["site_id", "comid", "site_lat", "site_lon"]
    if compute_area:
        cols.append("area_km2")
    cols.append("geometry")
    return basin[cols]


# --------------------------------------------------------------------------- #
# helper: drainage area in km^2                                               #
# --------------------------------------------------------------------------- #
def add_area_km2(
    gdf: gpd.GeoDataFrame,
    equal_area_crs: str = EQUAL_AREA_CRS,
    col: str = "area_km2",
) -> gpd.GeoDataFrame:
    """Add a drainage-area column in km^2, measured in an equal-area projection.

    Geometry is reprojected to ``equal_area_crs`` *only* to measure area; the
    returned GeoDataFrame keeps its original CRS and geometry. Idempotent --
    overwrites ``col`` if it already exists.

    Caveat: basins fetched with ``simplified=True`` have generalized boundaries, so
    their area is approximate. Use ``simplified=False`` if you need survey-grade area.
    """
    out = gdf.copy()
    out[col] = gdf.to_crs(equal_area_crs).area / 1e6
    return out


def save_basins(gdf: gpd.GeoDataFrame, path) -> str:
    """Write basins to disk, dispatching on the file extension.

    .parquet            GeoParquet -- recommended: keeps CRS + dtypes, single file,
                        fast to re-read with gpd.read_parquet (needs pyarrow).
    .gpkg               GeoPackage -- single-file, GIS-friendly, no column limits.
    .geojson / .json    GeoJSON (written in EPSG:4326).
    .shp                Shapefile -- lossy: 10-char column names, multi-file sidecars.

    Returns the path written.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".parquet":
        gdf.to_parquet(path)
    elif ext == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    elif ext in (".geojson", ".json"):
        gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    elif ext == ".shp":
        gdf.to_file(path)
    else:
        raise ValueError(f"Unsupported extension {ext!r}; use .parquet, .gpkg, .geojson, or .shp")
    return str(path)


def get_all_site_locations():
    loc_metadata = pd.read_csv(LOC_METADATA)
    return loc_metadata[["site_uid", "latitude", "longitude"]]


def make_all_basins():
    # get all locations
    locations = get_all_site_locations()
    n_sites = len(locations.site_uid.unique())
    print(f"Making the water basins for {n_sites} sites.")
    for i, uid in enumerate(locations.site_uid.unique()):
        lat = locations.loc[locations.site_uid == uid, "latitude"].iloc[0]
        lon = locations.loc[locations.site_uid == uid, "longitude"].iloc[0]
        basin = delineate_basin(lat, lon)
        save_basins(basin, path=TARGET_DIR / f"{uid}_basins.parquet")
        print(f"  ({i}/{n_sites}): saved {TARGET_DIR / f"{uid}_basins.parquet"}")


def main(api_keys):
    make_all_basins()


if __name__ == "__main__":
    main(None)
