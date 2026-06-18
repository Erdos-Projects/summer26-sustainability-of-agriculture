"""Access layer for basin data in data/basins/."""

import pandas as pd
import geopandas as gpd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASIN_DATA_DIR = _THIS_DIR / "basin_data"
_PREFERRED_CSV = _THIS_DIR / "preferred_basin.csv"

_PREFERRED_META_DF = None
_ALL_BASINS_DF = None
_ALL_BASINS_UNION_DF = None


def get_preferred_basin_metadata() -> pd.DataFrame:
    """Return preferred_basin.csv as a DataFrame (cached after first call)."""
    global _PREFERRED_META_DF
    if _PREFERRED_META_DF is None:
        if not _PREFERRED_CSV.exists():
            raise FileNotFoundError("preferred_basin.csv not found. Run make_basins.py to generate it.")
        _PREFERRED_META_DF = pd.read_csv(_PREFERRED_CSV)
    return _PREFERRED_META_DF


def get_basin(site_uid: str, type: int = 0) -> gpd.GeoDataFrame:
    """Return a basin polygon for site_uid.

    Parameters
    ----------
    site_uid : str
    type : int
        0 = preferred basin (default); 1, 2, or 3 = specific version.

    Raises FileNotFoundError if the parquet does not exist.
    Raises KeyError if type=0 and site_uid is not in preferred_basin.csv.
    """
    if type == 0:
        meta = get_preferred_basin_metadata()
        row = meta[meta["site_uid"] == site_uid]
        if row.empty:
            raise KeyError(f"No preferred basin entry for {site_uid}.")
        fname = row.iloc[0]["basin_name"]
    else:
        fname = f"{site_uid}_basin{type}.parquet"
    path = _BASIN_DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"No basin file: {path.name}")
    return gpd.read_parquet(path)


def get_all_basins() -> gpd.GeoDataFrame:
    """Return the preferred basin for every site, concatenated.

    Uses preferred_basin.csv to select the basin file per site.
    Result is cached after the first call.
    """
    global _ALL_BASINS_DF
    if _ALL_BASINS_DF is None:
        meta = get_preferred_basin_metadata()
        gdfs = []
        for _, row in meta.iterrows():
            path = _BASIN_DATA_DIR / row["basin_name"]
            if path.exists():
                gdfs.append(gpd.read_parquet(path))
        if not gdfs:
            raise FileNotFoundError("No preferred basin files found in basin_data/.")
        _ALL_BASINS_DF = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return _ALL_BASINS_DF


def get_all_basins_union() -> gpd.GeoDataFrame:
    """Return the dissolved union of all preferred basins as a single-row GeoDataFrame.

    Result is cached after the first call.
    """
    global _ALL_BASINS_UNION_DF
    if _ALL_BASINS_UNION_DF is None:
        all_df = get_all_basins()
        union = all_df.dissolve()[["geometry"]].reset_index(drop=True)
        union["area_km2"] = union.to_crs("EPSG:5070").area / 1e6
        _ALL_BASINS_UNION_DF = union
    return _ALL_BASINS_UNION_DF


def update_basin(site_uid: str, params: dict = None, basin_geom=None) -> None:
    """Update preferred_basin.csv for site_uid and optionally save a custom basin.

    Parameters
    ----------
    site_uid : str
    params : dict, optional
        Column–value pairs to update in the site_uid row of preferred_basin.csv.
    basin_geom : GeoJSON dict or GeoDataFrame, optional
        Custom basin geometry. Saved as {site_uid}_basin4.parquet; automatically
        sets basin_name and basin_type=4 in preferred_basin.csv.
    """
    global _PREFERRED_META_DF, _ALL_BASINS_DF, _ALL_BASINS_UNION_DF

    if params is None:
        params = {}

    if not _PREFERRED_CSV.exists():
        raise FileNotFoundError("preferred_basin.csv not found. Run make_basins.py to generate it.")

    df = pd.read_csv(_PREFERRED_CSV)
    mask = df["site_uid"] == site_uid
    if not mask.any():
        raise KeyError(f"No entry for {site_uid} in preferred_basin.csv.")

    if basin_geom is not None:
        if isinstance(basin_geom, dict):
            if basin_geom.get("type") == "FeatureCollection":
                gdf = gpd.GeoDataFrame.from_features(basin_geom["features"], crs="EPSG:4326")
            else:
                from shapely.geometry import shape

                gdf = gpd.GeoDataFrame(geometry=[shape(basin_geom)], crs="EPSG:4326")
        else:
            gdf = basin_geom
        out_path = _BASIN_DATA_DIR / f"{site_uid}_basin4.parquet"
        gdf.to_parquet(out_path)
        params = dict(params)
        params["basin_name"] = f"{site_uid}_basin4.parquet"
        params["basin_type"] = 4

    for col, val in params.items():
        if col not in df.columns:
            df[col] = None
        df.loc[mask, col] = val

    df.to_csv(_PREFERRED_CSV, index=False)

    _PREFERRED_META_DF = None
    _ALL_BASINS_DF = None
    _ALL_BASINS_UNION_DF = None
