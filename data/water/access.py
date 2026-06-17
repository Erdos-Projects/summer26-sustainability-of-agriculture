"""Read-only access layer for water quality data in data/water/.

All DataFrames returned by this module use 'datetime' as the index
(a UTC-aware DatetimeIndex). Site UIDs starting with 'WQS' are IWQIS
sites; those starting with 'USGS-' are USGS-NWIS sites.
"""

import pandas as pd
import numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_LOC_METADATA_PATH = _THIS_DIR / "site_location_metadata.csv"
_SITE_DATA_DIR = _THIS_DIR / "sites"

# lazy loading of location metadata
_LOCATION_META_DF = None


def _loc_df():
    global _LOCATION_META_DF
    if _LOCATION_META_DF is None:
        _LOCATION_META_DF = pd.read_csv(_LOC_METADATA_PATH)
    return _LOCATION_META_DF


def get_site_metadata() -> "pd.DataFrame":
    """Return the full location metadata DataFrame for all sites."""
    return _loc_df()


def get_all_water_sites():
    """Returns the unique identifiers of all the water quality sites as a list. These are the site_uid values.

    Returns
    -------
    list(str)
        a list containing all the water site identifiers
    """
    return list(_loc_df()["site_uid"].unique())


def get_all_iwqis_sites():
    """Return site_uids for all IWQIS sites (prefix 'WQS')."""
    return list(_loc_df()[_loc_df().site_uid.str.startswith("WQS")].site_uid.unique())


def get_all_usgs_sites():
    """Return site_uids for all USGS-NWIS sites (prefix 'USGS-')."""
    return list(_loc_df()[_loc_df().site_uid.str.startswith("USGS-")].site_uid.unique())


def get_site_data(site_uid: str):
    """Gets the full timeseries corresponding to a single site specified by site_uid

    Parameters
    ----------
    site_uid : str
        the unique identifier of the site

    Returns
    -------
    DataFrame
        the full data of the site
    """
    return pd.read_parquet(_SITE_DATA_DIR / f"{site_uid}_all_data.parquet")


def aggregate_by_interval(site_uid=None, df=None, value_col="nitrate_con", interval="1D", agg_func="mean"):
    """
    Aggregate a time series by a specified interval.

    Example
    -------
    hourly = aggregate_by_interval(df=site_df, value_col='nitrate_con', interval='1h')
    every_two_days = aggregate_by_interval(df=site_df, value_col='nitrate_con', interval='2D')

    Parameters
    ----------
    site_uid : str
        the unique identifier of a site whose data is to be aggregated
    df : pd.DataFrame
        a dataframe to be aggregated
    value_col : str or list
        Column(s) to aggregate.
    interval : str
        Pandas offset alias, e.g. '1h', '2h', '1D', '15min', '1W'.
    agg_func : str or callable
        Aggregation function, e.g. 'mean', 'sum', 'max', 'min', or a callable.

    Returns
    -------
    pd.DataFrame with a UTC-normalized DatetimeIndex.
    """
    if df is None:
        if site_uid is None:
            raise ValueError("provide either df or site_uid")
        df = get_site_data(site_uid)
    return df[value_col].resample(interval).agg(agg_func)


_ALL_BASINS_DF = None
_ALL_BASINS_UNION_DF = None


def get_all_basins() -> "geopandas.GeoDataFrame":
    """Return all site basin polygons as a single GeoDataFrame.

    One row per monitoring site. Use for downstream-site lookup: a spatial join
    of a point against this GDF returns the sites whose upstream catchment
    contains that point, i.e. the sites that are downstream of it.

    Requires ``data/water/basins/all_basins.parquet`` to exist; run
    ``make_basins.build_all_basins()`` to generate it.

    Returns
    -------
    GeoDataFrame (EPSG:4326), columns: site_id, comid, site_lat, site_lon, area_km2, geometry.
    """
    import geopandas as gpd

    global _ALL_BASINS_DF
    if _ALL_BASINS_DF is None:
        _ALL_BASINS_DF = gpd.read_parquet(_THIS_DIR / "basins" / "all_basins.parquet")
    return _ALL_BASINS_DF


def get_all_basins_union() -> "geopandas.GeoDataFrame":
    """Return the dissolved union of all site basins as a single-row GeoDataFrame.

    Use as an area-of-interest mask for modeling: clips rasters or filters
    features to the full extent of the monitored drainage network.

    Requires ``data/water/basins/all_basins_union.parquet`` to exist; run
    ``make_basins.build_all_basins_union()`` to generate it.

    Returns
    -------
    GeoDataFrame (EPSG:4326), columns: area_km2, geometry.
    """
    import geopandas as gpd

    global _ALL_BASINS_UNION_DF
    if _ALL_BASINS_UNION_DF is None:
        _ALL_BASINS_UNION_DF = gpd.read_parquet(_THIS_DIR / "basins" / "all_basins_union.parquet")
    return _ALL_BASINS_UNION_DF


def get_basins(site_uid: str):
    """Return the upstream drainage basin geometry for a site.

    Parameters
    ----------
    site_uid : str
        The unique identifier of the site.

    Returns
    -------
    GeoDataFrame (EPSG:4326)
        Columns: site_id, comid, site_lat, site_lon, area_km2, geometry.
    """
    import geopandas as gpd

    return gpd.read_parquet(_THIS_DIR / "basins" / f"{site_uid}_basins.parquet")


def make_site_timeseries_plot(site_uid, value_col="nitrate_con", interval="1D", agg_func="mean"):
    """Wrapper that integrates aggregate_by_interval with plotting code.

    Parameters
    ----------
    site_uid : str
        the unique identifier of the site to be plotted
    value_col : str or list
        Column(s) to aggregate.
    interval : str
        Pandas offset alias, e.g. '1h', '2h', '1D', '15min', '1W'.
    agg_func : str or callable
        Aggregation function, e.g. 'mean', 'sum', 'max', 'min', or a callable.
    """

    import plotly.express as px

    agg_df = aggregate_by_interval(site_uid=site_uid, value_col=value_col, interval=interval, agg_func=agg_func)
    fig = px.line(
        agg_df.reset_index(),
        x=agg_df.index.name,
        y=agg_df.columns.tolist() if isinstance(agg_df, pd.DataFrame) else agg_df.name,
        labels={"nitrate_con": "Nitrate mg/L"},
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        title_text=None,
        type="date",
    )
    fig.update_yaxes(fixedrange=True)
    fig.add_annotation(
        text=f"{site_uid} Daily Avg. Nitrate Concentration",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=11, color="#555"),
        xanchor="center",
    )
    fig.update_layout(margin=dict(l=40, r=10, t=20, b=50))
    return fig
