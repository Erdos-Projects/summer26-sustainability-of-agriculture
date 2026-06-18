"""Read-only access layer for water quality data in data/water/.

All DataFrames returned by this module use 'datetime' as the index
(a UTC-aware DatetimeIndex). Site UIDs starting with 'WQS' are IWQIS
sites; those starting with 'USGS-' are USGS-NWIS sites.
"""

import pandas as pd
import numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_METADATA_DIR = _THIS_DIR / "metadata"
_LOC_METADATA_PATH = _METADATA_DIR / "site_location_metadata.csv"
_SITE_DATA_DIR = _THIS_DIR / "sites"
_STATS_FILE = _METADATA_DIR / "site_statistics.csv"

# lazy loading of location metadata
_LOCATION_META_DF = None


def _loc_df():
    global _LOCATION_META_DF
    if _LOCATION_META_DF is None:
        _LOCATION_META_DF = pd.read_csv(_LOC_METADATA_PATH)
    return _LOCATION_META_DF


# lazy loading of stats data
_STATS_DF = None


def _stats_df():
    global _STATS_DF
    if _STATS_DF is None:
        if not _STATS_FILE.exists():
            raise FileNotFoundError(
                f"Site statistics not found at {_STATS_FILE}. Run make_water.py to generate them."
            )
        _STATS_DF = pd.read_csv(_STATS_FILE)
    return _STATS_DF


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


def get_full_data():
    """Gets all data in THIS_DIR / sites.

    Returns
    -------
    DataFrame
        the full water sites dataset.
    """
    dfs = {uid: get_site_data(uid) for uid in get_all_water_sites()}
    return pd.concat(dfs, names=["uid", "datetime"]).reset_index(level=1).reset_index(drop=True)


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


def get_full_stats():
    """Get all statistics from data/water/site_statistics.csv.
                column : description
      nitrate_sparsity : % rows with a non-nan nitrate_con entry
            first_date : earliest entry date
     last_date: latest : entry date
              lifespan : total time deployed
    Returns
    -------
    DataFrame
        DataFrame containing all site statistics.
    """
    print("Get stats")
    return _stats_df()


def get_stats(site_uid):
    """Get statistics on a site. Currently have
                column : description
      nitrate_sparsity : % rows with a non-nan nitrate_con entry
            first_date : earliest entry date
     last_date: latest : entry date
              lifespan : total time deployed

    Parameters
    ----------
    site_uid : str
        The unique identifier of the site

    Returns
    -------
    DataFrame
        One row dataframe with the site statistics
    """
    return _stats_df()[_stats_df()["site_uid"] == site_uid]


def make_site_timeseries_plot(site_uid=None, df=None, value_col="nitrate_con", interval="1D", agg_func="mean"):
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

    if df is None:
        if site_uid is None:
            raise ValueError("provide either df or site_uid")

        agg_df = aggregate_by_interval(site_uid=site_uid, value_col=value_col, interval=interval, agg_func=agg_func)
    else:
        agg_df = df

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
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.18,
        showarrow=False,
        font=dict(size=11, color="#555"),
        xanchor="center",
    )
    fig.update_layout(margin=dict(l=40, r=10, t=20, b=50))
    return fig


