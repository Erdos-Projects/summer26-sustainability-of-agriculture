import pandas as pd
import numpy as np
from pathlib import Path

site_data = Path("../../data/IWQIS/site_clean.csv")
sites_of_interest = Path("../../data/IWQIS/sites_of_interest.csv")
measures_data = Path("../../data/IWQIS/measures.csv")
full_data = Path("../../data/IWQIS/iwqis_alldata.csv")
site_data_dir = Path("../../data/IWQIS/site_data/")
site_data_ext = "_all_data.csv"

def get_iwqis_sites():
    return pd.read_csv(sites_of_interest, engine='python', on_bad_lines='warn')

def get_full_data():
    return pd.read_csv(full_data)

def get_site_uids():
    uids = get_iwqis_sites()["uid"].unique()
    uids = [str(id).strip() for id in uids]
    return uids

def get_site_data(site_uid: str):
    """Gets all data from a specified site.

    Parameters
    ----------
    site_uid : str
        the uid of the site

    Returns
    -------
    DataFrame
        a dataframe with all data from the specified site_uid
    """
    return pd.read_csv(Path(site_data_dir, site_uid + site_data_ext))

# stands for "water quality" but is missing some
def get_site_uids():
    uids = get_iwqis_sites()["uid"].unique()
    uids = [str(id).strip() for id in uids]
    return uids

# stands for "citizen"
def get_wq_sites():
    sites = get_iwqis_sites()
    sites = sites[[s.startswith("WQ") for s in sites.uid]]
    return sites

# stands for "citizen"
def get_ctz_sites():
    sites = get_iwqis_sites()
    sites = sites[[s.startswith("CTZ") for s in sites.uid]]
    return sites

def aggregate_by_interval(df, value_col, interval, agg_func='mean'):
    """
    Aggregate a time series by a specified interval.
    
    Example
    -------
    hourly = aggregate_by_interval(site_df, 'nitrate_con', '1h')
    every_two_days = aggregate_by_interval(site_df, 'nitrate_con', '2D')
    
    Parameters
    ----------
    df : pd.DataFrame
    timestamp_col : str
        Column containing timestamps in any pandas-parseable format.
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
    timestamp_col = 'datetime'
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col)
    return df[value_col].resample(interval).agg(agg_func)

def make_site_csvs():
    print("getting full data...", end="")
    full_data = get_full_data()
    print("done.")
    uids = full_data["site_uid"].unique()
    for uid in uids:
        full_data[full_data.site_uid == uid].to_csv(site_data_dir / f"{str(uid).strip()}_all_data.csv", index=False)
        print("wrote", site_data_dir / f"{str(uid).strip()}_all_data.csv")

def main():
    make_site_csvs()
    
if __name__ == "__main__":
    main()