import pandas as pd
import numpy as np

from .access import get_data, get_site_ids

# After the comment "Base dataframe", you can comment out any additional features you want to remove.


def make_site_df(site):

    data = get_data(site)

    # Daily precipitation
    precip = data.rain.groupby("date", as_index=False)["precip_in_1d"].mean()

    # Daily nitrate statistics
    water = (
        data.water.resample("1D")["nitrate_con"]
        .agg(
            nitrate_con="mean",
            nitrate_max="max",
        )
        .reset_index(names="date")
        .assign(
            date=lambda x: x["date"].dt.tz_localize(None),
            violation=lambda x: (x["nitrate_max"] > 10).astype(int),
        )
    )

    # Base dataframe
    df = water.merge(precip, on="date", how="left")

    # Crop features
    crops = data.crops.groupby("year", as_index=False).sum().drop(columns="node_id", errors="ignore")

    # Surplus N features
    nitrogen_2017 = data.surplus.query("year == 2017")

    surplus_n_2017 = nitrogen_2017["surplus_kgha"].mean()

    rain = data.rain.merge(
        data.grid[["node_id", "cell_area", "geometry"]],
        on="node_id",
        how="left",
    ).merge(
        nitrogen_2017[["node_id", "surplus_kgha"]],
        on="node_id",
        how="left",
    )

    rain["rain_x_surplus"] = rain["precip_in_1d"] * rain["surplus_kgha"]

    daily_rain_x_surplus = rain.groupby("date", as_index=False)["rain_x_surplus"].mean()

    df = df.merge(daily_rain_x_surplus, on="date", how="left")

    # Rolling rainfall features
    for window in [7, 14, 30]:
        df[f"rain_{window}d"] = df["precip_in_1d"].rolling(window, min_periods=1).sum()

        df[f"rain_x_surplus_{window}d"] = df["rain_x_surplus"].rolling(window, min_periods=1).sum()

    # Yearly crop features
    df["year"] = df["date"].dt.year

    df = df.merge(
        crops,
        on="year",
        how="left",
    )

    # Site-level features
    df["surplus_n_2017"] = surplus_n_2017
    df["site_id"] = site
    df["basin_area"] = data.basin_area

    # Autoregressive features
    df["nitrate_lag1"] = df["nitrate_con"]
    df["nitrate_lag2"] = df["nitrate_con"].shift(1)
    df["nitrate_lag3"] = df["nitrate_con"].shift(2)

    # Targets
    df["nitrate_tomorrow"] = df["nitrate_con"].shift(-1)
    df["violation_tomorrow"] = df["violation"].shift(-1)

    return df


# Run make_site_df for many sites and combine into one df


def make_multi_site_df(sites=None):

    if sites is None:
        sites = get_site_ids()

    return pd.concat(
        [make_site_df(site) for site in sites],
        ignore_index=True,
    )


def normalized_doy(date_series):
    """
    Map every date to a consistent DOY in 1–365, eliminating the leap-year shift.

    In leap years, dt.dayofyear assigns March 1 = DOY 61 and December 31 = DOY 366,
    which misaligns the same calendar date across leap and non-leap years. This
    function drops Feb 29 (by returning NaN for it) and shifts all post-Feb dates
    in leap years back by 1, so March 1 is always DOY 60 and Dec 31 is always DOY 365.

    Returns a Series of integers (1–365), with NaN for Feb 29 rows.
    """
    doy = date_series.dt.dayofyear
    leap = date_series.dt.is_leap_year
    feb29 = (date_series.dt.month == 2) & (date_series.dt.day == 29)

    # In leap years, shift DOY 61+ back by 1 (Mar 1 → 60, Dec 31 → 365)
    doy = doy.where(~leap | (doy < 61), doy - 1)

    # Mark Feb 29 as NaN so it can be dropped cleanly
    return doy.where(~feb29)


def make_daily_df(
    sites=None,
    weighted_mean_cols=("nitrate_con",),
    unweighted_mean_cols=("precip_in_1d",),
    sum_cols=(),
):
    """
    Aggregate per-site data across all sites into one row per date.

    Calls make_site_df() for each site, concatenates, then aggregates by date.
    February 29th is dropped and leap years are normalized so that the same
    calendar date always maps to the same DOY (1–365) regardless of leap year.

    Parameters
    ----------
    sites : list of str, optional
        Site IDs to include. Defaults to all sites from get_site_ids().
    weighted_mean_cols : tuple of str
        Columns to average weighted by basin_area (e.g. concentrations).
        These represent load-relevant quantities where larger basins matter more.
    unweighted_mean_cols : tuple of str
        Columns to average equally across sites (e.g. precip, rates).
    sum_cols : tuple of str
        Columns to sum across sites (e.g. violation_count).

    Returns
    -------
    DataFrame with one row per date and columns:
        - all aggregated feature columns
        - violation_rate  : fraction of sites with violation == 1
        - violation_count : total number of violations across sites
        - n_sites         : number of sites reporting that date
        - doy (1–365, leap-normalized), year
    """
    if sites is None:
        sites = get_site_ids()

    # Concatenate all site-level dataframes
    all_sites = pd.concat(
        [make_site_df(site) for site in sites],
        ignore_index=True,
    )
    all_sites["date"] = pd.to_datetime(all_sites["date"])

    # Normalize DOY and drop Feb 29
    all_sites["doy"] = normalized_doy(all_sites["date"])
    all_sites = all_sites.dropna(subset=["doy"])
    all_sites["doy"] = all_sites["doy"].astype(int)

    # Weighted mean helper (weights by basin_area)
    def wmean(grp, col):
        w = grp["basin_area"]
        return (grp[col] * w).sum() / w.sum()

    records = []
    for date, grp in all_sites.groupby("date"):
        row = {"date": date, "doy": grp["doy"].iloc[0], "n_sites": len(grp)}

        # Violation aggregations (always included)
        row["violation_rate"] = grp["violation"].mean()
        row["violation_count"] = grp["violation"].sum()

        # Basin-area-weighted means
        for col in weighted_mean_cols:
            if col in grp.columns:
                row[col] = wmean(grp, col)

        # Unweighted means
        for col in unweighted_mean_cols:
            if col in grp.columns:
                row[col] = grp[col].mean()

        # Sums
        for col in sum_cols:
            if col in grp.columns:
                row[col] = grp[col].sum()

        records.append(row)

    daily_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    daily_df["year"] = daily_df["date"].dt.year

    return daily_df


"""
    Collapse a daily dataframe into a single annual cycle: one row per DOY.
 
    Averages over all years for each day-of-year, giving a clean 365-point
    seasonal profile suitable for the F-test.
 
    Parameters
    ----------
    daily_df : DataFrame
        Output of make_daily_df(). Must contain 'doy' and 'year' columns.
    mean_cols : list of str, optional
        Columns to average over years. Defaults to all numeric columns
        except doy, year, violation_count, and n_sites.
    sum_cols : tuple of str
        Columns to average (not sum — we want the typical day, not the total)
        but which represent counts; listed separately for clarity.
 
    Returns
    -------
    DataFrame with one row per DOY (1–365) and columns:
        - doy
        - all aggregated feature columns
        - violation_logit : logit-transformed violation_rate for use in OLS
        - n_years         : number of distinct years contributing to each DOY
"""


def make_doy_df(
    daily_df,
    mean_cols=None,
    sum_cols=(),
):

    # Default: average everything numeric except bookkeeping columns
    exclude = {"doy", "year", "violation_count", "n_sites"}
    if mean_cols is None:
        mean_cols = [c for c in daily_df.select_dtypes(include="number").columns if c not in exclude]

    agg_dict = {col: (col, "mean") for col in mean_cols}
    agg_dict["n_years"] = ("year", "nunique")

    # Counts: average over years (typical violations per day across years)
    for col in sum_cols:
        if col in daily_df.columns:
            agg_dict[col] = (col, "mean")

    doy_df = daily_df.groupby("doy").agg(**agg_dict).reset_index().sort_values("doy")

    # Logit-transform violation_rate for OLS (avoids [0,1] boundary issues)
    if "violation_rate" in doy_df.columns:
        eps = 1e-4
        rate = doy_df["violation_rate"].clip(eps, 1 - eps)
        doy_df["violation_logit"] = np.log(rate / (1 - rate))

    return doy_df
