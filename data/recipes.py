"""
A collection of methods which build ready-to-model dataframes from the data.
"""

from .features import (
    agg_crops,
    agg_surplus,
    agg_weather,
    agg_weather_w_lag,
    daily_nitrate,
    nitrate_rolling,
    nitrate_avg_calendar,
    nitrate_avg_seasonal,
    doy_climatology_pure_signal,
)
from .transforms import (
    match_seasonal,
    flatten_buckets,
    merge_on_date,
    bucket_lags,
    lag_buckets,
    _VEL,
    _DEFAULT_DIST_EDGES_M,
)

# hyperparameters
_WINDOW = "31D"
_LAM = 10_000
_CENTER_WINDOW = False


def isaac_df1(site_uid, edges=_DEFAULT_DIST_EDGES_M, velocity=_VEL, roll_window=_WINDOW, center_roll=_CENTER_WINDOW):
    # first turn node_id into distance buckets
    # (0 = near, 1 = med, 2 = far) defined by EDGES_M, then bucket + lag weather
    cb = agg_crops(site_uid, edges=edges)
    sb = agg_surplus(site_uid, edges=edges)
    wb = agg_weather(site_uid, edges=edges)
    lag = bucket_lags(site_uid=site_uid, water_velocity=velocity, edges=edges)
    wb_lag = lag_buckets(wb, lags=lag, date_col="date", bucket_col="bucket")

    # target: daily nitrate (tz-naive); its dates define the rows
    n_daily = daily_nitrate(site_uid=site_uid).rename("nitrate_con")
    dates = n_daily.index

    # cross-site date-keyed reference features (distinct names so they don't collide)
    n_rolling = nitrate_rolling(window=roll_window, center=center_roll).rename("nitrate_roll")
    n_cal_D = nitrate_avg_calendar(freq="D").rename("nitrate_cal_d")
    n_cal_W = nitrate_avg_calendar(freq="W").rename("nitrate_cal_w")
    n_cal_M = nitrate_avg_calendar(freq="M").rename("nitrate_cal_m")
    pure_signal = doy_climatology_pure_signal(n_daily)  # date-indexed (doy_sin/doy_cos)

    # seasonal nitrate averages mapped onto the calendar dates
    def _help(d, name):
        return match_seasonal(dates=dates, seasonal=d).rename(name)

    n_doy = _help(nitrate_avg_seasonal(freq="D"), "nitrate_doy")
    n_woy = _help(nitrate_avg_seasonal(freq="W"), "nitrate_woy")
    n_moy = _help(nitrate_avg_seasonal(freq="M"), "nitrate_moy")

    # bucketed weather/crops/surplus flattened to wide (bucket -> _b{n} columns)
    w_wide = flatten_buckets(wb_lag)
    c_wide = flatten_buckets(cb)
    s_wide = flatten_buckets(sb)

    # ONE merge, pinned to the nitrate dates (spine): the date-keyed features
    # (weather + climatologies) left-join on date, the annual crops/surplus broadcast
    # by year. The spine matters because the cross-site climatologies span the whole
    # calendar and would otherwise balloon the row set.
    out = merge_on_date(
        [n_daily, n_rolling, n_cal_D, n_cal_W, n_cal_M, pure_signal, n_doy, n_woy, n_moy, w_wide, c_wide, s_wide],
        spine=dates,
    )

    # drop the tail nitrate dates that have no weather (gridMET ends ~a month before
    # the present): those rows are NaN across every weather bucket column.
    weather_cols = [c for c in w_wide.columns if c != "date"]
    out = out.dropna(subset=weather_cols, how="all").reset_index(drop=True)

    out.insert(0, "site_uid", site_uid)  # (site_uid, date) is the row key
    return out


def preet_df1(site_uid, edges=[], velocity=_VEL, lam=_LAM, center_roll=False):
    # first turn node_id into distance buckets
    # (0 = near, 1 = med, 2 = far) defined by EDGES_M, then bucket + lag weather
    cb = agg_crops(site_uid, edges=edges, lam=lam, normalize=True, exp=True)
    sb = agg_surplus(site_uid, edges=edges, lam=lam, normalize=True, exp=True)
    wb_lag = agg_weather_w_lag(site_uid, edges, water_velocity=velocity)

    # target: daily nitrate (tz-naive); its dates define the rows
    n_daily = daily_nitrate(site_uid=site_uid).rename("nitrate_con")
    dates = n_daily.index

    # cross-site date-keyed reference features (distinct names so they don't collide)
    n_rolling_3D = nitrate_rolling(window="3D", center=center_roll).rename("nitrate_roll")
    n_rolling_7D = nitrate_rolling(window="7D", center=center_roll).rename("nitrate_roll")
    n_rolling_14D = nitrate_rolling(window="14D", center=center_roll).rename("nitrate_roll")
    n_rolling_30D = nitrate_rolling(window="30D", center=center_roll).rename("nitrate_roll")
    n_cal_D = nitrate_avg_calendar(freq="D").rename("nitrate_cal_d")
    n_cal_W = nitrate_avg_calendar(freq="W").rename("nitrate_cal_w")
    n_cal_M = nitrate_avg_calendar(freq="M").rename("nitrate_cal_m")
    pure_signal = doy_climatology_pure_signal(n_daily)  # date-indexed (doy_sin/doy_cos)

    # seasonal nitrate averages mapped onto the calendar dates
    def _help(d, name):
        return match_seasonal(dates=dates, seasonal=d).rename(name)

    n_doy = _help(nitrate_avg_seasonal(freq="D"), "nitrate_doy")
    n_woy = _help(nitrate_avg_seasonal(freq="W"), "nitrate_woy")
    n_moy = _help(nitrate_avg_seasonal(freq="M"), "nitrate_moy")

    # bucketed weather/crops/surplus flattened to wide (bucket -> _b{n} columns)
    w_wide = flatten_buckets(wb_lag)
    c_wide = flatten_buckets(cb)
    s_wide = flatten_buckets(sb)

    # ONE merge, pinned to the nitrate dates (spine): the date-keyed features
    # (weather + climatologies) left-join on date, the annual crops/surplus broadcast
    # by year. The spine matters because the cross-site climatologies span the whole
    # calendar and would otherwise balloon the row set.
    out = merge_on_date(
        [
            n_daily,
            w_wide,
            c_wide,
            s_wide,
            n_rolling_3D,
            n_rolling_14D,
            n_rolling_14D,
            n_rolling_30D,
            n_cal_D,
            n_cal_W,
            n_cal_M,
            pure_signal,
            n_doy,
            n_woy,
            n_moy,
        ],
        spine=dates,
    )

    # drop the tail nitrate dates that have no weather (gridMET ends ~a month before
    # the present): those rows are NaN across every weather bucket column.
    weather_cols = [c for c in w_wide.columns if c != "date"]
    out = out.dropna(subset=weather_cols, how="all").reset_index(drop=True)

    out.insert(0, "site_uid", site_uid)  # (site_uid, date) is the row key
    return out
