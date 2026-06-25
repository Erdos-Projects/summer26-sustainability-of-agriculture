import pandas as pd
import numpy as np
from pathlib import Path

from .access import get_data, get_site_ids, get_grid

_THIS_DIR = Path(__file__).resolve().parent
_CACHE = _THIS_DIR / "_cache"


# Fixed absolute distance-bucket edges (metres). Fixed (not per-basin quantile)
# so "near"/"mid"/"far" mean the same physical distance band in every basin and
# the learned response transfers across sites. Default: <=50 km, 50-150 km, >150 km.
_DEFAULT_DIST_EDGES_M = (50_000, 150_000)


def _bucket_map(site_uid, edges=_DEFAULT_DIST_EDGES_M):
    """f: node_id -> dist_bucket, from the grid's dist_to_sensor (fixed bins)."""
    grid = get_grid(site_uid=site_uid)
    e = [-np.inf, *edges, np.inf]
    b = pd.cut(grid["dist_to_sensor"], bins=e, labels=list(range(len(edges) + 1)))
    return pd.Series(b.values, index=grid["node_id"], name="bucket")


def _bucket_lags(site_uid, v=1.0, edges=_DEFAULT_DIST_EDGES_M, max_lag_days=None):
    grid = get_grid(site_uid)
    b = grid["node_id"].map(_bucket_map(site_uid, edges))
    med = grid["dist_to_sensor"].groupby(b).median()
    lag = (med / (v * 86400)).round().astype(int)  # days = metres / (m/s * s/day)
    return lag.clip(upper=max_lag_days) if max_lag_days else lag  # Series: bucket -> lag_days


def _lag_weather_buckets(weather_b, lags, cols=None, date_col="date", bucket_col="bucket"):
    if cols is None:
        cols = [c for c in weather_b.columns if c not in (date_col, bucket_col)]
    parts = []
    for b, sub in weather_b.groupby(bucket_col, observed=True):
        sub = sub.sort_values(date_col).set_index(date_col).asfreq("D")
        sub[cols] = sub[cols].shift(int(lags.get(b, 0)))  # row t <- value from t-lag
        sub[bucket_col] = b
        parts.append(sub.reset_index())
    return pd.concat(parts, ignore_index=True)


def _agg_by_bucket(df, mapping, keys, col_agg):
    # index by node_id so each group's value Series carries node_id as its index;
    # that lets a weighted aggregator (_area_mean_curry) pull the matching per-cell
    # weights via values.index.
    x = df.set_index("node_id")
    x["bucket"] = x.index.map(mapping)
    g = x.groupby([*keys, "bucket"], observed=True)

    out = {}
    for col, how in col_agg.items():
        out[col] = g[col].agg(how)
    return pd.DataFrame(out).reset_index()


def _area_mean_curry(site_uid):
    grid = get_data(site_uid=site_uid).grid
    area_ha = pd.Series((grid.cell_area * grid.frac_cell_in_basin / 1e4).values, index=grid.node_id)

    def _func(values):
        # values is one group's Series indexed by node_id -> weight by those cells
        return float(np.average(values, weights=area_ha.loc[values.index]))

    return _func


def _get_agg_dicts(site_uid):
    func = _area_mean_curry(site_uid=site_uid)
    crop_agg_dict = {
        "Alfalfa": sum,
        "Corn": sum,
        "Fallow": sum,
        "Hay_Pasture": sum,
        "Nonag": sum,
        "Other": sum,
        "Small_Grains": sum,
        "Soybeans": sum,
    }

    surplus_agg_dict = {"total_kg_N": sum, "surplus_kgha": func}

    weather_agg_dict = {
        "precip_in_1d": func,
        "max_temp": func,
        "min_temp": func,
        "max_rel_humidity": func,
        "min_rel_humidity": func,
        "vpd": func,
        "solar_rad": func,
        "evapotranspiration": func,
        "fuel_moisture_1000h": func,
    }

    return (crop_agg_dict, surplus_agg_dict, weather_agg_dict)


def agg_crops_by_bucket(site_uid, edges=_DEFAULT_DIST_EDGES_M):
    d = get_data(site_uid=site_uid)
    f = _bucket_map(site_uid, edges)
    c_dict, _, _ = _get_agg_dicts(site_uid=site_uid)
    crops_b = _agg_by_bucket(d.crops, f, keys=["year"], col_agg=c_dict)
    return crops_b


def agg_surplus_by_bucket(site_uid, edges=_DEFAULT_DIST_EDGES_M):
    d = get_data(site_uid=site_uid)
    f = _bucket_map(site_uid, edges)
    _, s_dict, _ = _get_agg_dicts(site_uid=site_uid)
    surplus_b = _agg_by_bucket(d.surplus, f, keys=["year"], col_agg=s_dict)
    return surplus_b


def agg_weather_by_bucket(site_uid, edges=_DEFAULT_DIST_EDGES_M):
    d = get_data(site_uid=site_uid)
    f = _bucket_map(site_uid, edges)
    _, _, w_dict = _get_agg_dicts(site_uid=site_uid)
    weather_b = _agg_by_bucket(d.weather, f, keys=["date"], col_agg=w_dict)
    return weather_b


def agg_weather_by_bucket_w_lag(site_uid, edges=_DEFAULT_DIST_EDGES_M, water_velocity=1.0, max_lag_days=None):
    wb = agg_weather_by_bucket(site_uid=site_uid, edges=edges)
    lags = _bucket_lags(site_uid=site_uid, v=water_velocity, edges=edges, max_lag_days=max_lag_days)
    return _lag_weather_buckets(weather_b=wb, lags=lags)


def daily_nitrate(site_uid, agg_meth="max"):
    nitrate = get_data(site_uid=site_uid).water["nitrate_con"]
    nitrate.index = nitrate.index.tz_localize(None)
    return nitrate.resample("1D").agg(agg_meth)


def flatten_buckets(cb, sb, wb, fill=False):
    # weather is daily (date, bucket); crops/surplus are annual (year, bucket).
    # broadcast the annual frames onto each daily weather row by (year, bucket).
    wb["year"] = wb["date"].dt.year
    for fr in (cb, sb, wb):
        fr["bucket"] = fr["bucket"].astype(int)  # consistent merge-key dtype

    merged = wb.merge(cb, on=["year", "bucket"], how="left").merge(sb, on=["year", "bucket"], how="left")

    # The annual crops/surplus columns are NaN for years outside their coverage
    # (e.g. surplus ends 2017). fill=True carries each bucket's last known year
    # forward in time; fill=False leaves those rows NaN.
    if fill:
        annual_cols = [c for c in (*cb.columns, *sb.columns) if c not in ("year", "bucket")]
        merged = merged.sort_values(["bucket", "date"]).reset_index(drop=True)
        merged[annual_cols] = merged.groupby("bucket", observed=True)[annual_cols].ffill()

    # flatten the bucket dimension into the column names: one row per date, with
    # each feature expanded per bucket (e.g. precip_in_1d_b0 / _b1 / _b2).
    value_cols = [c for c in merged.columns if c not in ("date", "bucket", "year")]
    wide = merged.pivot(index="date", columns="bucket", values=value_cols)
    wide.columns = [f"{col}_b{int(b)}" for col, b in wide.columns]
    wide = wide.reset_index()
    return wide


# ── cross-site nitrate climatology (cached in data/_cache) ────────────────────


def _cache_series(name, compute, force=False):
    """Load a cached Series from data/_cache, computing + storing it if absent."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{name}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path).iloc[:, 0]
    s = compute()
    s.to_frame().to_parquet(path)
    return s


def _state_daily_base(force=False):
    """Cross-site mean nitrate_con per calendar day -- the base for all climatologies.

    Each site's nitrate is resampled to a daily mean; the per-site daily series are
    aligned and averaged across sites (skipping missing). Naive daily DatetimeIndex.
    Cached so the all-sites load happens once.
    """

    def compute():
        frames = []
        for s in get_site_ids():
            try:
                n = get_data(s).water["nitrate_con"]
            except (FileNotFoundError, KeyError):
                continue
            if n is None or n.dropna().empty:
                continue
            frames.append(n.resample("1D").mean().rename(s))
        base = pd.concat(frames, axis=1).mean(axis=1).rename("nitrate_con")
        if base.index.tz is not None:
            base.index = base.index.tz_localize(None)
        base.index.name = "date"
        return base

    return _cache_series("nitrate_state_daily", compute, force=force)


_CAL_RULE = {"D": "1D", "W": "1W", "M": "1MS"}


def nitrate_avg_calendar(freq="D", force=False):
    """Cross-site average nitrate_con over the calendar timeline.

    freq: "D" daily, "W" weekly, "M" monthly. Returns a Series indexed by period
    start date. freq="D" is the cross-site daily mean itself, so it reuses the
    base cache (nitrate_state_daily) rather than storing a duplicate; "W"/"M" are
    cached per freq.
    """
    if freq == "D":
        return _state_daily_base(force=force)

    def compute():
        return _state_daily_base(force=force).resample(_CAL_RULE[freq]).mean().rename("nitrate_con")

    return _cache_series(f"nitrate_calendar_{freq}", compute, force=force)


_SEASON_KEY = {
    "D": ("doy", lambda idx: idx.dayofyear),
    "W": ("week", lambda idx: idx.isocalendar().week.to_numpy()),
    "M": ("month", lambda idx: idx.month),
}


def nitrate_avg_seasonal(freq="D", force=False):
    """Cross-site average nitrate_con by seasonal position (collapsing years).

    freq: "D" day-of-year (1-366), "W" week-of-year (1-53), "M" month (1-12).
    Returns a Series indexed by that position -- the typical seasonal cycle of the
    cross-site average. Cached per freq.
    """
    name, keyfn = _SEASON_KEY[freq]

    def compute():
        base = _state_daily_base(force=force)
        s = base.groupby(keyfn(base.index)).mean().rename("nitrate_con")
        s.index.name = name
        return s

    return _cache_series(f"nitrate_seasonal_{freq}", compute, force=force)


def nitrate_rolling(window="31D", center=True, force=False):
    """Centered rolling average of the cross-site daily nitrate (nitrate_state_daily).

    Smooths the daily cross-site mean over the calendar timeline with a centered
    window (default 31 days), giving a slow-varying nitrate baseline that keeps
    both seasonal and year-to-year movement but removes day-to-day noise. The base
    is put on a complete daily index first, so an offset like "31D" is exactly
    `window` calendar days and `center=True` is well defined even across gaps.
    Indexed by date -- join on date. Cached per (window, center).
    """

    def compute():
        base = _state_daily_base(force=force).asfreq("D")  # regular daily index
        w = pd.Timedelta(window).days if isinstance(window, str) else window
        return base.rolling(w, center=center, min_periods=1).mean().rename("nitrate_con")

    return _cache_series(f"nitrate_rolling_{window}_c{int(center)}", compute, force=force)


def doy_climatology_pure_signal(s):
    """Cyclical (Fourier) calendar encodings of day-of-year.

    Returns sin/cos of the day-of-year angle, which encode where in the annual
    cycle each timestamp falls in a smooth, wrap-around way (Dec 31 ~ Jan 1).
    Unlike `doy_climatology`, this uses *only the dates* -- never the nitrate
    values -- so it is completely leakage-free; the model learns the seasonal
    shape from these two features itself.
    """
    doy = s.index.dayofyear
    return pd.DataFrame(
        {
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=s.index,
    )


def merge_on_dates(frames, how="outer", index=None):
    """Column-wise merge of date-indexed frames/series, aligned on their index.

    Each item in `frames` must be indexed by date (a DatetimeIndex); a Series
    contributes one column (its name). Alignment is on the date index, so values
    land on matching dates regardless of input order or length.

    Parameters
    ----------
    frames : iterable of DataFrame | Series
        The date-indexed objects to merge. Indexes must share tz-awareness
        (all naive or all tz-aware) to align.
    how : {"outer", "inner"}
        "outer" (default) keeps the union of all dates (missing -> NaN); "inner"
        keeps only dates present in every frame.
    index : DatetimeIndex, optional
        If given, the result is reindexed to it (a left-join onto a chosen spine,
        e.g. a site's dates); otherwise the merged index is returned sorted.

    Returns
    -------
    DataFrame indexed by date.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("frames is empty")

    merged = pd.concat(frames, axis=1, join=how)
    merged = merged.reindex(index) if index is not None else merged.sort_index()

    dups = merged.columns[merged.columns.duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"duplicate columns after merge: {dups} — rename before merging")
    return merged


# seasonal-position extractors: how to turn a date into the index a seasonal
# series is keyed on (matches nitrate_avg_seasonal's "D"/"W"/"M").
_SEASON_POS = {
    "doy": lambda d: d.dt.dayofyear,
    "week": lambda d: d.dt.isocalendar().week.astype("int64"),
    "month": lambda d: d.dt.month,
}
_SEASON_ALIAS = {"D": "doy", "W": "week", "M": "month"}


def match_seasonal(dates, seasonal, freq=None):
    """Map a seasonal Series (indexed by doy / week / month) onto a column of dates.

    `seasonal` is a lookup keyed on a seasonal position -- day-of-year, ISO
    week-of-year, or month (e.g. from nitrate_avg_seasonal). This computes that
    position from each date and looks it up, returning one value per date.

    Parameters
    ----------
    dates : Series | array-like | DatetimeIndex
        Reference dates (e.g. a DataFrame's "date" column). If a Series, the
        result keeps its index so it can be assigned straight back onto the frame.
    seasonal : Series
        Seasonal lookup indexed by position. Its index name ("doy"/"week"/"month")
        selects the position automatically unless `freq` is given.
    freq : optional
        Override the position: "D"/"doy", "W"/"week", or "M"/"month". Needed only
        if `seasonal.index.name` is missing.

    Returns
    -------
    Series of matched values aligned to `dates` (named like `seasonal`).
    """
    key = _SEASON_ALIAS.get(freq, freq) or seasonal.index.name
    if key not in _SEASON_POS:
        raise ValueError(
            f"could not determine seasonal position (freq={freq!r}, "
            f"seasonal.index.name={seasonal.index.name!r}); pass freq as one of "
            f"D/W/M (or doy/week/month)"
        )
    dt = pd.to_datetime(dates)
    if isinstance(dt, pd.DatetimeIndex):
        dt = pd.Series(dt, index=dt)
    pos = _SEASON_POS[key](dt)
    return pos.map(seasonal).rename(seasonal.name)


# hyperparameters
EDGES_M = (50_000, 150_000)
VEL = 1.0
WINDOW = "31D"
CENTER_WINDOW = False


def site_df(uid, edges=EDGES_M, velocity=VEL, roll_window=WINDOW, center_roll=CENTER_WINDOW):
    # first turn node_id into distance buckets
    # (0 = near, 1 = med, 2 = far) defined by EDGES_M, then bucket + lag weather
    cb = agg_crops_by_bucket(uid, edges=edges)
    sb = agg_surplus_by_bucket(uid, edges=edges)
    wb = agg_weather_by_bucket_w_lag(uid, edges=edges, water_velocity=velocity)

    # target: daily nitrate (tz-naive); its dates define the rows
    n_daily = daily_nitrate(site_uid=uid).rename("nitrate_con")
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

    # merge all date-based features (EXCEPT weather, crucially), restricted to the
    # site's nitrate dates
    time_df = merge_on_dates(
        [n_daily, n_rolling, n_cal_D, n_cal_W, n_cal_M, pure_signal, n_doy, n_woy, n_moy],
        index=dates,
    )

    # attach the bucketed weather + crops/surplus by date (once per row, not per
    # bucket); inner join drops the tail nitrate dates that have no weather
    # (gridMET ends ~a month before the present, so the most recent days are cut).
    wide = flatten_buckets(cb, sb, wb)
    out = time_df.merge(wide, left_index=True, right_on="date", how="inner")

    out.insert(0, "site_uid", uid)  # (site_uid, date) is the row key
    return out
