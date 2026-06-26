import pandas as pd
import numpy as np
from pathlib import Path

from .access import get_data, get_site_ids
from .transforms import (
    _DEFAULT_DIST_EDGES_M,
    _VEL,
    _bucket_map,
    _standard_agg_dicts,
    _exp_decay_agg_dicts,
    agg_grid_to_buckets,
    bucket_lags,
    lag_buckets,
)

_THIS_DIR = Path(__file__).resolve().parent
_CACHE = _THIS_DIR / "_cache"


def agg_crops(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, exp=False):
    if exp == False:
        c_dict, _, _ = _standard_agg_dicts(site_uid=site_uid)
    else:
        c_dict, _, _ = _exp_decay_agg_dicts(site_uid=site_uid, lam=lam, normalize=normalize)

    d = get_data(site_uid=site_uid)
    mapping = _bucket_map(site_uid, edges)
    crops_b = agg_grid_to_buckets(d.crops, mapping, keys=["year"], col_agg=c_dict)
    return crops_b


def agg_surplus(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, exp=False):
    if exp == False:
        _, s_dict, _ = _standard_agg_dicts(site_uid=site_uid)
    else:
        _, s_dict, _ = _exp_decay_agg_dicts(site_uid=site_uid, lam=lam, normalize=normalize)

    d = get_data(site_uid=site_uid)
    mapping = _bucket_map(site_uid, edges)
    surplus_b = agg_grid_to_buckets(d.surplus, mapping, keys=["year"], col_agg=s_dict)
    return surplus_b


def agg_weather(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, exp=False):
    if exp == False:
        _, _, w_dict = _standard_agg_dicts(site_uid=site_uid)
    else:
        _, _, w_dict = _exp_decay_agg_dicts(site_uid=site_uid, lam=lam, normalize=normalize)

    d = get_data(site_uid=site_uid)
    mapping = _bucket_map(site_uid, edges)
    weather_b = agg_grid_to_buckets(d.weather, mapping, keys=["date"], col_agg=w_dict)
    return weather_b


def agg_weather_w_lag(
    site_uid,
    edges=_DEFAULT_DIST_EDGES_M,
    lam=10_000,
    normalize=False,
    exp=False,
    water_velocity=_VEL,
):
    wb = agg_weather(site_uid, edges, lam, normalize, exp)
    lags = bucket_lags(site_uid, water_velocity, edges)
    wb = lag_buckets(wb, lags, date_col="date", bucket_col="bucket")
    return wb


def agg_site_to_buckets(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, mixed=True):
    """The default site-to-bucket aggregation. Crops and surplus are aggregated with exponential decay, weather is aggregated without decay."""
    cb = agg_crops(site_uid, edges=edges, lam=lam, normalize=normalize, exp=True)
    sb = agg_surplus(site_uid, edges=edges, lam=lam, normalize=normalize, exp=True)
    wb = agg_weather(site_uid, edges=edges, lam=lam, normalize=normalize, exp=False)
    return cb, sb, wb


def daily_nitrate(site_uid, agg_meth="max"):
    nitrate = get_data(site_uid=site_uid).water["nitrate_con"]
    nitrate.index = nitrate.index.tz_localize(None)
    return nitrate.resample("1D").agg(agg_meth)


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
