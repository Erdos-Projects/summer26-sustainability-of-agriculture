"""Feature & target builders for the nitrate models.

Each function turns raw site data into a model-ready column (or small frame), keyed by
date and/or year so they compose through data.transforms.merge_on_date. Quick reference:

Targets
  daily_nitrate(site)              Regression target: nitrate concentration resampled to one value per day.
  nitrate_violations(site, thr)    Classification target: 1 if the day's nitrate is at/above a threshold, else 0.

Spatial covariate aggregates (per site, binned by each cell's distance to the sensor)
  agg_crops(site)                  Annual crop-area land use, aggregated to (year, distance-bucket).
  agg_surplus(site)                Annual nitrogen surplus, aggregated to (year, distance-bucket).
  agg_weather(site)                Daily weather (precip/temp/humidity/...), aggregated to (date, distance-bucket).
  agg_weather_w_lag(site)          agg_weather with each bucket shifted back by its water travel-time lag.
  agg_site_to_buckets(site)        Convenience bundle: (crops, surplus, weather) bucketed for one site.

Static & neighbour features
  site_static(site)                Time-invariant site descriptors: sensor lat/lon, log basin area, cell-distance spread.
  lagged_sensor_nitrate(uids, k)   Past daily nitrate of the given site(s), shifted k days back (own history or neighbours).

Cross-site nitrate climatology (cached; all built from the all-sites daily mean)
  nitrate_avg_calendar(freq)       Causal cross-site daily-mean nitrate, lagged one day (yesterday's basin-wide level).
  nitrate_avg_seasonal(freq)       Typical seasonal cycle: cross-site mean nitrate by day-of-year / week / month.
  nitrate_rolling(window)          Smoothed recent cross-site level: trailing rolling mean that excludes today.
  doy_climatology_pure_signal(s)   Leakage-free seasonality: sin/cos Fourier encoding of day-of-year (uses dates only).

Most cross-site series share _state_daily_base() (the all-sites daily-mean nitrate) and are cached in data/_cache.
"""

import pandas as pd
import numpy as np
from functools import lru_cache
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
    """Annual crop areas aggregated to (year, bucket). `exp` -> distance-decay weights, else sum."""
    if exp == False:
        c_dict, _, _ = _standard_agg_dicts(site_uid=site_uid)
    else:
        c_dict, _, _ = _exp_decay_agg_dicts(site_uid=site_uid, lam=lam, normalize=normalize)

    d = get_data(site_uid=site_uid)
    mapping = _bucket_map(site_uid, edges)
    crops_b = agg_grid_to_buckets(d.crops, mapping, keys=["year"], col_agg=c_dict)
    return crops_b


def agg_surplus(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, exp=False):
    """Annual N surplus aggregated to (year, bucket). `exp` -> distance-decay weights, else sum."""
    if exp == False:
        _, s_dict, _ = _standard_agg_dicts(site_uid=site_uid)
    else:
        _, s_dict, _ = _exp_decay_agg_dicts(site_uid=site_uid, lam=lam, normalize=normalize)

    d = get_data(site_uid=site_uid)
    mapping = _bucket_map(site_uid, edges)
    surplus_b = agg_grid_to_buckets(d.surplus, mapping, keys=["year"], col_agg=s_dict)
    return surplus_b


def agg_weather(site_uid, edges=_DEFAULT_DIST_EDGES_M, lam=10_000, normalize=False, exp=False):
    """Daily weather aggregated to (date, bucket) by area-weighted mean (or exp-decay if `exp`)."""
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
    """`agg_weather` with each bucket shifted back by its travel-time lag (see bucket_lags)."""
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


@lru_cache(maxsize=256)
def daily_nitrate(site_uid, agg_meth="max"):
    """Target: nitrate_con resampled to one value per day (default daily max), tz-naive.

    Memoized per (site_uid, agg_meth); the returned Series is shared, so treat it as
    read-only (copy before mutating).
    """
    nitrate = get_data(site_uid=site_uid).water["nitrate_con"]
    if nitrate.index.tz is not None:
        nitrate = nitrate.tz_localize(None)  # new Series -- never mutate the cached frame
    return nitrate.resample("1D").agg(agg_meth)


def nitrate_violations(site_uid, threshold=10, agg_meth="max"):
    """Binary target: 1 if the day's nitrate is at or above `threshold` (default 10
    mg/L, the drinking-water limit), else 0.

    Built from daily_nitrate (default daily max, so any exceedance during the day
    counts). Days with no observation stay NA (unknown, not 0) so they can be dropped
    before training just like the regression target.
    """
    daily = daily_nitrate(site_uid, agg_meth=agg_meth)
    viol = (daily >= threshold).astype("Int8")
    viol[daily.isna()] = pd.NA
    return viol.rename("nitrate_violations")


def lagged_sensor_nitrate(site_uids, shift, agg_meth="max"):
    """Daily nitrate of each site in `site_uids`, shifted `shift` days into the past.

    For each uid, daily_nitrate is put on a regular daily index and shifted so the
    value on date t is that sensor's reading from `shift` days earlier (t - shift) --
    only past values are exposed. Returns a date-indexed DataFrame with one column
    per site, named "{uid}_lag{shift}".

    Use it for a sensor's own history (the sensor sees its past values) or for
    neighbouring sensors. Sites are aligned on the union of dates (missing -> NaN).
    """
    cols = {}
    for uid in site_uids:
        s = daily_nitrate(uid, agg_meth=agg_meth).asfreq("D").shift(shift)
        cols[f"{uid}_lag{shift}"] = s
    return pd.concat(cols, axis=1)


@lru_cache(maxsize=None)
def site_static(site_uid):
    """Time-invariant site descriptors derived from existing data (sensor location,
    basin size, grid geometry).

    Memoized per site; the returned dict is shared, so treat it as read-only.
    Constant within a site -- they add nothing to a single-site model, but across
    sites they vary and let a pooled model place a site (ungauged-basin transfer).
    Returns a dict of scalars:
        lat, lon            -- sensor coordinates (climate / spatial gradients)
        log_basin_area      -- log10 of basin area in m^2 (spans orders of magnitude)
        mean_dist_to_sensor -- mean cell distance (basin size/travel proxy, metres)
        max_dist_to_sensor  -- basin span (metres)
    """
    d = get_data(site_uid=site_uid)
    lon, lat = d.sensor_location
    dist = d.grid["dist_to_sensor"]
    return {
        "lat": float(lat),
        "lon": float(lon),
        "log_basin_area": float(np.log10(d.basin_area)),
        "mean_dist_to_sensor": float(dist.mean()),
        "max_dist_to_sensor": float(dist.max()),
    }


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


def nitrate_avg_calendar(freq="D", force=False):
    """Cross-site average nitrate_con on the PREVIOUS day (causal daily nowcast).

    The daily cross-site mean (nitrate_state_daily), lagged one day so the value on
    date t reflects only data through t-1 -- no same-day leakage. Indexed by day.

    Only freq="D" is supported: the weekly/monthly variants were retired (a causal
    calendar-bucket average is either a per-period sawtooth or just duplicates the
    trailing `nitrate_rolling`). For a smoothed recent level use `nitrate_rolling`;
    for the seasonal cycle use `nitrate_avg_seasonal` or the doy Fourier terms.
    """
    if freq != "D":
        raise ValueError(
            f"nitrate_avg_calendar only supports freq='D' (got {freq!r}); the W/M "
            f"variants were retired -- use nitrate_rolling (recent level) or "
            f"nitrate_avg_seasonal (seasonal cycle) instead."
        )

    def compute():
        # asfreq -> regular daily index so shift(1) is exactly one calendar day
        base = _state_daily_base(force=force).asfreq("D")
        return base.shift(1).rename("nitrate_con")

    return _cache_series("nitrate_calendar_D_lag1", compute, force=force)


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
    """Rolling average of the cross-site daily nitrate (nitrate_state_daily), excluding today.

    Smooths the daily cross-site mean over the calendar timeline, giving a
    slow-varying nitrate baseline that keeps seasonal and year-to-year movement but
    removes day-to-day noise. The base is put on a complete daily index first, so an
    offset like "31D" is exactly `window` calendar days. The result is lagged one day
    so the value on date t uses only data through t-1 (the window never includes
    today). Indexed by date -- join on date. Cached per (window, center).
    """

    def compute():
        base = _state_daily_base(force=force).asfreq("D")  # regular daily index
        w = pd.Timedelta(window).days if isinstance(window, str) else window
        roll = base.rolling(w, center=center, min_periods=1).mean()
        return roll.shift(1).rename("nitrate_con")  # exclude today: value at t uses <= t-1

    return _cache_series(f"nitrate_rolling_{window}_c{int(center)}_lag1", compute, force=force)


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
