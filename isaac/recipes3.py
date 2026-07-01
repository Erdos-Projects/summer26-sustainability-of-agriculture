import sys

sys.path.insert(0, "../")

from data.features import (
    agg_crops,
    agg_surplus,
    agg_weather_w_lag,
    daily_nitrate,
    nitrate_avg_except_this,
    rolling_nitrate_avg_except_this,
    doy_climatology_pure_signal,
    site_static,
    nitrate_daily_rolling,
    nitrate_violations_rolling,
    nitrate_anomaly_z,
    nitrate_spike,
    lagged_sensor_nitrate,
)
from data.transforms import flatten_buckets, merge_on_date
from functools import lru_cache

# Best feature-construction geometry from the exp10 grid search: bucket edge (m) / water
# travel velocity (m/s) / crop-surplus exp-decay length (m). REG and CLF picked different optima.
REG_EDGE, REG_VEL, REG_LAM = 50_000, 2.1, 100_000  # exp10 best lofo_r2:  e50k_v2.1_l100k
CLF_EDGE, CLF_VEL, CLF_LAM = 5_000, 2.1, 20_000  # exp10 best lofo_auc: e5k_v2.1_l20k

# Weather rolling-average ladder: a target window W gets a trailing mean over every k in this
# ladder with k <= W. k=1 ("today") is already the daily weather block, so only k>1 add
# columns -- e.g. W=7 -> roll3d + roll7d, W=3 -> roll3d, W=1 -> none.
ROLL_WEATHER_LADDER = (1, 3, 7, 14, 30, 60)


def _add_static(site, d):
    for k, v in site_static(site).items():
        d[k] = v
    return d


def _weather_windows(window):
    return [k for k in ROLL_WEATHER_LADDER if k <= window]


def _rolling_weather(wb, windows):
    """Trailing k-day means of every weather column for each k>1 in `windows`, one date-keyed
    frame per k with columns suffixed _rollKd. The k=1 case is the daily weather (wb) itself,
    so it is never duplicated here."""
    w = wb.sort_values("date").reset_index(drop=True)
    cols = [c for c in w.columns if c != "date"]
    frames = []
    for k in windows:
        if k <= 1:
            continue
        r = w[cols].rolling(k, min_periods=1).mean()
        r.columns = [f"{c}_roll{k}d" for c in cols]
        r["date"] = w["date"]
        frames.append(r)
    return frames


@lru_cache(maxsize=None)
def _agg_block_cached(site, edge, vel, lam):
    """The expensive, window-INDEPENDENT spatial aggregations (weather-with-lag, exp-decay crop,
    exp-decay surplus), memoized per (site, edge, vel, lam) -- the args are scalars so the call
    is hashable (unlike the raw agg_* which take an unhashable `edges` list). window/min_obs
    never touch these, so a window x min_obs sweep reuses one computation per site across all its
    recipes instead of rebuilding it each time. The returned frames are READ-ONLY -- callers copy
    (_rolling_weather, merge_on_date never mutate their inputs), matching the shared-frame pattern
    used elsewhere."""
    wb = flatten_buckets(agg_weather_w_lag(site, edges=[edge], exp=False, water_velocity=vel))
    cb = flatten_buckets(agg_crops(site, edges=[edge], lam=lam, exp=True))
    sb = flatten_buckets(agg_surplus(site, edges=[edge], lam=lam, exp=True))
    return wb, cb, sb


@lru_cache(maxsize=None)
def _cross_site_nitrate_cached(site):
    """Window- AND geometry-independent cross-site nitrate features: the neighbour-average lags
    (shifts 1/2/3/5) and the rolling neighbour averages (7/14/30/60d). Memoized per site (they
    depend only on the site, not on edge/vel/lam/window), so a sweep computes them once per site
    and shares them across both tasks. Read-only, like _agg_block_cached."""
    lagged_avgs = tuple(nitrate_avg_except_this(site, shift=k) for k in (1, 2, 3, 5))
    rolling_avg_not_this = rolling_nitrate_avg_except_this(site, windows=(7, 14, 30, 60))
    return lagged_avgs, rolling_avg_not_this


def _covariate_block(site, n, edge, vel, lam, window, roll_nitrate_windows=(7, 14, 30, 60)):
    """The feature scaffold: lagged whole-basin weather, exp-decay crop and surplus aggregations
    (memoized via _agg_block_cached), the pure calendar signal, the cross-site nitrate lags, and
    the window-scaled rolling weather (see ROLL_WEATHER_LADDER). Returns a fresh list each call.

    `roll_nitrate_windows` picks which rolling cross-site nitrate windows to append (a subset of
    the cached 7/14/30/60d set; () to omit them). Per the experiment audit these help REG -- with
    the gain concentrated in 7d -- but HURT CLF (recipe_CLF1 without 0.824 > recipe_CLF1.1 with
    0.817), so REG keeps {7} and CLF omits them."""
    wb, cb, sb = _agg_block_cached(site, edge, vel, lam)
    lagged_avgs, roll_n_all = _cross_site_nitrate_cached(site)
    doy = doy_climatology_pure_signal(n)
    feats = [wb, cb, sb, doy, *lagged_avgs]
    if roll_nitrate_windows:
        feats.append(roll_n_all[[f"roll_n_avg_except_this{w}d" for w in roll_nitrate_windows]])
    feats += _rolling_weather(wb, _weather_windows(window))
    return feats


def _best_features_REG(site, n, window=1):
    """Best known REG feature list (no target). Rolling cross-site nitrate trimmed to the 7d window
    (REG1.1 importance concentrates there; 14/30/60d were near-zero)."""
    return _covariate_block(site, n, REG_EDGE, REG_VEL, REG_LAM, window, roll_nitrate_windows=(7,))


def _best_features_CLF(site, n, window=1, roll_nitrate_windows=()):
    """Best known CLF feature list (no target). Same scaffold as REG, different basin geometry; the
    rolling cross-site nitrate windows are omitted by default (they hurt CLF -- see audit)."""
    return _covariate_block(site, n, CLF_EDGE, CLF_VEL, CLF_LAM, window, roll_nitrate_windows=roll_nitrate_windows)


def _target_maker(site, task="reg", window=1, min_obs=1):
    n = daily_nitrate(site).rename("nitrate_con")
    if task == "reg":
        feat = _best_features_REG(site, n, window)
        target = nitrate_daily_rolling(site, window=window, min_obs=min_obs).rename("nitrate_con")
    elif task == "clf":
        feat = _best_features_CLF(site, n, window)
        target = nitrate_violations_rolling(site, window=window, min_obs=min_obs).rename("violation")
    else:
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")
    return _add_static(site, merge_on_date([target, *feat], spine=n.index))


# BEST PARAMETERS WINDOW=1, MIN_OBS=1.
# THESE PARAMETERS FROM OLD EXPERIMENT (13)
# KEPT FOR BACKWARDS COMPATABILITY
def recipe_maker(task, window=1, min_obs=1):
    if task not in ("reg", "clf"):
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")

    def recipe(site):
        return _target_maker(site, task=task, window=window, min_obs=min_obs)

    return recipe


# default is window=1, min_obs=1
# this is also the best version of the recipe to date
def recipe_REG(site, window=1, min_obs=1):
    return _target_maker(site, task="reg", window=window, min_obs=min_obs)


# default is window=1, min_obs=1
# the default is the best version of the recipe to date
def recipe_CLF(site, window=1, min_obs=1):
    return _target_maker(site, task="clf", window=window, min_obs=min_obs)


# ── spike recipes: target = deviation from the site's own trailing baseline ──────────────────
SPIKE_WINDOW, SPIKE_K, SPIKE_MIN_OBS = 21, 2.0, 5  # baseline window (days), z-threshold, min baseline obs


SPIKE_AR_LAGS = (1, 2, 3, 7, 14)  # own-nitrate autoregression lags for the gauged-site spike variant


def _spike_target_maker(site, task="reg", window=SPIKE_WINDOW, k=SPIKE_K, min_obs=SPIKE_MIN_OBS, own_ar_lags=()):
    """Same feature scaffold as recipe_REG/_CLF (at window=1), but the target is a deviation from
    the site's OWN trailing rolling-mean baseline: REG -> the continuous standardized anomaly z(t),
    CLF -> the binary spike (z >= k). Site-relative by construction, so it targets within-site
    dynamics rather than the between-site level the rolling-max / violation targets struggle with.

    `own_ar_lags` appends the site's OWN past nitrate at those day-lags (autoregression). Empty by
    default (recipe_SPIKE, transfer-safe); recipe_SPIKE_AR turns it on for gauged-site modelling."""
    n = daily_nitrate(site).rename("nitrate_con")
    if task == "reg":
        feat = _best_features_REG(site, n)
        target = nitrate_anomaly_z(site, window=window, min_obs=min_obs).rename("nitrate_con")
    elif task == "clf":
        feat = _best_features_CLF(site, n)
        target = nitrate_spike(site, window=window, k=k, min_obs=min_obs).rename("violation")
    else:
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")
    if own_ar_lags:  # the site's own past nitrate -- causal under chronological CV, gauged-site only
        feat = [*feat, *(lagged_sensor_nitrate([site], shift=L) for L in own_ar_lags)]
    return _add_static(site, merge_on_date([target, *feat], spine=n.index))


def recipe_SPIKE(task, window=SPIKE_WINDOW, k=SPIKE_K, min_obs=SPIKE_MIN_OBS):
    """Factory (mirrors recipe_maker): returns a site -> frame recipe with the spike target baked
    in. task='reg' -> z(t) anomaly target; task='clf' -> binary spike target (k is the z-threshold,
    unused for reg since z is continuous). No own-history -> transfer-safe (see recipe_SPIKE_AR)."""
    if task not in ("reg", "clf"):
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")

    def recipe(site):
        return _spike_target_maker(site, task=task, window=window, k=k, min_obs=min_obs)

    return recipe


def recipe_SPIKE_AR(task, window=SPIKE_WINDOW, k=SPIKE_K, min_obs=SPIKE_MIN_OBS, own_ar_lags=SPIKE_AR_LAGS):
    """recipe_SPIKE PLUS the site's own past nitrate (autoregression at `own_ar_lags`). For
    GAUGED-site / individual modelling (evaluate with cook_one / compare_fleet, chronological CV):
    own history is the strongest signal (cf. exp7 recipe_B, median R2 ~0.88), but it assumes a
    sensor at the site, so -- unlike recipe_SPIKE -- it does NOT transfer to ungauged virtual sites."""
    if task not in ("reg", "clf"):
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")

    def recipe(site):
        return _spike_target_maker(site, task=task, window=window, k=k, min_obs=min_obs, own_ar_lags=own_ar_lags)

    return recipe
