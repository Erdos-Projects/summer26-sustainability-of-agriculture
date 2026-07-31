"""Feature recipes: the model-ready frames src.models.train and the deploy path consume.

Four recipes, two per task. recipe_REG / recipe_CLF are the full sets. light_REG / light_CLF are what the STATIC WIDGET can assemble in a browser at an arbitrary dropped pin.

The light pair is the light2 configuration (runs 14/16) minus Alfalfa_expT. That is a HEAD-TO-HEAD result, not an inference from importances: light2 beat light3 on every REG metric (lofo_r2 0.3706 vs 0.3251, between_r2 0.3037 vs 0.2307) and on all three CLF LOFO metrics (lofo_prauc 0.7004 vs 0.6701, lofo_auc, lofo_mcc), losing only CLF's between_rate_r2 and Brier. light3 was an attempt to carry the full recipes' permutation-importance cuts across to light, and it was a net loss -- so light now keeps its own lags and its full static block, and only Alfalfa_expT is cut.

Treat the permutation-importance reasoning in this file with care generally: perm is NOT additive, so summing a correlated block's individual scores understates it badly. The weather rings correlate at r = 0.997 (b0/b1); max_temp/min_temp at 0.952; vpd/evapotranspiration at 0.898. Cuts justified by a SUM over such a block are unsound. The cuts that survive scrutiny are the ones on features that are not redundant with anything: precip_in_1d (max |r| 0.331 against any other weather variable) and Alfalfa_expT.

CURRENT FEATURE SETS
--------------------

At full bucket depth, EXCLUDING the long-run block: recipe_CLF 51, light_CLF 49, recipe_REG 49, light_REG 50. Long-run adds 9 x buckets x len(LONGRUN_STATS[task]) on top -- 27 per stat at three rings -- so the totals move whenever that constant does. "/b" marks a block that repeats per distance bucket:

    static              lat, lon, mean_dist_to_sensor    (+ log_basin_area, max_dist_to_sensor)
    calendar (4)        doy_sin, doy_cos, doy_sin2, doy_cos2
    cross-site          rest_of_state_nitrate_lag1       (+ lag2, lag3, lag5 for CLF)
    cross-site roll (1) roll_n_avg_except_this7d         (REG only)
    weather (1/b)       fuel_moisture_1000h              (bucketed in BOTH families -- see LIGHT_WEATHER_EDGES)
    crops, share (8/b)  pct_{alfalfa,corn,fallow,hay_pasture,nonag,other,small_grains,soybeans}
    crops, exp (7)      {Corn,Fallow,Hay_Pasture,Nonag,Other,Small_Grains,Soybeans}_expT<lam>
    surplus (1/b + 1)   surplus_kgha_norm per bucket, surplus_kgha_expT<lam>
    long-run (9/b/stat) pct_<class>_<stat>_b{k}, surplus_kgha_norm_<stat>_b{k}   (stat in LONGRUN_STATS)

The long-run block is the per-year crop and surplus shares reduced over features.LONGRUN_YEARS -- one scalar per site, so its width is 9 x buckets x len(LONGRUN_STATS[task]) and NOT a function of the calendar. It is the only block that sees the TIME SERIES of the CDL raster rather than one year of it, and being site-constant it cannot move within-site skill at all; it exists for the between-site metrics. It also fills a real hole: surplus_global stops at 2017 while training rows run to 2026, so the per-year surplus columns are NaN on roughly two thirds of the pooled rows and the reduced ones are not.

Dropped from BOTH families on replicated negative permutation importance: the eight weather variables other than fuel moisture (WEATHER_KEEP), Alfalfa_expT (EXPT_DROP), and rest_of_state_nitrate_lag{2,3,5} for REG (REG_LAGS). doy_cos2 is KEPT despite reading ~0.00% -- it is the other half of the semiannual pair, and the tree simply prefers splitting on sin2.

The *_expT blocks are never bucketed, in either family (see _agg_block_compute). The exp-decay tag carries its lambda, so a retuned decay length changes the column NAMES -- deploy/predict.py raises on that rather than NaN-filling it.

WATCH OUT: every recipe's column COUNT is basin-dependent, because a bucket only appears if some cell falls in it. The long-run block is bucketed too, so it varies the same way -- a basin occupying two rings emits 18 of its 27 columns per stat, not 27 with nine NaNs. deploy/predict.py handles this by reindexing to booster.feature_names and NaN-filling the absent rings -- and ONLY the absent rings; see _assert_no_skew. A per-COMID feature store does the same, emitting the full bucketed row per reach with NaN where a ring is empty, which is still one static row per reach.
"""

from src.features.features import (
    agg_crops,
    agg_crops_normalized,
    agg_surplus,
    agg_surplus_normalized,
    agg_weather_w_lag,
    daily_nitrate,
    nitrate_avg_except_this,
    rolling_nitrate_avg_except_this,
    doy_climatology_pure_signal,
    site_static,
    longrun_from_blocks,
    longrun_select,
    nitrate_daily_rolling,
    nitrate_violations_rolling,
    nitrate_anomaly_z,
    nitrate_spike,
    lagged_sensor_nitrate,
)
from src.features.transformers import flatten_buckets, merge_on_date
from functools import lru_cache
import pandas as pd

# Water travel velocity (m/s) and crop-surplus exp-decay length (m), per task. The velocity is the
# exp10 grid-search optimum; the decay length has since been retuned to a much sharper 2 km (exp10
# had picked 100 km for REG and 20 km for CLF against lofo only).
REG_VEL, REG_LAM = 2.1, 2_000
CLF_VEL, CLF_LAM = 2.1, 2_000

# Distance-bucket boundaries (m), per task.
#
# CLF's 2km riparian inner bucket comes from exp18/exp19: +0.039 lofo_prauc, the classifier's near-field flushing signal resolving violation risk. (Antecedent-precip rolling sums were tested too -- exp17 -- but only overlapped that gain, so they stayed out.)
#
# REG's 5km inner ring is a DELIBERATE TRADE AGAINST exp18, which found finer buckets overfit REG and lowered its LOFO. It is taken for between-site skill: distance rings are the only block that resolves the spatial ARRANGEMENT of land use, which is a between-site property, and REG model 13 shows the two existing rings carrying genuinely different signals (pct_corn is near-field at +0.0405/b0 vs +0.0056/b1; surplus_kgha_norm and the perennials are far-field, +0.0075..+0.0100 at b1 against roughly -0.002 at b0). Watch lofo_r2 on the retrain: exp18 predicts it falls, and if between_r2 does not rise to pay for it, revert to a single 50 km edge.
REG_EDGES = (5_000, 50_000)
CLF_EDGES = (2_000, 5_000)

# The only weather variable that survives, in EVERY recipe. Across all 8 distinct models logged -- both tasks, both aggregation schemes, both datasets -- fuel_moisture_1000h holds 4-9% of total permutation importance while the other eight sum to roughly zero (individually -0.0008..+0.0006, and precip_in_1d never above 0.0002). It is a 1000-hour (~42 day) fuel moisture, so it is heavily smoothed in space and time, which is also why it compresses to almost nothing for the browser.
WEATHER_KEEP = ("fuel_moisture_1000h",)

# Crop classes whose exp-decay column is dropped. Alfalfa_expT is negative in three of the four most
# recent runs -- the only *_expT that fails to replicate.
EXPT_DROP = ("Alfalfa",)

# Cross-site nitrate lags, per task. REG rides on lag1 (39% of total perm) plus the 7d rolling mean (10%) and gets nothing from the rest (lag2/3/5 all within +-0.35%). CLF spreads further -- lag1 33%, lag2 1.5%, lag5 0.6%, lag3 0.4-3.6% -- so it keeps the ladder.
REG_LAGS = (1,)
CLF_LAGS = (1, 2, 3, 5)

# Which long-run reductions each task consumes (features.longrun_from_blocks emits both). The block is the per-year crop/surplus shares reduced over features.LONGRUN_YEARS, broadcast alongside site_static -- one scalar per site, so it cannot move within-site skill at all and is aimed squarely at the between-site metrics.
#
# The static build packs BOTH stats regardless (build_forecast.LONGRUN_PACK_STATS), and the browser resolves columns by name against the booster's own feature list, so narrowing either tuple is a retrain rather than another reach pass.
#
# rotation_index is deliberately absent. It is measured dead in the sibling repo (exps 32/32c): +0.80 correlated with pct_corn_sd and worse on both tasks, because differencing the BASIN MEAN cancels the field-scale rotation it is meant to catch -- half the fields flipping corn->soy while the other half flip back leaves the mean almost unmoved.
LONGRUN_STATS = {"reg": ("mean",), "clf": ("mean",)}

# Weather rolling-average ladder: a target window W gets a trailing mean over every k in this
# ladder with k <= W. k=1 ("today") is already the daily weather block, so only k>1 add
# columns -- e.g. W=7 -> roll3d + roll7d, W=3 -> roll3d, W=1 -> none.
ROLL_WEATHER_LADDER = (1, 3, 7, 14, 30, 60)


def _site_kwargs(site):
    """Map a site identifier -- a site_uid (str) or a SiteData -- to the keyword the
    generalized data.features/data.transforms helpers expect (site_uid= vs site_data=)."""
    return {"site_uid": site} if isinstance(site, str) else {"site_data": site}


def _site_uid_of(site):
    """The site_uid label used for cross-site exclusion: the string itself, or a SiteData's
    .site_uid. A virtual site's uid is absent from the state, so nothing is excluded and the
    neighbour features become the full rest-of-state average."""
    return site if isinstance(site, str) else site.site_uid


def _add_static(site, d, drop=(), extra=None):
    """Append the time-invariant site descriptors, then `extra` (the long-run composition block, already narrowed to the task's stats). `drop` omits some descriptors: the light recipes cut the ones whose permutation importance went negative once the crop/surplus blocks were normalized (see LIGHT_STATIC_DROP).

    Assigned as scalars rather than merged, so merge_on_date's duplicate guard has already run by the time we get here -- a name colliding with a merged column would replace it with a constant, in silence, leaving a frame of exactly the right shape. Hence the explicit check.
    """
    scalars = {k: v for k, v in site_static(**_site_kwargs(site)).items() if k not in drop}
    scalars.update(extra or {})
    clash = sorted(set(scalars) & set(d.columns))
    if clash:
        raise ValueError(f"static block would overwrite merged column(s): {clash}")
    return d.assign(**scalars)


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


def _tag_values(frame, suffix, keep=None):
    """Rename an aggregated frame's value columns (all but year/bucket) by appending `suffix`, so a
    variant coexists with its siblings under distinct names. `keep`, if given, first restricts the
    value columns to that list (used to drop total_kg_N, keeping only surplus_kgha)."""
    struct = [c for c in ("year", "bucket") if c in frame.columns]
    val = [c for c in frame.columns if c not in struct]
    if keep is not None:
        val = [c for c in val if c in keep]
        frame = frame[struct + val]
    return frame.rename(columns={c: f"{c}{suffix}" for c in val})


def _keep_weather(wb, keep=WEATHER_KEEP):
    """Restrict a flattened weather block to `keep` (matched on prefix, so bucket suffixes survive).

    Prefix matching covers both shapes: unbucketed the frame carries bare names (fuel_moisture_1000h), and with a bucketed edge set flatten_buckets has already appended _b0/_b1.
    """
    return wb[[c for c in wb.columns if c == "date" or c.startswith(tuple(keep))]]


def _drop_expT(cb_exp, drop=EXPT_DROP):
    """Remove the exp-decay columns for `drop`'s crop classes, whatever lambda tagged them."""
    bad = tuple(f"{k}_expT" for k in drop)
    return cb_exp[[c for c in cb_exp.columns if not c.startswith(bad)]]


def _agg_block_compute(site, edges, vel, lam, weather_edges=None, weather=True):
    """The expensive, window-INDEPENDENT spatial aggregations (weather-with-lag, crop composition,
    two surplus encodings) for a site_uid OR a SiteData (via _site_kwargs). `edges` is a tuple of
    bucket boundaries (m). Returned frames are READ-ONLY -- callers copy (_rolling_weather,
    merge_on_date never mutate their inputs).

    `weather_edges` overrides `edges` for the weather block alone (None = share them). The light recipes set it to () because weather is the only block that varies by DATE: bucketing it multiplies a COMID x date series, while the crop and surplus buckets cost one extra static row per COMID. The importances back the split -- fuel_moisture_1000h scores higher unbucketed in REG (8.7% of total perm against 6.4%) and flat in CLF.
    """
    kw = _site_kwargs(site)
    e = list(edges)
    we = list(edges if weather_edges is None else weather_edges)
    KGHA = ["surplus_kgha"]

    # wb is None when the caller has no weather to aggregate. Only the static-site reach pass does that: it builds thousands of these and reconstructs its one weather column from a shared low-rank basis, so paying _weather_for_grid per basin would dominate the build for a column it discards.
    wb = flatten_buckets(agg_weather_w_lag(**kw, edges=we, exp=False, water_velocity=vel)) if weather else None

    # Crops as a COMPOSITION (pct_corn_b0, ...), not a pixel-count sum: the sum encodes basin size
    # more than land use, which is exactly the between-site confound the model struggles with.
    cb = flatten_buckets(agg_crops_normalized(**kw, edges=e))
    cb_exp = flatten_buckets(_tag_values(agg_crops(**kw, edges=(), lam=lam, exp=True), f"_expT{lam}"))

    # Two surplus encodings, kept side by side because they answer different questions: the
    # membership-weighted mean is the size-invariant intensity, the exp-decay sum retains the
    # near-field weighting. total_kg_N is dropped (via keep=) rather than carried: it is
    # surplus_kgha x area summed over the basin, i.e. a pure basin-size proxy -- reintroducing it
    # would undo the normalization.
    #
    # The two MUST carry different tags. They are distinct aggregations emitting the same base
    # column name, so a shared tag collides in merge_on_date and pandas silently renames both to
    # _x/_y -- two near-collinear copies under names nothing can select. (Upstream shipped exactly
    # that bug; see recipe_REG4.json.meta.json in the sustag repo.)
    sb = flatten_buckets(_tag_values(agg_surplus_normalized(**kw, edges=e), "_norm", keep=KGHA))
    sb_exp = flatten_buckets(_tag_values(agg_surplus(**kw, edges=(), lam=lam, exp=True), f"_expT{lam}", keep=KGHA))

    # Reduced from cb/sb rather than re-aggregated, so it costs nothing and cannot disagree with the per-year columns beside it. Both stats always; the task picks its block via LONGRUN_STATS, which therefore stays out of this function's cache key.
    lr = longrun_from_blocks(cb, sb)

    return wb, cb, cb_exp, sb, sb_exp, lr


@lru_cache(maxsize=None)
def _agg_block_cached(site, edges, vel, lam, weather_edges=None, weather=True):
    """Memoized per (site_uid, edges, vel, lam, weather_edges, weather) -- the args are hashable
    (site_uid str, edges tuple, scalars) unlike the raw agg_* which take an unhashable `edges` list.
    window/min_obs never touch these, so a window x min_obs sweep reuses one computation per site
    across all its recipes. Only reachable from the site_uid path (see _agg_block)."""
    return _agg_block_compute(site, edges, vel, lam, weather_edges, weather)


def _agg_block(site, edges, vel, lam, weather_edges=None, weather=True):
    """Dispatch: use the site_uid lru_cache when `site` is a (hashable) string, else compute
    directly for an unhashable SiteData (a virtual site is built once, so no caching needed)."""
    if isinstance(site, str):
        return _agg_block_cached(site, edges, vel, lam, weather_edges, weather)
    return _agg_block_compute(site, edges, vel, lam, weather_edges, weather)


def _cross_site_nitrate_compute(site, lags=(1, 2, 3, 5), rolls=(7, 14, 30, 60)):
    """The cross-site nitrate neighbour features (lags 1/2/3/5 + rolling 7/14/30/60d) for a
    site_uid OR SiteData. `site` is used only as the exclusion label (via _site_uid_of)."""
    uid = _site_uid_of(site)
    lagged_avgs = tuple(nitrate_avg_except_this(uid, shift=k) for k in lags)
    rolling_avg_not_this = rolling_nitrate_avg_except_this(uid, windows=(7, 14, 30, 60))
    return lagged_avgs, rolling_avg_not_this


@lru_cache(maxsize=None)
def _cross_site_nitrate_cached(site, lags=(1, 2, 3, 5), rolls=(7, 14, 30, 60)):
    """Window- AND geometry-independent cross-site nitrate features, memoized per site_uid (they
    depend only on the site, not on edge/vel/lam/window). Only reachable from the site_uid path
    (see _cross_site_nitrate). Read-only, like _agg_block_cached."""
    return _cross_site_nitrate_compute(site, lags, rolls)


def _cross_site_nitrate(site, lags=(1, 2, 3, 5), rolls=(7, 14, 30, 60)):
    """Dispatch: site_uid lru_cache when possible, else compute directly for a SiteData."""
    if isinstance(site, str):
        return _cross_site_nitrate_cached(site, lags, rolls)
    return _cross_site_nitrate_compute(site, lags, rolls)


def _covariate_block(
    site,
    n,
    edges,
    vel,
    lam,
    window,
    roll_nitrate_windows=(7, 14, 30, 60),
    lags=CLF_LAGS,
    longrun_stats=(),
    statics=None,
):
    """The feature scaffold: lagged whole-basin weather, the crop composition and both surplus
    aggregations (memoized via _agg_block), the pure calendar signal, the cross-site nitrate lags,
    and the window-scaled rolling weather (see ROLL_WEATHER_LADDER). Returns a fresh list each call.

    `site` may be a site_uid (str, cached path) or a SiteData (virtual/ungauged, cache bypassed).
    `n` is a bare date-carrier -- only n.index feeds doy (and, upstream, the merge spine); no
    nitrate values enter the features, so a waterless virtual site works with a weather spine.

    `roll_nitrate_windows` picks which rolling cross-site nitrate windows to append (a subset of
    the cached 7/14/30/60d set; () to omit them). Per the experiment audit these help REG -- with
    the gain concentrated in 7d -- but HURT CLF (recipe_CLF1 without 0.824 > recipe_CLF1.1 with
    0.817), so REG keeps {7} and CLF omits them.

    The weather and expT blocks are filtered exactly as the light recipes filter them (WEATHER_KEEP, EXPT_DROP): the evidence for both cuts came from the FULL-recipe runs, so applying it only to light would have been arbitrary. Trimming weather here removes 24 of the 27 weather columns.
    """
    wb, cb, cb_exp, sb, sb_exp, lr = _agg_block(site, edges, vel, lam)
    if statics is not None:
        statics.update(longrun_select(lr, longrun_stats))
    wb = _keep_weather(wb)
    lagged_avgs, roll_n_all = _cross_site_nitrate(site, lags=lags)
    doy = doy_climatology_pure_signal(n)
    feats = [wb, cb, _drop_expT(cb_exp), sb, sb_exp, doy, *lagged_avgs]
    if roll_nitrate_windows:
        feats.append(roll_n_all[[f"roll_n_avg_except_this{w}d" for w in roll_nitrate_windows]])
    feats += _rolling_weather(wb, _weather_windows(window))
    return feats


def _best_features_REG(site, n, window=1, statics=None):
    """Best known REG feature list (no target). Three-ring geometry (see REG_EDGES -- a deliberate
    trade against exp18); rolling cross-site nitrate trimmed to the 7d window (REG1.1 importance
    concentrates there; 14/30/60d were near-zero); cross-site lags trimmed to lag1 (2/3/5 are all
    within +-0.35% perm); no antecedent-precip (it hurts REG, exp17)."""
    return _covariate_block(
        site,
        n,
        REG_EDGES,
        REG_VEL,
        REG_LAM,
        window,
        roll_nitrate_windows=(7,),
        lags=REG_LAGS,
        longrun_stats=LONGRUN_STATS["reg"],
        statics=statics,
    )


def _best_features_CLF(site, n, window=1, roll_nitrate_windows=(), statics=None):
    """Best known CLF feature list (no target). Riparian 2km inner bucket (exp18/exp19, +0.039
    lofo_prauc -- the classifier's near-field flushing signal); rolling cross-site nitrate omitted
    by default (it hurts CLF -- see audit); the full lag ladder, which CLF unlike REG does use."""
    return _covariate_block(
        site,
        n,
        CLF_EDGES,
        CLF_VEL,
        CLF_LAM,
        window,
        roll_nitrate_windows=roll_nitrate_windows,
        lags=CLF_LAGS,
        longrun_stats=LONGRUN_STATS["clf"],
        statics=statics,
    )


def _assemble(site, task, spine, window, target=None, light=False):
    """Shared assembly for the gauged and virtual paths: build the task's feature list on the
    given `spine` (a DatetimeIndex of output rows), merge, add the static descriptors, and
    optionally prepend a `target` column.

    `n` is a bare date-carrier built from `spine` -- only its index feeds doy and the merge
    timeline (no nitrate values enter the features), so a waterless virtual site works by
    supplying a weather-derived spine and no target.

    `light` selects the browser-buildable feature set (see _light_features). It routes through the same merge and the same static block, so light and full frames stay structurally identical.
    """
    n = pd.Series(index=pd.DatetimeIndex(spine), dtype="float64")
    if task not in ("reg", "clf"):
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")
    statics = {}
    if light:
        feat = _light_features(site, n, task, statics=statics)
    elif task == "reg":
        feat = _best_features_REG(site, n, window, statics=statics)
    else:
        feat = _best_features_CLF(site, n, window, statics=statics)
    frames = feat if target is None else [target, *feat]
    drop = LIGHT_STATIC_DROP[task] if light else ()
    return _add_static(site, merge_on_date(frames, spine=n.index), drop=drop, extra=statics)


def build_feature_frame(site, task="reg", spine=None, window=1, light=False):
    """Model-ready feature frame for `site` (a site_uid OR a SiteData), WITHOUT a target.

    `spine` is the DatetimeIndex of output rows. It defaults to the site's daily-nitrate index
    (the gauged timeline, matching recipe_REG/_CLF minus the target). For an ungauged/virtual
    site (no water) pass a spine derived from the weather window -- e.g. the TARGET_YEAR daily
    dates -- and the frame is produced without ever touching water. The deploy virtual recipe
    calls this.

    `light=True` builds the light feature set instead -- the one the static widget can assemble client-side. Pair it with a model trained on light_REG / light_CLF; the column sets differ.
    """
    if spine is None:
        spine = daily_nitrate(**_site_kwargs(site)).index
    return _assemble(site, task, spine=spine, window=window, target=None, light=light)


def _target_maker(site, task="reg", window=1, min_obs=1, light=False):
    n = daily_nitrate(**_site_kwargs(site)).rename("nitrate_con")
    if task == "reg":
        target = nitrate_daily_rolling(**_site_kwargs(site), window=window, min_obs=min_obs).rename("nitrate_con")
    elif task == "clf":
        target = nitrate_violations_rolling(**_site_kwargs(site), window=window, min_obs=min_obs).rename("violation")
    else:
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")
    return _assemble(site, task, spine=n.index, window=window, target=target, light=light)


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


# ── light recipes: the feature set the static widget can build in a browser ──────────────────
# These exist for the static site, where the feature frame has to be assembled client-side at an arbitrary dropped pin. The governing fact is that snap_comid maps every possible pin onto one of ~61k NHD reaches, so ANY feature that depends only on the basin is a pure function of that COMID and can be precomputed offline for every reach (~4 MB for all of them). The same is true of the cross-site nitrate series, which is statewide and has no basin dependency at all (~0.6 MB). What is NOT free is per-basin daily weather: it is COMID x date, so it is the one block whose cost scales with both the reach count and the calendar.
#
# Everything below follows from that. The recipes keep whatever is precomputable and spend bytes only where a variable earns them.

# Bucket geometry, per task, same as the full recipes. Distance rings are NOT a cost the browser pays: bucket membership is a property of the COMID, so the crop and surplus buckets are extra columns in a static per-reach row, not extra bytes per day. Only weather is COMID x date, and it opts out below.
#
# Bucketing earns its place. The one controlled experiment in the logs (CLF models 1 -> 3, identical data, the only change being 19 added _b2 columns) improved every metric: loso_auc +0.0072, lofo_auc +0.0131, lofo_prauc +0.0092, between_rate_r2 +0.0174. And the buckets are complementary rather than redundant -- in REG model 13, pct_corn is a near-field signal (perm +0.0405 at b0 against +0.0056 at b1) while surplus_kgha_norm and the perennials are far-field (+0.0075..+0.0100 at b1 against roughly -0.002 at b0). Collapsing them costs exactly what that predicts: surplus_kgha_norm falls from 2.87% of total perm bucketed to 0.75% in the single-bucket light run.
#
# The edges are the full recipes' edges, per task: the geometry was tuned separately for REG and CLF and one shared edge set would impose CLF's near-field rings on REG.
LIGHT_EDGES = {"reg": REG_EDGES, "clf": CLF_EDGES}

# Weather stays whole-basin whatever the crops do -- see _agg_block_compute's `weather_edges`.
# Distance rings for the WEATHER block specifically -- see _agg_block_compute's `weather_edges`.
# Keyed by task, for the same reason as LIGHT_EDGES: one shared tuple hands REG the classifier's
# near-field geometry, which is not what either task's evidence points at.
#
# The original argument for holding weather at whole-basin was STORAGE -- it is the only COMID x DATE
# block, so bucketing it looked like tripling the one thing that scales with the calendar. THAT
# ARGUMENT IS DEAD. The static build ships the basin mean as a projection onto a shared low-rank
# basis (widget/static/build_forecast.py), and a bucketed block is just three linear functionals of
# the same node field: 3 x 64 coefficients per reach instead of 64, roughly +8.7 MB across all
# 16,762 reaches, shared modes unchanged. It is now purely a modelling choice.
#
# Both tasks bucket weather on their OWN rings, which is what run 14/16 (light2) used and what the
# head-to-head favours -- light2 beats light3 on every REG metric and on all three CLF LOFO metrics.
#
# The permutation-importance reading that once argued for whole-basin here does NOT survive scrutiny:
# it summed the three rings' individual importances (7.93%) against the single column's (8.89%), and
# permutation importance is not additive. The rings correlate at r = 0.997 (b0/b1) and 0.83 at worst,
# so permuting one is repaired by its neighbours and each reads far below its joint contribution.
# 7.93% is a floor, not an estimate, and the comparison cannot distinguish the two.
LIGHT_WEATHER_EDGES = {"reg": REG_EDGES, "clf": CLF_EDGES}

# Velocity and decay length are keyed by task for the same reason as the edges. They happen to be equal for both tasks today, so this is currently a no-op -- but it is the difference between light_CLF tracking CLF's geometry and silently inheriting REG's the next time the two diverge.
LIGHT_VEL = {"reg": REG_VEL, "clf": CLF_VEL}
LIGHT_LAM = {"reg": REG_LAM, "clf": CLF_LAM}

# Cross-site lags: lag1 + lag3 for both tasks, which is what light2 used.
#
# This DIVERGES from the full recipes' REG_LAGS / CLF_LAGS, and the divergence is now the evidenced
# direction rather than an oversight. The full recipes' ladders were set from per-feature permutation
# importance on full-recipe runs; the light pair has since been compared head to head, and the
# arm carrying CLF's longer ladder (light3, lags 1/2/3/5) lost 0.030 lofo_prauc and 0.039 lofo_mcc to
# the arm with {1,3}. Two extra columns against only 20 basin families is exactly where LOFO frays.
LIGHT_LAGS = {"reg": (1, 3), "clf": (1, 3)}

# The light pair's long-run block. Kept separate from LONGRUN_STATS for the same reason as every other LIGHT_* constant: the two families are tuned independently, and the light pair is the one that ships.
LIGHT_LONGRUN_STATS = {"reg": ("mean",), "clf": ("mean",)}

# Nothing is dropped from the static block. It USED to cut log_basin_area and max_dist_to_sensor on
# replicated negative permutation importance -- and that was measured against loso_r2, which is ~60%
# within-site variance, so it systematically under-weights exactly what a pure between-site feature
# contributes. The head-to-head settled it the other way: light2 keeps both and scores between_r2
# 0.3037 against light3's 0.2307, with lofo_r2 also higher. The cut was wrong; the risk flagged when
# it was made is the one that materialised.
LIGHT_STATIC_DROP = {"reg": (), "clf": ()}


def _light_features(site, n, task, statics=None):
    """The light feature scaffold: whole-basin weather (one variable), both crop encodings, both surplus encodings, the pure calendar signal, and the cross-site nitrate neighbours.

    Same blocks and the same code path as _covariate_block, so the light and full recipes cannot drift apart in how a block is computed. What is left of the difference: the static drops above, and the fact that weather opts out of the distance buckets. `window` is fixed at 1 -- the rolling-weather ladder is a no-op there, and antecedent weather would reintroduce exactly the per-basin daily storage the light set exists to avoid.
    """
    wb, cb, cb_exp, sb, sb_exp, lr = _agg_block(
        site, LIGHT_EDGES[task], LIGHT_VEL[task], LIGHT_LAM[task], weather_edges=LIGHT_WEATHER_EDGES[task]
    )
    if statics is not None:
        statics.update(longrun_select(lr, LIGHT_LONGRUN_STATS[task]))
    lagged_avgs, roll_n_all = _cross_site_nitrate(site, lags=LIGHT_LAGS[task], rolls=(7, 60))
    doy = doy_climatology_pure_signal(n)
    # Every block except wb is static per COMID, so bucketing them is free: they cost one precomputed row per reach, not bytes per day. Only the weather block is filtered and unbucketed.
    feats = [_keep_weather(wb), cb, _drop_expT(cb_exp), sb, sb_exp, doy, *lagged_avgs]
    # Mirrors the full recipes: the rolling cross-site nitrate helps REG (concentrated in 7d) but hurts CLF -- see the experiment audit referenced in _best_features_REG/_CLF.
    if task == "reg":
        feats.append(roll_n_all[["roll_n_avg_except_this7d"]])
    return feats


def light_REG(site, window=1, min_obs=1):
    """recipe_REG restricted to what a browser can build at an arbitrary pin. See the light-recipes section header."""
    return _target_maker(site, task="reg", window=window, min_obs=min_obs, light=True)


def light_CLF(site, window=1, min_obs=1):
    """recipe_CLF restricted to what a browser can build at an arbitrary pin. See the light-recipes section header."""
    return _target_maker(site, task="clf", window=window, min_obs=min_obs, light=True)


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
    statics = {}
    if task == "reg":
        feat = _best_features_REG(site, n, statics=statics)
        target = nitrate_anomaly_z(site, window=window, min_obs=min_obs).rename("nitrate_con")
    elif task == "clf":
        feat = _best_features_CLF(site, n, statics=statics)
        target = nitrate_spike(site, window=window, k=k, min_obs=min_obs).rename("violation")
    else:
        raise ValueError(f"Expected 'reg' or 'clf', got {task}")
    if own_ar_lags:  # the site's own past nitrate -- causal under chronological CV, gauged-site only
        feat = [*feat, *(lagged_sensor_nitrate([site], shift=L) for L in own_ar_lags)]
    return _add_static(site, merge_on_date([target, *feat], spine=n.index), extra=statics)


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
