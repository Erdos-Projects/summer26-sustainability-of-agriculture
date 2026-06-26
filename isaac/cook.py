"""
cook.py -- reusable harness for comparing nitrate "recipes".
============================================================

A *recipe* is a function  site_uid -> DataFrame  containing a target column, a `date`
column, and feature columns (what recipe_A/B/C produce). Recipes stay pure
(site -> frame) and know nothing about CV; this module supplies the evaluation.

You tell cook WHICH column is the target and how to score it by passing `target=` and
`task=` to the cook functions (defaults: target="nitrate_con", task="reg"). cook reads
that column off the recipe's output as y and excludes it from the features. For
classification the recipe builds its own binary target column and you pass task="clf"
(scored by AUC / PR-AUC / Brier); cook never re-derives the target from a threshold.

Two evaluation modes, differing only in their LEAKAGE AXIS:

  cook_one(recipe, site)     INDIVIDUAL site modelling. Leakage axis = TIME, so CV is
                             chronological (expanding TimeSeriesSplit). Answers: can the
                             recipe model one site's dynamics? Compared vs a persistence
                             baseline (predict yesterday).
  cook_many(recipe, sites)   CROSS-SITE modelling. Leakage axis = SPACE, so CV groups by
                             site (LOSO, optimistic) and by basin conflict-component
                             (LOBO, honest -- via data/splits.py). Answers: does the
                             recipe transfer to an unseen basin?
  cook_fleet(recipe, sites)  bonus: run cook_one on each site, summarise the distribution
                             (individual modelling across the fleet).

Cross-site performance is reported as a DECOMPOSITION rather than one number:
  between_r2  predicted vs actual per-site means -- does it rank site levels?
  within_r2   after removing each site's mean    -- does it track daily movement?
  macro_r2    median per-site R2 (equal weight per site, vs row-weighted overall)
  loso_r2 / lobo_r2  the leakage bracket (lobo <= real generalisation <= loso)

Comparison is paired: GroupKFold is deterministic, so every recipe sees identical folds
for the same `sites` -- differences are attributable to features, not split noise.

Run:  python isaac/cook.py
"""

import warnings

warnings.filterwarnings("ignore")
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from data.splits import split_groups
from data.access import get_site_ids

TARGET = "nitrate_con"  # default target column if the caller doesn't pass target=
_STRUCTURAL = {"site_uid", "site", "date", "year", "datetime"}  # bookkeeping cols, never features

# fixed, somewhat regularized model config
_DEFAULT_XGB = dict(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=42,
    early_stopping_rounds=50,
)

# fast configuration
FAST_XGB = dict(n_estimators=800, learning_rate=0.05, max_depth=4)

_BASIN = None  # lazy {site -> conflict-component id} loader


# ── shared helpers ─────────────────────────
def _check_target(recipe, df, target):
    """Fail fast (and loud) if a recipe didn't actually produce the requested target."""
    if target not in df.columns:
        name = getattr(recipe, "__name__", repr(recipe))
        raise KeyError(
            f"recipe {name!r} produced columns {list(df.columns)} -- no target {target!r}; "
            f"pass the right target= to cook_one/cook_many."
        )


def _features(df, target):
    """The feature columns: everything except the target and the bookkeeping columns."""
    return [c for c in df.columns if c != target and c not in _STRUCTURAL]


def _target(df, target, task):
    """Read the target column off the recipe's output frame (int-cast for clf)."""
    y = df[target]
    return y.astype("int64") if task == "clf" else y


def _model(task, **overrides):
    """Wrapper for returning the appropriate model, Classifier vs Regressor"""
    cfg = {**_DEFAULT_XGB, **overrides}
    if task == "clf":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **cfg)
    return xgb.XGBRegressor(eval_metric="rmse", **cfg)


def _fit(task, Xtr, ytr, Xval, yval, **xgb_kw):
    """Fitter helper. Used basically so extra model keywords can be handed off."""
    m = _model(task, **xgb_kw)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return m


def _oof_predict(model, X, task):
    """Out of fold predictor helper to differentiate between the two cases"""
    return model.predict_proba(X)[:, 1] if task == "clf" else model.predict(X)


def _score(y, pred, task):
    """Pooled metrics for a set of (y, prediction)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if task == "clf":
        two = len(np.unique(y)) == 2
        return dict(
            auc=roc_auc_score(y, pred) if two else np.nan,
            prauc=average_precision_score(y, pred) if two else np.nan,
            brier=brier_score_loss(y, pred),
            base=float(y.mean()),
        )
    return dict(
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        mae=float(mean_absolute_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


def _per_site_score(y, pred, task):
    """One-site score, or NaN when undefined (one class / no variance)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if task == "clf":
        return roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan
    return r2_score(y, pred) if (len(y) > 1 and np.std(y) > 0) else np.nan


def basin_groups(site_series):
    """site_uids -> basin conflict-component ids (data/splits.py); sites missing from the
    basin graph become their own singleton group so they can't leak into a component."""
    global _BASIN
    if _BASIN is None:
        _BASIN = split_groups()
    nxt = max(_BASIN.values(), default=-1) + 1
    mapping = {}
    for s in pd.unique(site_series):
        mapping[s] = _BASIN.get(s, nxt)
        nxt += s not in _BASIN
    return site_series.map(mapping)


# ── INDIVIDUAL site: chronological CV ────────────────
def cook_one(recipe, site, target=TARGET, task="reg", n_splits=5, test_size=90, **xgb_kw):
    """Expanding-window CV for `recipe` on one site (time is the only leakage axis).

    `target`/`task` say which output column is the label and how to score it. Pools
    out-of-fold predictions across folds and scores once (stable, unlike averaging noisy
    per-fold R2); each fold early-stops on the chronological tail of its train set.
    Regression also reports the persistence (predict-yesterday) baseline RMSE.
    """
    df = recipe(site)
    _check_target(recipe, df, target)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    feat = _features(df, target)
    X, y = df[feat], _target(df, target, task)
    oof = np.full(len(df), np.nan)

    for tr, te in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(X):
        cut = int(len(tr) * 0.85)  # chronological val tail -> early stop
        m = _fit(task, X.iloc[tr[:cut]], y.iloc[tr[:cut]], X.iloc[tr[cut:]], y.iloc[tr[cut:]], **xgb_kw)
        oof[te] = _oof_predict(m, X.iloc[te], task)

    ok = ~np.isnan(oof)
    row = dict(site=site, n_test=int(ok.sum()), n_feat=len(feat), **_score(y[ok], oof[ok], task))
    if task == "reg":
        row["persist_rmse"] = _persistence_rmse(df, ok, target)
    return row


def _persistence_rmse(df, ok, target):
    """RMSE of the naive 'predict yesterday's own value' baseline on the tested rows."""
    s = pd.Series(df[target].to_numpy(), index=pd.to_datetime(df["date"])).asfreq("D")
    yhat = s.shift(1).reindex(pd.to_datetime(df["date"])).to_numpy()
    y = df[target].to_numpy()
    m = ok & ~np.isnan(yhat)
    return float(np.sqrt(mean_squared_error(y[m], yhat[m]))) if m.any() else np.nan


# ── CROSS-SITE: pooled, basin-grouped CV ─────────
def _pool(recipe, sites, target, min_rows=500):
    """Stack many recipe frames into one long dataframe, tagged with `site`.

    Sites whose data fails to build, or that are too short, are skipped; if NOTHING is
    usable the actual skip reasons are surfaced (instead of a bare 'no usable frame'),
    so a recipe bug doesn't hide behind an opaque error after a long run.
    """
    frames, skipped = [], []
    for s in sites:
        try:
            d = recipe(s)
        except Exception as e:
            skipped.append(f"{type(e).__name__}: {e}")
            continue
        _check_target(recipe, d, target)  # a missing target is a bug -> raise, never skip
        d = d.dropna(subset=[target])
        if len(d) < min_rows:
            skipped.append(f"only {len(d)} usable rows (< {min_rows})")
            continue
        d = d.copy()
        d["site"] = s
        frames.append(d)
    if not frames:
        from collections import Counter

        reasons = "; ".join(f"{n}x {r}" for r, n in Counter(skipped).most_common(3))
        raise ValueError(f"no sites produced a usable frame ({len(skipped)} sites skipped) -- {reasons}")
    return pd.concat(frames, ignore_index=True)


def _grouped_oof(X, y, groups, task, n_splits, seed=0, **xgb_kw):
    """Out-of-fold predictions from GroupKFold; each fold early-stops on a random
    held-out slice of its TRAIN rows (test groups stay fully unseen)."""
    n_groups = pd.Series(groups).nunique()
    folds = min(n_splits, n_groups)
    oof = np.full(len(y), np.nan)
    rng = np.random.RandomState(seed)
    for tr, te in GroupKFold(folds).split(X, y, groups=groups):
        v = rng.permutation(tr)
        cut = int(len(v) * 0.85)
        m = _fit(task, X.iloc[v[:cut]], y.iloc[v[:cut]], X.iloc[v[cut:]], y.iloc[v[cut:]], **xgb_kw)
        oof[te] = _oof_predict(m, X.iloc[te], task)
    return oof


def cook_many(recipe, sites=None, target=TARGET, task="reg", n_splits=5, **xgb_kw):
    """Pooled cross-site CV with site- (LOSO) and basin- (LOBO) grouped holdouts.

    `sites` defaults to all sites (get_site_ids); _pool silently drops any that fail to
    build or are too short. `target`/`task` say which output column is the label and how
    to score it. Returns the decomposition (between/within/macro) plus the LOSO/LOBO
    bracket. For classification the 'level' analogue is the per-site violation RATE.
    """
    if sites is None:
        sites = get_site_ids()
    pool = _pool(recipe, sites, target)
    feat = _features(pool, target)
    X, y = pool[feat], _target(pool, target, task)
    oof_site = _grouped_oof(X, y, pool["site"], task, n_splits, **xgb_kw)
    oof_basin = _grouped_oof(X, y, basin_groups(pool["site"]), task, n_splits, **xgb_kw)
    return dict(
        n_sites=pool["site"].nunique(),
        n_rows=len(pool),
        n_feat=len(feat),
        **_cross_metrics(pool, np.asarray(y), oof_site, oof_basin, task),
    )


def _cross_metrics(pool, y, oof_site, oof_basin, task):
    site = pool["site"].to_numpy()
    tab = pd.DataFrame({"y": y, "p": oof_site, "g": site})
    site_means = tab.groupby("g")[["y", "p"]].mean()  # per-site level
    per_site = tab.groupby("g").apply(lambda d: _per_site_score(d.y, d.p, task))
    overall = _score(y, oof_site, task)

    if task == "clf":
        return dict(
            loso_auc=overall["auc"],
            lobo_auc=_score(y, oof_basin, task)["auc"],
            prauc=overall["prauc"],
            brier=overall["brier"],
            base=overall["base"],
            between_rate_r2=r2_score(site_means.y, site_means.p),  # ranks violation rates
            macro_auc=float(np.nanmedian(per_site)),
        )
    sm = tab.groupby("g")["y"].transform("mean").to_numpy()
    pm = tab.groupby("g")["p"].transform("mean").to_numpy()
    return dict(
        loso_r2=overall["r2"],
        lobo_r2=_score(y, oof_basin, task)["r2"],
        rmse=overall["rmse"],
        between_r2=r2_score(site_means.y, site_means.p),  # ranks site levels
        within_r2=r2_score(y - sm, oof_site - pm),  # daily, level removed
        macro_r2=float(np.nanmedian(per_site)),
    )


def cook_fleet(recipe, sites=None, **kw):
    """Run cook_one on each site (per-site models) and summarise the distribution.

    `sites` defaults to all sites (get_site_ids); sites that fail are skipped. `target=`
    / `task=` (and any model kwargs) are forwarded to cook_one."""
    if sites is None:
        sites = get_site_ids()
    rows = []
    for s in sites:
        try:
            rows.append(cook_one(recipe, s, **kw))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    metric = "auc" if kw.get("task") == "clf" else "r2"
    return dict(
        n_sites=len(df), **{f"median_{metric}": df[metric].median(), f"mean_{metric}": df[metric].mean()}, per_site=df
    )


# ── comparison (paired: same sites/folds for every recipe) ────
def compare_one(recipes, site, **kw):
    """Table of cook_one metrics, one row per named recipe (same site). target=/task=
    (and model kwargs) are forwarded to cook_one."""
    return pd.DataFrame([{"recipe": n, **cook_one(fn, site, **kw)} for n, fn in recipes.items()]).set_index("recipe")


def compare_many(recipes, sites=None, progress=True, **kw):
    """Table of cook_many metrics, one row per named recipe (same sites + folds).

    `sites` defaults to all sites; resolved once here so every recipe sees the same set.
    target=/task= (and model kwargs) are forwarded to cook_many. With progress=True a
    single self-overwriting status line reports which recipe is running and elapsed time.
    """
    if sites is None:
        sites = get_site_ids()
    rows, n, t0 = [], len(recipes), time.time()
    for i, (name, fn) in enumerate(recipes.items(), 1):
        if progress:
            print(f"\rcompare_many: [{i}/{n}] {name:<28.28s} elapsed {time.time() - t0:4.0f}s", end="", flush=True)
        rows.append({"recipe": name, **cook_many(fn, sites, **kw)})
    if progress:
        print(f"\rcompare_many: done {n}/{n} recipes in {time.time() - t0:.0f}s" + " " * 30)
    return pd.DataFrame(rows).set_index("recipe")


# ── demo / smoke test ───────────────────
if __name__ == "__main__":
    from data.features import (
        agg_crops,
        agg_surplus,
        agg_weather_w_lag,
        daily_nitrate,
        nitrate_violations,
        doy_climatology_pure_signal,
        site_static,
        lagged_sensor_nitrate,
    )
    from data.transforms import flatten_buckets, merge_on_date

    EDGES, VEL = [], 0.8

    def _covariates(site):
        wb = flatten_buckets(agg_weather_w_lag(site, edges=EDGES, water_velocity=VEL))
        cb = flatten_buckets(agg_crops(site, edges=EDGES, lam=10_000, exp=True))
        sb = flatten_buckets(agg_surplus(site, edges=EDGES, lam=10_000, exp=True))
        nd = daily_nitrate(site).rename(TARGET)
        return nd, [wb, cb, sb, doy_climatology_pure_signal(nd)]

    def recipe_A(site):  # covariates + calendar (target defaults to 'nitrate_con')
        nd, parts = _covariates(site)
        return merge_on_date([nd, *parts], spine=nd.index)

    def recipe_A_static(site):  # + static site descriptors
        d = recipe_A(site)
        for k, v in site_static(site).items():
            d[k] = v
        return d

    def recipe_B(site):  # + own autoregression
        nd, parts = _covariates(site)
        own = [lagged_sensor_nitrate([site], shift=k) for k in (1, 2, 3, 7)]
        return merge_on_date([nd, *parts, *own], spine=nd.index)

    def recipe_violation(site):  # builds a binary 'violation' target column
        nd, parts = _covariates(site)  # nd is used only for the spine/calendar, NOT as a feature
        v = nitrate_violations(site).rename("violation")
        return merge_on_date([v, *parts], spine=nd.index)

    reg_recipes = {"A_covariates": recipe_A, "A_static": recipe_A_static, "B_+own_AR": recipe_B}
    sites = [s for s in get_site_ids() if daily_nitrate(s).dropna().shape[0] >= 1500][:25]

    print(f"\nINDIVIDUAL (cook_one) on {sites[0]}:")
    print(compare_one(reg_recipes, sites[0]).round(3).to_string())

    print(f"\nCROSS-SITE regression (cook_many) on {len(sites)} sites:")
    print(compare_many(reg_recipes, sites).round(3).to_string())

    print(f"\nCROSS-SITE classification -- caller passes target='violation', task='clf':")
    print(compare_many({"violation": recipe_violation}, sites, target="violation", task="clf").round(3).to_string())
