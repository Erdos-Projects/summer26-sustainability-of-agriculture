"""
cook.py -- reusable harness for comparing nitrate "recipes".
============================================================

A *recipe* is a function  site_uid -> DataFrame  containing a target column, a `date` column, and feature columns (what recipe_REG/recipe_CLF and the light pair produce). Recipes stay pure (site -> frame) and know nothing about CV; this module supplies the evaluation.

Tell cook which column is the target with `target_col=` (default "nitrate_con") and which task with `task=` ("reg" or "clf", default "reg").

    cook_many(recipe, sites)      one recipe, pooling included
    compare_many({...}, sites)    many recipes, one row each

Two holdouts, run over the same pooled rows:
    LOSO = Leave One Site Out    -- optimistic; nested basins leak
    LOFO = Leave One Family Out  -- the honest number, and what everything is aggregated from

EVERY reported aggregate comes from the LOFO out-of-fold vector. They used to come from LOSO, which made every diagnostic optimistic; one `loso_` column per task survives solely to report the leakage gap. Cross-site performance is a DECOMPOSITION rather than one number:
    lofo_between_r2   predicted vs actual per-site means -- does it rank site levels?
    lofo_within_r2    after removing each site's mean    -- does it track daily movement?
    lofo_macro_r2     median per-site R2 (equal weight per site, vs row-weighted overall)

NO EARLY STOPPING (2026-07-28). Each fold fits on 100% of its training rows with no eval_set, so a non-None `early_stopping_rounds` raises from XGBoost -- deliberately, because nothing here can honour it. The rule it replaces carved a random 15% of ROWS as its stopping slice, so the slice held the same sites as the training data: it was validated in-distribution while the score is out-of-family. It never fired (mean_best_iter sat at the ceiling in all 7 configs of the REG sweep), and measured on one LOFO fold the watched RMSE was still falling at tree 1498/1500 while out-of-family R2 PEAKED AT TREE 175 and gave back 0.0324 R2 -- about 4x the REG noise floor. The tree count is now an explicit tuned hyperparameter: src/models/tune.py resolves it per config by a prefix scan, and src/models/train.py refuses to fit a deployable model without one. See notes/early-stopping-report.md in the sustag repo.
"""

# this makes all type annotations lazy, so no runtime cost for annotations
from __future__ import annotations

# control over runtime warning messages
import warnings

warnings.filterwarnings("ignore")
import os, sys, time, json
from typing import Any, Callable, Literal, Sequence, TypeAlias

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root/
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)

from src.splits.conflict_graph import split_groups
from src.data.access import get_site_ids

# ── type aliases ───────────────────────────
Recipe: TypeAlias = Callable[[str], pd.DataFrame]  # a recipe: site_uid -> feature/target frame
Task: TypeAlias = Literal["reg", "clf"]  # regression vs binary classification
Model: TypeAlias = "xgb.XGBClassifier | xgb.XGBRegressor"  # fitted estimator

TARGET = "nitrate_con"  # default target column if the caller doesn't pass target_col=
_STRUCTURAL = {"site_uid", "site", "date", "year", "datetime"}  # bookkeeping cols, never features

# Fixed, somewhat regularized model config. `n_estimators` here is a fallback for exploratory calls
# only -- it is a REAL tree count now, not a ceiling early stopping would cut short, so anything
# being scored seriously should pass the count tune.py resolved for that recipe.
_DEFAULT_XGB = dict(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=42,
    early_stopping_rounds=None,
)

_PERM_REPEATS = 5  # shuffles per feature per fold for the permutation-importance test
_FAR_BUDGET = 0.10  # false-alarm-rate budget for _imbalance_suite's recall_at_far (not a reported column)

# lazy {site : conflict-graph component id} loader
# maps a site to its group in the conflict-graph from data.splits
_BASINGRPS = None


# ── POOL -- sites in, one long frame out ──────────────────────────────────────
def _check_target(recipe: Recipe, df: pd.DataFrame, target: str) -> None:
    """Fail quickly if a recipe didn't actually produce the requested target.
    One more way to avoid spending an hour on training for a crash at the end
    """
    name = getattr(recipe, "__name__", repr(recipe))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recipe {name!r} returned {type(df).__name__}, expected a DataFrame -- did you "
            f"forget to merge_on_date(...) the parts into one frame (e.g. return a (Series, parts) "
            f"tuple by mistake)?"
        )
    if target not in df.columns:
        raise KeyError(
            f"recipe {name!r} produced columns {list(df.columns)} -- no target {target!r}; "
            f"pass the right target_col= to cook_many."
        )


def _features(df: pd.DataFrame, target: str) -> list[str]:
    """Fetches feature columns: everything except the target and the bookkeeping columns."""
    return [c for c in df.columns if c != target and c not in _STRUCTURAL]


def _target(df: pd.DataFrame, target: str, task: Task) -> pd.Series:
    """Read the target column off the recipe's output frame (int-cast for clf)."""
    y = df[target]
    return y.astype("int64") if task == "clf" else y


def _pool(
    recipe: Recipe, sites: Sequence[str], target: str, min_rows: int = 500, progress_label: str | None = None
) -> pd.DataFrame:
    """Build the mega dataframe. Stack many recipe frames into one long dataframe, tagged with `site`.

    Use `sites` to pass an explicit site list (REQUIRED: but if used via cook_many or compare_many, it'll pass all valid sites.)
    Use `target` to specify the target column in the dataframe (REQUIRED)
    Use `min_rows` to specify the minimum number of non-nan rows a site must have to make it into the dataframe

    If `progress_label` is given, a self-overwriting status line reports cooking progress (sites cooked so far / total, and how many produced a usable frame).
    """
    sites = list(sites)
    total = len(sites)
    frames, skipped = [], []
    for i, s in enumerate(sites, 1):
        if progress_label is not None:
            print(
                f"\r  pooling {progress_label}: site {i}/{total} ({len(frames)} usable so far)",
                end="",
                flush=True,
            )
        try:
            d = recipe(s)
        except Exception as e:
            skipped.append((s, f"{type(e).__name__}: {e}"))  # build failure; reported in the end-summary
            continue

        _check_target(recipe, d, target)  # a missing target is a bug -> raise, never skip
        d = d.dropna(subset=[target])  # drop any rows missing the target
        if len(d) < min_rows:
            skipped.append((s, f"only {len(d)} usable rows (< {min_rows})"))
            continue
        d = d.copy()
        d["site"] = s
        frames.append(d)
    if progress_label is not None:
        # overwrite the last heartbeat with the accurate completed total (trailing spaces
        # clear leftovers from the longer in-progress line), then newline to keep it
        print(
            f"\r  cooked {progress_label}: {len(frames)}/{total} sites usable (min_rows >= {min_rows})" + " " * 20,
            flush=True,
        )
        # list dropped sites AFTER the heartbeat line -- an inline print mid-loop gets clobbered by
        # the next \r heartbeat, so both build failures AND the <min_rows drops were invisible.
        for s, reason in skipped:
            print(f"    [skipped] {s}: {reason}")
    if not frames:
        from collections import Counter

        # if there aren't ANY frames, builds a human-readable error message explaining
        # (hopefully) why the sites got skipped.
        reasons = "; ".join(f"{n}x {r}" for r, n in Counter(r for _, r in skipped).most_common(3))
        raise ValueError(f"no sites produced a usable frame ({len(skipped)} sites skipped) -- {reasons}")
    out = pd.concat(frames, ignore_index=True)
    # float32 for the FEATURES only: XGBoost's DMatrix casts to float32 internally anyway, so this
    # is exact rather than approximate -- verified here, fold predictions are bit-identical -- and it
    # halves pool memory while trimming fit time. The target keeps its dtype, so every metric is
    # still computed in float64 and none of the scoring arithmetic moves.
    f64 = [c for c in out.columns if c != target and out[c].dtype == "float64"]
    if f64:
        out[f64] = out[f64].astype("float32")
    return out


# ── FIT -- frame + folds in, fitted models out ────────────────────────────────


def _model(task: Task, **overrides: Any) -> Model:
    """Wrapper for returning the appropriate model, Classifier vs Regressor.
    Pass a different model configuration, for instance, as an unwrapped dictionary.
    It will override the defaults.
    """
    cfg = {**_DEFAULT_XGB, **overrides}
    if task == "clf":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **cfg)
    return xgb.XGBRegressor(eval_metric="rmse", **cfg)


def _oof_predict(model: Model, X: pd.DataFrame, task: Task) -> np.ndarray:
    """Just an 'Out Of Fold' prediction wrapper. Used solely to differentiate between the two primary tasks."""
    return model.predict_proba(X)[:, 1] if task == "clf" else model.predict(X)


def basin_groups(sites_mega_series: pd.Series) -> pd.Series:
    """Take in the mega multi-site dataframe but only the sites_uid column.

    Return a mapping from index of sites_mega_series to conflict graph id. Example with fake conflict_graph_id numbers:
    site_mega_series         output pd.Series
    index site_uid        index conflict_graph_id
    0     WQS0039         0            4
    1     WQS0039    ->   1            4
    2     WQS0115         2            7
    3     WQS0003         3            2

    """

    # lazily load the basin conflict graph
    global _BASINGRPS
    if _BASINGRPS is None:
        _BASINGRPS = split_groups()

    # this is the max integer not already assigned
    nxt = max(_BASINGRPS.values(), default=-1) + 1

    # build a mapping from site_uid to conflict graph conn comp id
    mapping = {}
    for s in pd.unique(sites_mega_series):
        # if site s is in the basin graph, use its component id
        mapping[s] = _BASINGRPS.get(s, nxt)
        # otherwise, add 1 to the counter
        nxt += s not in _BASINGRPS
    return sites_mega_series.map(mapping)


def folds_true_lofo(families, max_holdout_pct: float = 0.2):
    """(train_idx, test_idx) pairs for TRUE leave-one-family-out: one fold per basin family.

    _grouped_models runs GroupKFold(min(n_splits, n_groups)), so at n_splits=5 each fold holds out roughly a FIFTH of the families at once and a family shares its fold with several others. This holds out EXACTLY ONE family and trains on every other site -- the honest leave-one-out, and the regime deployment actually operates in (every known site trained on, one unseen basin predicted).

    `max_holdout_pct` caps how much of the pool a single held-out family may be, as a fraction of total rows. A family above the cap is never held out: its fold would score a large slice of the data against a model trained on what is left, so that one fold dominates the pooled OOF and its idiosyncrasies become the headline number. Such a family is still TRAINED ON in every fold -- it is barred from being the test set, not dropped from the data. Its rows simply never receive an OOF prediction, which is what the coverage line reports.

    Families are visited in sorted order so folds are deterministic. A family that is the whole pool yields no fold (no training rows would be left).
    """
    fam = np.asarray(families)
    total = len(fam)
    counts = pd.Series(fam).value_counts()
    for f in sorted(counts.index):
        if total and counts[f] / total > max_holdout_pct:
            continue
        test = np.flatnonzero(fam == f)
        train = np.flatnonzero(fam != f)
        if len(test) and len(train):
            yield train, test


def _fold_models(X, y, folds, task, **xgb_kw):
    """Fit one model per (train_idx, test_idx) fold; yield (test_idx, model).

    `folds` = any iterable of index pairs: _grouped_models' GroupKFold for LOSO/LOFO, folds_true_lofo for one-family-at-a-time. Fits on ALL of a fold's training rows -- no validation slice, no eval_set, no early stopping. A non-None `early_stopping_rounds` therefore raises from XGBoost, deliberately: nothing here can honour it, and the tree count is tune.py's job (see the module docstring).
    """
    for train, test in folds:
        m = _model(task, **xgb_kw)
        m.fit(X.iloc[np.asarray(train)], y.iloc[np.asarray(train)], verbose=False)
        yield test, m


def _grouped_models(X, y, groups, task, n_splits, **xgb_kw):
    """Fit one model per GroupKFold split. "Rows with the same group class must be kept together."

    The two main kinds of group identities:
        a. groups = site_uid col of X: then one SITE is left out, LOSO
        b. groups = conflict graph id: then one FAMILY of basins is left out, LOFO

    min(n_splits, n_groups) folds, NOT leave-one-out -- see folds_true_lofo for the literal one. The fold/split logic lives here so both OOF prediction and feature importance consume the SAME models: no duplicated splitting, and importance is free (the models already exist).
    """
    folds = min(n_splits, pd.Series(groups).nunique())
    if folds < 2:
        return  # <2 groups -> grouped CV undefined (e.g. LOFO with one basin); caller gets all-NaN OOF
    yield from _fold_models(X, y, GroupKFold(folds).split(X, y, groups=groups), task, **xgb_kw)


def _fold_oof(models, X, task: Task, n: int) -> np.ndarray:
    """Out-of-fold prediction vector from already-fitted (model, test_idx) pairs. Split from fitting so the same fits also serve gain and permutation importance."""
    oof = np.full(n, np.nan)
    for m, te in models:
        oof[te] = _oof_predict(m, X.iloc[te], task)
    return oof


def _grouped_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    task: Task,
    n_splits: int,
    **xgb_kw: Any,
) -> np.ndarray:
    """OOF predictions under GroupKFold (groups held out intact). A wrapper around _grouped_models for callers that want only the predictions, not the models or their importances."""
    oof = np.full(len(y), np.nan)
    for te, m in _grouped_models(X, y, groups, task, n_splits, **xgb_kw):
        oof[te] = _oof_predict(m, X.iloc[te], task)
    return oof


# ── SCORE -- predictions in, metrics out ──────────────────────────────────────


def _best_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over the probability-threshold sweep, with the threshold that hits it.

    The 0/1 violation target is rare, so a fixed 0.5 cutoff on a calibrated P(violation) scores F1~=0 and tells you nothing; the max-F1 operating point is the honest 'how good is the exceedance *decision* if you pick a sensible cutoff'. The threshold is tuned on these same OOF rows, so read it as mildly optimistic and always alongside prauc (which is threshold-free). Not a reported column -- _cross_metrics dropped it because max-F1 cherry-picks a threshold AND weights precision == recall, contradicting the beta=2 deployment; kept here because _score is also the tuner's per-prefix scorer.
    """
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    prec, rec, thr = precision_recall_curve(y, pred)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.nanargmax(f1))
    # precision_recall_curve returns len(thr) == len(prec) - 1; the last point (recall 0) has no cutoff
    return float(f1[i]), float(thr[i]) if i < len(thr) else float("nan")


def _imbalance_suite(
    y: np.ndarray | pd.Series, pred: np.ndarray | pd.Series, far: float | None = None
) -> dict[str, float]:
    """Class-imbalance-robust metrics for a set of (y, P(positive)).

    prauc_lift    average precision / base rate -- >1 beats a random ranker; imbalance-normalised.
                  The ONLY one of the four that is threshold-free, and the only one _cross_metrics
                  reports. Bounded above by 1/base_rate, so it is not comparable across cohorts of
                  different prevalence.
    f2            best F2 over the PR sweep (beta=2 -> recall weighted beta^2 = 4x precision)
    mcc           best Matthews correlation over the ROC sweep
    recall_at_far recall achievable at a false-alarm rate (FPR) <= `far`

    The last three are MAXIMA over a threshold sweep tuned on the same rows they are scored on, so they answer "best achievable with a perfect hindsight cutoff", not "what you will get". That is why only prauc_lift survives into the reported set; the honest operating points come from _beta_point and beta_operating_points, which commit to a threshold per beta.
    """
    far = _FAR_BUDGET if far is None else far
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    keys = ("prauc_lift", "f2", "mcc", "recall_at_far")
    if len(np.unique(y)) < 2:
        return {k: np.nan for k in keys}

    base = float(y.mean())
    prauc = average_precision_score(y, pred)

    prec, rec, _ = precision_recall_curve(y, pred)
    den_f2 = 4.0 * prec + rec  # F_beta with beta=2 -> (1+4)PR / (4P+R)
    f2 = np.divide(5.0 * prec * rec, den_f2, out=np.zeros_like(prec), where=den_f2 > 0)

    fpr, tpr, _ = roc_curve(y, pred)
    P = float(y.sum())
    N = float(len(y) - P)
    tp, fp = tpr * P, fpr * N
    fn, tn = P - tp, N - fp
    den_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den_mcc, out=np.zeros_like(den_mcc), where=den_mcc > 0)
    within = fpr <= far  # operating points inside the false-alarm budget

    return dict(
        prauc_lift=float(prauc / base) if base > 0 else float("nan"),
        f2=float(np.nanmax(f2)),
        mcc=float(np.nanmax(mcc)),
        recall_at_far=float(tpr[within].max()) if within.any() else 0.0,
    )


# The widget's beta slider values (0.5..4.0 step 0.5). The slider snaps to these, so it looks each
# up exactly rather than interpolating -- keep the two in step.
BETA_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_PR_CURVE_POINTS = 256  # downsample width for the stored curve; see operating_points()
_BETA = 2.0  # recall emphasis of the deployed operating point (recall weighted beta^2 = 4x precision)


def beta_operating_points(y, pred, betas=BETA_GRID) -> list[dict]:
    """For each beta, the threshold tau maximising F_beta, with the operating point there.

    Returns recall (TP/(TP+FN)), precision (TP/(TP+FP)) and fdr (= 1 - precision, the share of
    alarms that are false). One precision_recall_curve sweep drives every beta.

    Everything a caller might want follows from these plus the base rate: given (p, recall,
    precision), TP = recall*p, FP = TP*(1-precision)/precision, FN = p - TP, TN = (1-p) - FP, so
    accuracy, FPR and F1 are all derivable without storing another curve.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present in the predictions to tune a threshold.")

    # precision_recall_curve returns prec/rec of length n+1; the final point (recall 0, precision 1)
    # has no threshold, so drop it to align prec/rec with the n thresholds (cf. _best_f1).
    prec, rec, thr = precision_recall_curve(y, pred)
    P, R = prec[:-1], rec[:-1]

    rows = []
    for b in betas:
        b2 = b * b
        den = b2 * P + R
        fbeta = np.divide((1 + b2) * P * R, den, out=np.zeros_like(P), where=den > 0)
        i = int(np.nanargmax(fbeta))
        rows.append({
            "beta": float(b),
            "tau": float(thr[i]),
            "recall": float(R[i]),
            "precision": float(P[i]),
            "fdr": float(1.0 - P[i]),  # false-discovery rate = share of our alarms that are false
        })
    return rows


def operating_points(y, pred, betas=BETA_GRID) -> dict:
    """Everything needed to reconstruct any decision threshold, from one PR sweep.

    Returned as three parts, because they answer different questions and only one of them is exact:

      beta_table  the EXACT operating point at each beta in the grid. The widget snaps its slider to
                  these, so they must not be reconstructed from the downsampled curve below.
      pr_curve    (recall, precision, tau) resampled to a fixed grid, for any beta or tau asked for
                  later. Downsampled because the full curve has one point per distinct predicted
                  value -- ~10^5 of them -- and this has to fit in a JSON log.
      base_rate   prevalence IN THE SCORED ROWS, which is not the pooled prevalence when the OOF
                  covers only part of the pool (a --true-lofo run with a cap). FDR is
                  prevalence-dependent, so quoting it against the wrong base rate misstates it.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    ys, ps = y[ok], pred[ok]

    prec, rec, thr = precision_recall_curve(ys, ps)
    P, R, T = prec[:-1], rec[:-1], thr
    # Resample on a monotone recall grid: precision_recall_curve returns recall DESCENDING, and
    # np.interp needs its x ascending, hence the reversal.
    grid = np.linspace(R.min(), R.max(), min(_PR_CURVE_POINTS, len(R)))
    order = np.argsort(R)
    return {
        "beta_table": beta_operating_points(ys, ps, betas),
        "base_rate": float(ys.mean()),
        "coverage": float(ok.mean()),  # share of pooled rows that received a prediction at all
        "n_scored": int(ok.sum()),
        "pr_curve": {
            "recall": [float(v) for v in grid],
            "precision": [float(v) for v in np.interp(grid, R[order], P[order])],
            "tau": [float(v) for v in np.interp(grid, R[order], T[order])],
        },
    }


def _score(y: np.ndarray | pd.Series, pred: np.ndarray | pd.Series, task: Task) -> dict[str, float]:
    """Pooled metrics for a set of (y, prediction). NaN predictions are dropped (e.g. an
    all-NaN OOF from a grouped split with too few groups -> all metrics NaN).

    Emits BARE metric names (r2, prauc, auc, rmse); the loso_/lofo_ prefixes are attached by _cross_metrics. tune.py's tree-count scan calls this directly -- one metric, thousands of times -- so it needs the unprefixed form.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    if len(y) == 0:  # nothing to score (e.g. LOFO with a single basin group)
        keys = ("auc", "prauc", "f1", "f1_thresh", "brier", "base") if task == "clf" else ("rmse", "mae", "r2")
        return {k: np.nan for k in keys}
    if task == "clf":
        two = len(np.unique(y)) == 2
        f1, f1_thresh = _best_f1(y, pred)
        return dict(
            auc=roc_auc_score(y, pred) if two else np.nan,
            prauc=average_precision_score(y, pred) if two else np.nan,
            f1=f1,  # best-F1 over the threshold sweep (exceedance-decision quality)
            f1_thresh=f1_thresh,  # the P(violation) cutoff that achieves it
            brier=brier_score_loss(y, pred),
            base=float(y.mean()),
        )
    return dict(
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        mae=float(mean_absolute_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


def _per_site_score(y: np.ndarray | pd.Series, pred: np.ndarray | pd.Series, task: Task) -> float:
    """One-site score, or NaN when undefined (one class / no variance)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if task == "clf":
        return roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan
    return r2_score(y, pred) if (len(y) > 1 and np.std(y) > 0) else np.nan


TAIL_FRAC = 0.10  # the worst DECILE -- the outlier question. Wider fracs stop seeing the tail: see _tail_rank.


def _tail_rank(site_means: pd.DataFrame, frac: float = TAIL_FRAC) -> dict[str, float]:
    """How well the site ranking finds the WORST sites, which between_r2 does not answer.

    between_r2 is an R^2 over site means, so it is dominated by the bulk and says nothing about which sites sit at the top -- and a boosted ensemble shrinks extremes toward the mean, so it can score respectably while flattening exactly the sites a siting or triage decision cares about. Two numbers, both read off the held-out (LOFO) site means, so they describe ungauged basins rather than fitted ones:

        site_ap      Average precision for "this site is in the worst `frac`", scored by its predicted mean. Chance is `frac` itself, so read it against that and not against 0. Preferred over ROC-AUC for the same reason lofo_prauc is: on an imbalanced label ROC-AUC flatters badly.
        captured     Of the excess the best possible shortlist would find, the share this one does: (mean actual over the PREDICTED worst k - cohort mean) / (mean actual over the TRUE worst k - cohort mean). 0 is a random shortlist, 1 is the best achievable. The only one in the target's own units, so the only one a stakeholder reads directly.
    A shrinkage diagnostic (the slope of actual on predicted across the flagged sites) is deliberately NOT here. Measured on the 79-site REG cohort it ran 1.120 / 0.517 / 1.075 at frac 0.1 / 0.25 / 0.5 -- non-monotone in frac, which a stable property is not, because it fits a line through k=8 noisy points. In a fixed-frac score column that is noise with a name. _experiment21.py computes it across the whole ladder, which is where it can actually be read.

    WATCH THE FRAC. At frac=0.5 these ask "which half of the state is the problem half" -- well powered at ~80 sites, and a real screening question. They are NOT tail-sensitive there: measured against a synthetic model whose top 15% is deliberately compressed, site_ap and captured both score it HIGHER than a uniformly-good model (0.995 vs 0.955 and 0.991 vs 0.879), because the compression happens inside the top half and a half-split cannot see it. At frac=0.1 the same pair separates them 2.7x the right way. Pass frac=0.1 when the question is genuinely about outliers.

    n < 10 sites, or a cohort with no spread in the site means, returns NaN rather than a number built on nothing.
    """
    keys = ("site_ap", "captured")
    n = len(site_means)
    if n < 10:
        return dict.fromkeys(keys, float("nan"))
    y, p = site_means["y"].to_numpy(float), site_means["p"].to_numpy(float)
    k = max(1, int(round(frac * n)))
    order_y, order_p = np.argsort(-y), np.argsort(-p)
    label = np.zeros(n, int)
    label[order_y[:k]] = 1
    cohort, best = y.mean(), y[order_y[:k]].mean()
    got = y[order_p[:k]].mean()
    return {
        "site_ap": float(average_precision_score(label, p)) if label.min() != label.max() else float("nan"),
        "captured": float((got - cohort) / (best - cohort)) if best != cohort else float("nan"),
    }


def _decomposition(y: np.ndarray, oof: np.ndarray, site: np.ndarray, task: Task) -> dict[str, float]:
    """between / within / macro decomposition of ONE out-of-fold prediction vector, plus the tail-ranking pair."""
    ok = ~(np.isnan(y) | np.isnan(oof))
    if ok.sum() < 2:
        keys = ("between_rate_r2", "macro_auc") if task == "clf" else ("between_r2", "within_r2", "macro_r2")
        return {k: float("nan") for k in (*keys, "site_ap", "captured")}
    tab = pd.DataFrame({"y": y[ok], "p": oof[ok], "g": site[ok]})
    site_means = tab.groupby("g")[["y", "p"]].mean()
    per_site = tab.groupby("g").apply(lambda d: _per_site_score(d.y, d.p, task))
    tail = _tail_rank(site_means)
    if task == "clf":
        return dict(
            between_rate_r2=r2_score(site_means.y, site_means.p),
            macro_auc=float(np.nanmedian(per_site)),
            **tail,
        )
    sm = tab.groupby("g")["y"].transform("mean").to_numpy()
    pm = tab.groupby("g")["p"].transform("mean").to_numpy()
    return dict(
        between_r2=r2_score(site_means.y, site_means.p),
        within_r2=r2_score(tab.y.to_numpy() - sm, tab.p.to_numpy() - pm),
        macro_r2=float(np.nanmedian(per_site)),
        **tail,
    )


def _beta_point(y: np.ndarray, pred: np.ndarray, beta: float = _BETA) -> dict[str, float]:
    """Recall and FDR at the threshold maximising F_beta -- the DEPLOYED operating point.

    Replaces recall_at_far, which read the ROC curve at FPR <= 10%: ~4x stricter than what ships (beta=2 implies FPR ~= 0.44). FDR = FP/(TP+FP) is prevalence-dependent, so read it against `base`. tau is tuned on the same OOF it is scored on -> mildly optimistic.
    """
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    if len(np.unique(y)) < 2:
        return {"recall_at_beta": float("nan"), "fdr_at_beta": float("nan")}
    prec, rec, _ = precision_recall_curve(y, pred)
    den = (beta**2) * prec + rec
    fb = np.divide((1 + beta**2) * prec * rec, den, out=np.zeros_like(prec), where=den > 0)
    i = int(np.nanargmax(fb))
    return {"recall_at_beta": float(rec[i]), "fdr_at_beta": float(1.0 - prec[i])}


def _cross_metrics(
    pool: pd.DataFrame,
    y: np.ndarray,
    oof_site: np.ndarray,
    oof_family: np.ndarray,
    task: Task,
) -> dict[str, float]:
    """The reported score set -- 10 CLF / 6 REG columns.

    What was dropped, and why:
      * every loso_* except loso_auc/loso_r2 -- they correlate >=0.91 with one another, and LOSO
        ANTI-correlates with LOFO across recipes (REG -0.29, CLF -0.48), so it is misleading as a
        ranking criterion. One column survives to report the LOSO-LOFO leakage gap.
      * lofo_f1 (max-F1 cherry-picks a threshold AND weights precision == recall, contradicting the
        beta=2 deployment), lofo_mcc / lofo_f2 / lofo_recall_at_far (0.75-0.95 correlated with the
        rest, and all three are hindsight maxima over a threshold sweep).
      * persist_skill -- an exact monotone transform of loso_r2 (rho = +-1.00) whose baseline the
        product cannot use: predict-yesterday needs a history a virtual site does not have.
      * spearman (0.92 with loso_r2), and the LOSO-computed rmse/between/within/macro.

    ALL aggregates now come from the LOFO OOF; they used to come from LOSO, which made every diagnostic optimistic. `lofo_prauc`/`lofo_r2` are load-bearing names -- tune.py ranks by them.
    """
    site = pool["site"].to_numpy()
    overall = _score(y, oof_site, task)  # LOSO -- retained only for the leakage gap
    lofo = _score(y, oof_family, task)  # LOFO -- the honest number everything else is based on

    if task == "clf":
        return dict(
            loso_auc=overall["auc"],
            lofo_prauc=lofo["prauc"],  # HEADLINE: average precision on the family-grouped OOF
            lofo_auc=lofo["auc"],
            lofo_prauc_lift=_imbalance_suite(y, oof_family)["prauc_lift"],
            **{f"lofo_{k}": v for k, v in _beta_point(y, oof_family).items()},
            lofo_brier=lofo["brier"],
            base=lofo["base"],
            **{f"lofo_{k}": v for k, v in _decomposition(y, oof_family, site, task).items()},
        )
    return dict(
        loso_r2=overall["r2"],
        lofo_r2=lofo["r2"],
        lofo_rmse=lofo["rmse"],  # human-readable mg/L; monotone in lofo_r2, never rank on it
        **{f"lofo_{k}": v for k, v in _decomposition(y, oof_family, site, task).items()},
    )


# ── IMPORTANCE -- both ride the FAMILY fits ───────────────────────────────────


def _gain_importance(fold_models, feat: list[str]) -> pd.Series:
    """Mean XGBoost GAIN importance across the fold-models, indexed by feature, sorted desc."""

    cols = [pd.Series(m.feature_importances_, index=feat) for m, _ in fold_models]
    return pd.concat(cols, axis=1).mean(axis=1).sort_values(ascending=False)


def _perm_importance(fold_models, X, y, feat: list[str], task: Task, cols=None, seed: int = 0) -> pd.Series:
    """Permutation importance for `cols` (default: all `feat`), averaged over folds. For each fold model, shuffle a column on that fold's HELD-OUT rows and measure the drop in score (R2 for reg, ROC-AUC for clf), _PERM_REPEATS times.

    Same score-drop semantics as sklearn.inspection.permutation_importance, done by hand so the column set is RESTRICTABLE -- a full per-column pass is ~K * n_feat * _PERM_REPEATS rescorings, and a caller screening one added block only needs those columns.

    Folds whose held-out set can't be scored (e.g. a single-class clf window -> AUC undefined) are skipped; a column with no scorable fold comes back NaN.
    """
    cols = list(feat) if cols is None else [c for c in cols if c in feat]
    if not cols:
        return pd.Series(dtype=float)
    score_fn = r2_score if task == "reg" else roc_auc_score
    rng = np.random.RandomState(seed)
    fold_models = list(fold_models)
    n = len(fold_models)
    acc = {c: [] for c in cols}
    for i, (m, te) in enumerate(fold_models, 1):
        # slow step -> self-overwriting status line
        print(f"\r    perm fold {i}/{n} ({len(cols)} cols x {_PERM_REPEATS})", end="", flush=True)
        Xte = X.iloc[te].copy()
        yte = y.iloc[te]
        try:
            base = score_fn(yte, _oof_predict(m, Xte, task))
        except ValueError:
            continue  # fold unscoreable (e.g. single-class clf window) -> skip
        for c in cols:
            orig = Xte[c].to_numpy().copy()
            drops = []
            for _ in range(_PERM_REPEATS):
                Xte[c] = rng.permutation(orig)
                try:
                    drops.append(base - score_fn(yte, _oof_predict(m, Xte, task)))
                except ValueError:
                    pass
            Xte[c] = orig  # restore before the next column
            if drops:
                acc[c].append(float(np.mean(drops)))
    print(flush=True)  # newline so the finished status line stays put (not erased)
    return pd.Series({c: (float(np.mean(v)) if v else np.nan) for c, v in acc.items()}).sort_values(ascending=False)


# ── API -- pooled cross-site CV ───────────────────────────────────────────────


def cook_many(
    recipe: Recipe,
    sites: Sequence[str] | None = None,
    target_col: str = TARGET,
    task: Task = "reg",
    n_splits: int = 5,
    extra_importance_test: bool = False,
    progress: bool | str = False,
    min_rows: int = 500,
    true_lofo: bool = False,
    max_holdout_pct: float = 0.2,
    perm_cols=None,
    loso: bool = True,
    pool: pd.DataFrame | None = None,
    **xgb_kw: Any,
) -> dict:
    """Pooled cross-site CV with site- (LOSO) and family- (LOFO) grouped holdouts.

    Two passes over the same rows under two holdouts. The MODELS are kept, not just their predictions: they were being fitted and discarded anyway, so both importances ride the FAMILY fits for free. That matters -- LOSO and LOFO anti-correlate across recipes (REG -0.29, CLF -0.48), so an importance scored on LOSO models answers a systematically different question from the lofo_* columns beside it.

    `true_lofo` makes LOFO literal: one fold per basin family, that family held out, every other site trained on -- instead of GroupKFold(5), which holds out about a fifth of the ROWS at once. It returns the IDENTICAL column set, aggregated identically, so it is a drop-in alternative.

    HOW MUCH IT CHANGES THE NUMBERS DEPENDS ENTIRELY ON THE COHORT, and on this one it barely does. GroupKFold keeps a group intact, so a family too large to fit in a fifth-sized fold already gets a fold to itself -- which IS leave-one-family-out for it. Here families 0 and 2 are 40.9% and 27.8% of pooled rows, so both regimes train on exactly 59.1% and 72.2% respectively when those are the test set, and they coincide on 68.7% of the data. Only the 18 small families gain (+8..10% training data each), which is +2.25% row-weighted overall.

    Since the returned columns carry no marker of which regime produced them, record it wherever you log the run -- src/models/train.py writes `true_lofo` and `max_holdout_pct` for exactly this reason.

    Parameters
    ----------
    recipe : Recipe
        Callable site_uid -> feature/target frame.
    sites : Sequence[str], optional
        Site ids; default the full cohort (get_site_ids).
    target_col : str, default TARGET
    task : {'reg', 'clf'}, default 'reg'
    n_splits : int, default 5
        Folds for both holdouts.
    extra_importance_test : bool, default False
        Also compute permutation importance (slow).
    progress : bool or str, default False
        Pooling progress label; True uses recipe.__name__.
    min_rows : int, default 500
        Per-site inclusion floor on non-NaN-target rows.
    true_lofo : bool, default False
        One family per fold instead of GroupKFold.
    max_holdout_pct : float, default 0.2
        With true_lofo, families above this row share are never held out.
    perm_cols : sequence, optional
        Restrict permutation importance to these columns; None = all.
    loso : bool, default True
        Run the site-grouped pass. It is a FULL second CV whose only product is the leakage-gap column (`loso_r2` / `loso_auc`); everything reported is aggregated from the family pass. Off, the column set is unchanged and that one column is NaN -- which is what src/models/tune.py does per config, running one site pass for the winner instead.
    pool : pd.DataFrame, optional
        A prebuilt pool from `_pool`, to avoid re-pooling. `recipe`, `sites` and `min_rows` are then unused for pooling. src/models/train.py passes one so a training run pools once instead of once for the CV and again for the deployable fit.
    **xgb_kw
        XGBoost overrides. `n_estimators` is a real tree count -- there is no early stopping.

    Returns
    -------
    dict
        Score columns + 'importance' (+ 'importance_perm', + 'operating_points' for clf).

    See Also
    --------
    compare_many : the same, one row per named recipe.
    """
    if sites is None:
        sites = get_site_ids()
    label = (progress if isinstance(progress, str) else getattr(recipe, "__name__", "recipe")) if progress else None
    if pool is None:
        pool = _pool(recipe, sites, target_col, min_rows=min_rows, progress_label=label)
    feat = _features(pool, target_col)
    X, y = pool[feat], _target(pool, target_col, task)

    basin_grp = basin_groups(pool["site"])
    n_families = int(pd.Series(basin_grp).nunique())
    if n_families < 2:  # leave-one-basin-out needs >=2 families to hold one out
        print(
            f"  [LOFO skipped] {n_families} basin family across these {pool['site'].nunique()} "
            f"sites (need >=2) -- lofo_* = NaN",
            file=sys.stderr,
        )

    fam_folds = None
    if true_lofo:
        fam_folds = list(folds_true_lofo(basin_grp, max_holdout_pct))
        if not fam_folds:
            print(
                f"  [LOFO skipped] no basin family is <= {max_holdout_pct:.0%} of the {len(pool)} pooled "
                f"rows, so none is eligible to hold out -- lofo_* = NaN. Raise max_holdout_pct.",
                file=sys.stderr,
            )

    # The site pass is a FULL second CV whose only product is the leakage gap; everything else is
    # aggregated from the family pass. With loso=False it is skipped, oof_site stays all-NaN and
    # _cross_metrics emits a NaN loso_* -- the column set is identical either way, so no consumer
    # has to care.
    site_models = (
        [(m, te) for te, m in _grouped_models(X, y, pool["site"], task, n_splits, **xgb_kw)] if loso else []
    )
    fam_models = [
        (m, te)
        for te, m in (
            _fold_models(X, y, fam_folds, task, **xgb_kw)
            if true_lofo
            else _grouped_models(X, y, basin_grp, task, n_splits, **xgb_kw)
        )
    ]

    oof_site = _fold_oof(site_models, X, task, len(y))
    oof_family = _fold_oof(fam_models, X, task, len(y))

    if true_lofo:
        # Reported to stderr, NOT as a column: this regime must return the identical column set so
        # every existing consumer keeps working. Coverage still matters -- rows in a family too big
        # to hold out never enter the OOF, so the scores describe only the eligible part of the
        # cohort, and that is not visible in any of the numbers themselves.
        covered = float(np.mean(~np.isnan(oof_family))) if len(oof_family) else float("nan")
        print(
            f"  [true LOFO] {len(fam_folds)}/{n_families} families held out one at a time "
            f"(cap {max_holdout_pct:.0%} of rows); scores cover {covered:.1%} of pooled rows",
            file=sys.stderr,
        )

    # The honest LOFO out-of-fold vector is right here, and it is the only expensive ingredient of a
    # decision-threshold table -- src/models/tune_threshold.py used to re-run this whole grouped CV
    # just to get it back. Derive the operating points now so every logged run carries its own,
    # rather than depending on someone remembering to run a second script.
    ops = None
    if task == "clf":
        scored = np.asarray(y, dtype=float)[~np.isnan(oof_family)]
        if len(np.unique(scored)) > 1:  # a fold set with one class present cannot define a threshold
            ops = operating_points(y, oof_family)

    out = dict(
        n_sites=pool["site"].nunique(),
        n_families=n_families,  # # of basin families = # of LOFO groups (low -> noisy LOFO)
        n_rows=len(pool),
        n_feat=len(feat),
        **_cross_metrics(pool, np.asarray(y), oof_site, oof_family, task),
        # Gain rides the FAMILY models, like every other reported quantity. A diagnostic that
        # describes a different set of models from the scores beside it is worse than a
        # discontinuity in the log.
        importance=_gain_importance(fam_models, feat),
    )
    if ops is not None:
        # Nested, so compare_many can lift the whole thing out of the row in one pop -- these are
        # dicts and lists, and would otherwise poison the numeric scoreboard DataFrame.
        out["operating_points"] = ops

    # status messages. np.mean of an empty list is a NaN plus a RuntimeWarning, so the LOSO detail
    # is only printed when that pass actually ran.
    n_total = pool["site"].nunique()
    if site_models:
        test_per_fold = [pool["site"].iloc[te].nunique() for _, te in site_models]
        train_per_fold = [n_total - t for t in test_per_fold]
        loso_note = (
            f"{len(site_models)}-fold LOSO "
            f"(~{int(np.mean(train_per_fold))} train / ~{int(np.mean(test_per_fold))} test per fold)"
        )
    else:
        loso_note = "LOSO skipped (loso=False -> loso_* is NaN)"
    print(f"  cooked {label}: {n_total} sites, {loso_note}, {len(fam_models)}-fold LOFO")
    if extra_importance_test:
        out["importance_perm"] = _perm_importance(fam_models, X, y, feat, task, cols=perm_cols)
    return out


def compare_many(
    recipes: dict[str, Recipe],
    sites: Sequence[str] | None = None,
    progress: bool = True,
    **kw: Any,
) -> pd.DataFrame:
    """Table of cook_many metrics, one row per named recipe (same sites + folds).

    `sites` defaults to all sites; resolved once here so every recipe sees the same set.
    target_col=/task=/min_rows= (and model kwargs) are forwarded to cook_many. With progress=True each
    recipe prints a header line (which recipe / elapsed) followed by cook_many's own
    self-overwriting per-site cooking line.

    Importances land in .attrs['importance'] / .attrs['importance_perm'] and the classifier
    threshold tables in .attrs['operating_points'], so the returned table stays numeric.
    """
    if sites is None:
        sites = get_site_ids()
    # `pool` rides through **kw to every cook_many call, so with more than one recipe every recipe
    # after the first would be silently scored on the FIRST recipe's columns -- a wrong number that
    # looks entirely plausible. Refuse instead.
    if kw.get("pool") is not None and len(recipes) > 1:
        raise ValueError(
            f"compare_many got a prebuilt pool for {len(recipes)} recipes -- every recipe would be scored on the "
            f"FIRST recipe's columns. Pass pool= only for a single recipe (src/models/train.py's case)."
        )
    rows, imps, perms, ops, n, t0 = [], {}, {}, {}, len(recipes), time.time()
    for i, (name, fn) in enumerate(recipes.items(), 1):
        if progress:
            print(f"compare_many: [{i}/{n}] {name:<28.28s} elapsed {time.time() - t0:4.0f}s", flush=True)
        r = cook_many(fn, sites, progress=(name if progress else False), **kw)
        imps[name] = r.pop("importance")
        if "importance_perm" in r:
            perms[name] = r.pop("importance_perm")
        if "operating_points" in r:
            ops[name] = r.pop("operating_points")  # dicts/lists -> attrs; the table stays numeric
        rows.append({"recipe": name, **r})
    if progress:
        print(f"compare_many: done {n}/{n} recipes in {time.time() - t0:.0f}s")
    out = pd.DataFrame(rows).set_index("recipe")
    out.attrs["importance"] = imps  # {recipe -> per-feature mean gain}; see importance_table()
    if perms:
        out.attrs["importance_perm"] = perms  # permutation importances (extra_importance_test=True)
    if ops:
        out.attrs["operating_points"] = ops  # {recipe -> beta_table + pr_curve + base_rate} (clf)
    return out


# ── VIEWS -- reading importances back out ─────────────────────────────────────


def importance_table(result, topn: int | None = None, key: str = "importance") -> pd.DataFrame:
    """Tidy feature-importance view: features (rows) x recipes (columns), values = mean
    importance, sorted by the row mean. Accepts the output of compare_many (reads
    result.attrs[key]), or a {name -> Series} dict, or a single Series.

    key="importance" (gain, default) or key="importance_perm" (permutation, only present if
    the compare_many call used extra_importance_test=True). `topn` keeps the top-N features.
    """
    if isinstance(result, pd.Series):
        imps = {key: result}
    elif isinstance(result, dict):
        imps = result
    else:  # a compare_many DataFrame
        imps = result.attrs.get(key)
        if not imps:
            raise ValueError(
                f"no {key!r} importances found -- pass a compare_many result"
                + (" run with extra_importance_test=True" if key == "importance_perm" else "")
            )
    tab = pd.DataFrame(imps)  # features (union) x recipes; features absent from a recipe -> NaN
    tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index]  # order by avg importance
    tab.index.name = "features"  # so to_csv writes a proper header instead of "Unnamed: 0"
    return tab if topn is None else tab.head(topn)


def importance_breakdown(
    result, recipe: str | None = None, key: str = "importance", topn: int | None = None
) -> pd.DataFrame:
    """Per-feature importance for ONE recipe, as 'raw' score + 'pct' (share of the total).

    Accepts a single Series, a {name -> Series} dict, or a compare_many result (reads
    result.attrs[key]); for a multi-recipe result pass recipe= to pick one. Sorted desc.

    Units depend on `key`:
      key="importance"      raw = fraction of the model's total GAIN (already sums to ~1),
                            so pct is that as a percentage -- "X% of the model's gain".
      key="importance_perm" raw = drop in the score (R2/AUC) when the feature is permuted on
                            held-out rows (metric units; READ THIS ONE). pct is its share of
                            the total, which is only loosely meaningful (can be skewed by
                            negative entries) -- prefer the raw column for permutation.

    Permutation importance is NOT ADDITIVE: correlated features repair each other's shuffle, so
    each reads far below its joint contribution and a SUM over a correlated block is a floor, not
    an estimate. Do not justify dropping a block by summing its members.
    """
    if isinstance(result, pd.Series):
        s = result
    else:
        imps = result if isinstance(result, dict) else result.attrs.get(key)
        if not imps:
            raise ValueError(f"no {key!r} importances found -- pass a compare_many result or a Series")
        if recipe is None:
            if len(imps) != 1:
                raise ValueError(f"result has {len(imps)} recipes {list(imps)}; pass recipe=")
            recipe = next(iter(imps))
        s = imps[recipe]
    s = s.dropna().sort_values(ascending=False)
    out = pd.DataFrame({"raw": s, "pct": 100 * s / s.sum()})
    out.index.name = "features"
    return out if topn is None else out.head(topn)


def save_comparison(result: pd.DataFrame, path: str) -> list[str]:
    """Save a compare_many result to CSV(s) for later comparison.

    Writes the metric table to '<path>.csv'. Importances are per-feature vectors that can't
    share the flat metric CSV, so any in result.attrs are written to sidecar files
    '<path>_importance.csv' (gain) and '<path>_importance_perm.csv' (permutation, if the run
    used extra_importance_test=True). Each sidecar carries the raw score per recipe (one
    '<recipe>' column each). Returns the list of files written; pair with load_comparison().
    Call this on the object compare_many RETURNED -- reshaping a DataFrame drops .attrs, taking
    the importances with it.
    """
    stem = path[:-4] if path.endswith(".csv") else path
    written = [f"{stem}.csv"]
    result.to_csv(written[0])
    for key, suffix in (("importance", "_importance"), ("importance_perm", "_importance_perm")):
        if result.attrs.get(key):
            raw = importance_table(result, key=key)  # features x recipes, raw mean importance
            fp = f"{stem}{suffix}.csv"
            raw.to_csv(fp)
            written.append(fp)
    print(f"Wrote {len(written)} files:")
    for f in written:
        print(f"  {f}")
    return written


def load_comparison(path: str) -> pd.DataFrame:
    """Load a comparison saved by save_comparison().

    Returns the metric DataFrame with any sidecar importances re-attached to .attrs (as
    {recipe -> Series} of RAW scores), so importance_table(result[, key=...]) works on it
    exactly as on a fresh compare_many result.
    """
    stem = path[:-4] if path.endswith(".csv") else path
    out = pd.read_csv(f"{stem}.csv", index_col="recipe")
    for key, suffix in (("importance", "_importance"), ("importance_perm", "_importance_perm")):
        fp = f"{stem}{suffix}.csv"
        if os.path.exists(fp):
            tab = pd.read_csv(fp, index_col=0)  # features x recipes (raw scores)
            out.attrs[key] = {c: tab[c].dropna() for c in tab.columns}
    return out


# ── train & persist a deployable model ───────────────────


def fit_full(
    recipe: Recipe,
    sites: Sequence[str] | None = None,
    target_col: str = TARGET,
    task: Task = "reg",
    val_frac: float = 0.1,
    seed: int = 42,
    extra_importance_test: bool = False,
    save_path: str | None = None,
    progress: bool | str = False,
    min_rows: int = 500,
    pool: pd.DataFrame | None = None,
    **xgb_kw: Any,
) -> tuple[Model, list[str], dict[str, pd.Series]]:
    """Fit ONE deployable model on the full pooled dataset (no cross-validation holdout).

    cook_many only produces cross-validated fold-models for *scoring* -- each is trained on N-1 sites, so none is a single model fit on all the data. This pools every usable row across `sites` (same pooling cook_many uses) and fits one estimator on 100% of them, suitable for saving and predicting on new sites.

    THE TREE COUNT IS THE CALLER'S. There is no early stopping anywhere (see the module docstring), so `n_estimators` is fit exactly as passed; src/models/train.py reads it from the tuning run and refuses to proceed without one.

    A `val_frac` random slice is carved ONLY when extra_importance_test is on, to serve as the held-out set for permutation importance -- the shipped model is always the one trained on every row. (The slice used to be carved unconditionally, which silently discarded 10% of the data for nothing on every run that did not ask for permutation importance.)

    Always computes GAIN importance (free, from the shipped model). With extra_importance_test=True also computes PERMUTATION importance on the val_frac holdout using a model trained without those rows (honest -- that model never saw them). Both are returned in the importance dict and, if `save_path` is given, written to CSV sidecars.

    `pool` accepts a prebuilt frame from `_pool` so a caller that has already pooled does not pay for it twice -- src/models/train.py scores with cook_many and then fits here on the same rows, which used to build the identical frame from scratch both times.

    If `save_path` is given, also persists the model: '<save_path>' (booster) + '<save_path>.meta.json' + '<stem>_importance.csv' (+ '<stem>_importance_perm.csv').
    """
    if sites is None:
        sites = get_site_ids()
    label = (progress if isinstance(progress, str) else getattr(recipe, "__name__", "recipe")) if progress else None
    if pool is None:
        pool = _pool(recipe, sites, target_col, min_rows=min_rows, progress_label=label)
    feat = _features(pool, target_col)
    X, y = pool[feat], _target(pool, target_col, task)

    val = perm_model = None
    if extra_importance_test:
        val = np.random.RandomState(seed).rand(len(X)) < val_frac
        perm_model = _model(task, **xgb_kw)
        perm_model.fit(X[~val], y[~val], verbose=False)

    m = _model(task, **xgb_kw)  # the shipped model: every row, the caller's tree count
    m.fit(X, y, verbose=False)

    importance = {"importance": pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)}
    if extra_importance_test:  # perm on the holdout, using the model that never saw it
        importance["importance_perm"] = _perm_importance(
            [(perm_model, np.flatnonzero(val))], X, y, feat, task, seed=seed
        )

    if save_path is not None:
        for f in save_model(m, feat, save_path, task=task, target_col=target_col):
            print(f"  wrote {f}")
        stem = save_path[:-5] if save_path.endswith(".json") else save_path
        for f in _save_importance(importance, stem):
            print(f"  wrote {f}")
    return m, feat, importance


def save_model(
    model: Model,
    feat: Sequence[str],
    path: str,
    task: Task = "reg",
    target_col: str = TARGET,
    extra: dict | None = None,
) -> list[str]:
    """Persist a fitted model (XGBoost native booster) plus a sidecar of what's needed to use it.

    Writes '<path>' (the booster in XGBoost's version-robust JSON format) and
    '<path>.meta.json' (the ordered feature columns, task, and target name). The sidecar is what lets you line up new data's columns at predict time. Pair with load_model(). Returns the files written.

    `extra` is merged into the sidecar -- train.py uses it for beta_table / base_rate, so a shipped classifier carries its own decision thresholds and deploy.predict.threshold_for_beta finds them without a separate tuning run.
    """
    model.get_booster().save_model(path)  # booster-level: avoids the sklearn-wrapper save_model quirk
    meta = {"feat": list(feat), "task": task, "target": target_col, "model_class": type(model).__name__}
    meta.update(extra or {})
    with open(f"{path}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return [path, f"{path}.meta.json"]


def _save_importance(importance: dict[str, pd.Series], stem: str) -> list[str]:
    written = []
    for key, suffix in (("importance", "_importance"), ("importance_perm", "_importance_perm")):
        s = importance.get(key)
        if s is not None:
            fp = f"{stem}{suffix}.csv"
            s.rename("importance").rename_axis("features").to_frame().to_csv(fp)
            written.append(fp)
    return written


def load_model(path: str) -> tuple[Model, dict]:
    """Load a model saved by save_model(). Returns (model, meta)."""
    with open(f"{path}.meta.json") as f:
        meta = json.load(f)
    m = xgb.XGBClassifier() if meta["task"] == "clf" else xgb.XGBRegressor()
    m.load_model(path)
    return m, meta
