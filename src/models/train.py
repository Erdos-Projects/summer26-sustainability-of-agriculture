import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))  # repo root on path

from src.eval.cook import compare_many, fit_full, save_model, _pool
from src.features.recipes import recipe_REG, recipe_CLF, light_REG, light_CLF
from src.data.access import get_site_ids

# NO EARLY STOPPING anywhere, which is why the task bases below carry no `n_estimators`: nothing learns a tree count at fit time, so it is a per-recipe TUNED value and lives in RECIPE_XGB with the rest of that recipe's config. A recipe without one cannot be fitted -- build() raises UntunedRecipe rather than defaulting. The rule these configs used to carry (early_stopping_rounds=50) watched a random 15% of ROWS, so it validated in-distribution while the score is out-of-family -- it never fired, and the ceiling silently did the regularizing. See src/eval/cook.py's module docstring.

# current best for regression (from `python _tune.py reg`: lofo_r2 0.343)
REAL_XGB_REG = dict(
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=0.0,
    random_state=42,
)

# current best for classification, but all the results in _tune.py were nearly identical.
REAL_XGB_CLF = dict(
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=0.0,
    random_state=42,
)

# Per-RECIPE overrides, layered on the task base above. THE WHOLE tuned config for a recipe lives here, tree count included: it is the one place a fit's hyperparameters come from, it is in version control, and a diff shows what moved.
#
# The two families are different models -- light_CLF carries 49 columns against recipe_CLF's 51, on a different bucket geometry -- so the depth and learning rate that suit one need not suit the other. tune.py tunes all four separately and prints a paste-ready block per recipe; without somewhere to PUT a light-specific winner, that tuning could be measured and then not applied, which is worse than not measuring it.
#
# A recipe with no `n_estimators` here is UNTUNED and cannot be fitted. That is deliberate: the tree count is only meaningful for the config it was tuned WITH, so taking it from anywhere other than this dict risks pairing one sweep's count with another sweep's depth. tune.py's models/lofo_tune.csv is the tuner's own scoreboard, not an input to training.
RECIPE_XGB = {
    # lofo_r2=0.4440
    "recipe_REG": {
        "n_estimators": 920,
        "max_depth": 4,
        "learning_rate": 0.01,
        "reg_lambda": 1181.04,
        "min_child_weight": 78.736,
        "subsample": 0.5,
    },
    # lofo_auc=0.8710
    # lofo_prauc=0.7175
    "recipe_CLF": {"n_estimators": 570, "max_depth": 5, "reg_lambda": 1.050671, "min_child_weight": 1.050671},
    # lofo_r2=0.4415
    "light_REG": {
        "n_estimators": 890,
        "max_depth": 4,
        "learning_rate": 0.01,
        "reg_lambda": 1181.04,
        "min_child_weight": 78.736,
        "subsample": 0.5,
    },
    # lofo_auc=0.8723
    # lofo_prauc=0.7203
    "light_CLF": {
        "n_estimators": 870,
        "max_depth": 4,
        "learning_rate": 0.02,
        "reg_lambda": 315.201205,
        "min_child_weight": 105.067068,
    },
}


def xgb_for(recipe, task):
    """Effective XGBoost config for one recipe: the task base with that recipe's overrides applied."""
    base = REAL_XGB_REG if task == "reg" else REAL_XGB_CLF
    return {**base, **RECIPE_XGB.get(getattr(recipe, "__name__", str(recipe)), {})}


OUT = Path(__file__).resolve().parent / "models"  # src/models/models -- anchored (CWD-independent, like _LOGFILE)
_LOGFILE = _ROOT / "logs" / "fulltrain_logs.json"

# Default row-share cap for --true-lofo, deliberately looser than cook.folds_true_lofo's own 0.2.
#
# That 0.2 suits a cohort of many comparable families. THIS cohort is 81 sites in only 20 families,
# two of which dominate -- family 0 is 40.9% of pooled rows and family 2 is 27.8%. At a 0.2 cap both
# are barred from ever being the test set, and the resulting lofo_* numbers describe just 31.3% of
# the data while looking exactly like numbers that describe all of it. Measured coverage by cap:
# 0.2 -> 18/20 families and 31.3% of rows, 0.3 -> 19/20 and 59.1%, 0.5 -> 20/20 and 100%.
#
# 0.5 therefore scores every family here. Lower it deliberately (e.g. --true-lofo 0.3) if you
# specifically want the largest family kept out of the test set; the run prints its actual coverage
# either way.
DEFAULT_MAX_HOLDOUT_PCT = 0.5


class UntunedRecipe(RuntimeError):
    """Raised when a deployable fit is asked for and no tuning run has resolved its tree count."""


def _tuned_iters(xgb, recipe_name, task):
    """`xgb`'s tree count, or UntunedRecipe naming the sweep that would produce one.

    No fallback and no default: nothing learns a tree count at fit time, and on the measured LOFO fold running to a 1500 ceiling instead of the 175 trees the holdout wanted gave back 0.0324 R2, ~4x the REG noise floor.
    """
    n = xgb.get("n_estimators")
    if n is not None and not pd.isna(n):
        return int(n)
    family = "light" if recipe_name.startswith("light") else "full"
    raise UntunedRecipe(
        f"No n_estimators for {recipe_name!r} in train.RECIPE_XGB.\n"
        f"There is no early stopping any more, so nothing can learn the tree count at fit time.\n"
        f"Run:  python fulltune.py --family {family} --task {task}\n"
        f"  (or, for a single stage:  python tune.py --family {family} --task {task} --search depth,lr)\n"
        f"then paste the block it prints -- n_estimators included -- into RECIPE_XGB[{recipe_name!r}]."
    )


def _to_native(o):
    """json fallback for the numpy scalars in the metric / importance tables."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def log_metadata(name, recipe, target_col, task, xgb, scores, file=None, true_lofo=False, max_holdout_pct=None):
    """Append one training-run record to _LOGFILE and return the integer key it was filed under.

    `true_lofo` records WHICH LOFO regime produced the lofo_* columns. They are the identical column
    names either way, and true LOFO trains each fold on more data and so scores HIGHER -- so without
    this field a run is not comparable to its neighbours in the log and nothing in the numbers says
    so. False means the GroupKFold default; a float `max_holdout_pct` accompanies a true run.
    """
    imp = scores.attrs.get("importance", {}).get(name)
    imp_perm = scores.attrs.get("importance_perm", {}).get(name)
    model_entry = {
        "name": name,
        "recipe": getattr(recipe, "__name__", str(recipe)),  # a recipe is a function -> store its name
        "features": list(imp.index) if imp is not None else [],  # full feature column list (gain-ranked)
        "target_col": target_col,
        "task": task,
        # Which CV/score generation produced this row. 1 = every aggregate computed from the LOSO
        # OOF, models early-stopped on a random 15% row slice. 2 = aggregates from the LOFO OOF,
        # no early stopping, tuned tree count. The two are NOT comparable, and nothing in the
        # numbers themselves says so.
        "cv_schema": 2,
        "true_lofo": bool(true_lofo),
        # Recorded whatever the regime: read alongside true_lofo, which says whether it was applied.
        "max_holdout_pct": max_holdout_pct,
        "xgb": dict(xgb),
        "score": scores.loc[name].to_dict(),  # the scalar metric row (n_sites, lofo_r2, ...)
        "importance": imp.to_dict() if imp is not None else {},  # feature -> mean gain
        "importance_perm": imp_perm.to_dict() if imp_perm is not None else {},  # feature -> perm importance
    }
    # Decision thresholds, for classifiers. Derived from the LOFO out-of-fold vector the CV already
    # produced, so every run carries a reproducible beta table without a second grouped CV.
    ops = (scores.attrs.get("operating_points") or {}).get(name)
    if ops:
        model_entry |= {
            "beta_table": ops["beta_table"],
            # Prevalence IN THE SCORED ROWS, which is not the pooled `base` in `score` when the OOF
            # covers only part of the pool (a capped --true-lofo run). FDR is prevalence-dependent,
            # so the two must not be read interchangeably.
            "base_rate": ops["base_rate"],
            "beta_table_coverage": ops["coverage"],
            "pr_curve": ops["pr_curve"],
        }

    _LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    if _LOGFILE.exists() and _LOGFILE.stat().st_size > 0:
        with open(_LOGFILE) as f:
            log = json.load(f)
    else:
        log = {}
    key = max((int(k) for k in log), default=-1) + 1  # next integer not currently in the log
    log[str(key)] = model_entry
    with open(_LOGFILE, "w") as f:
        json.dump(log, f, indent=2, default=_to_native)
    return key


def build(name, recipe, target_col, task, xgb, final_iters=None, min_rows=300, true_lofo=False, max_holdout_pct=0.2):
    """CV-score `recipe`, log the run, then fit and save the deployable model on 100% of the rows.

    The tree count is resolved BEFORE the CV, not after: a CV pass is the expensive half of this and there is no point paying for it only to discover at the end that nothing can be shipped.
    """
    # first check the directory exists before committing
    OUT.mkdir(exist_ok=True)

    # The tuned count also becomes the CV's n_estimators, so the numbers logged below describe the
    # model that actually ships rather than one fitted at a different depth of boosting.
    if final_iters is None:
        final_iters = _tuned_iters(xgb, getattr(recipe, "__name__", ""), task)
    final_iters = int(final_iters)
    xgb = {**xgb, "n_estimators": final_iters}

    print(f"\n===== {name} ({task}) =====")
    print(f"  tree count: {final_iters} (tuned; there is no early stopping)")

    # Pool ONCE. The CV and the deployable fit want the identical frame, and each used to build it
    # from scratch -- on this cohort that is the single most expensive thing a training run does.
    print("[1/3] pooling...")
    pool = _pool(recipe, get_site_ids(), target_col, min_rows=min_rows, progress_label=name)

    lofo_regime = f"true LOFO, holdout cap {max_holdout_pct:.0%}" if true_lofo else "LOFO via GroupKFold"
    print(f"[2/3] evaluating cross-site (LOSO/{lofo_regime})...")
    scores = compare_many(
        {name: recipe},
        sites=None,
        target_col=target_col,
        task=task,
        extra_importance_test=True,
        min_rows=min_rows,
        true_lofo=true_lofo,
        max_holdout_pct=max_holdout_pct,
        pool=pool,
        **xgb,
    )
    print(scores.round(3).to_string())
    key = log_metadata(
        name, recipe, target_col, task, xgb, scores, true_lofo=true_lofo, max_holdout_pct=max_holdout_pct
    )
    print(f"  logged run #{key} -> {_LOGFILE}")

    # then fit the deployable model on ALL rows, at the same tuned tree count the CV just scored
    print(f"[3/3] fitting deployable model on all rows ({final_iters} trees, from the tuning run)...")
    model, feat, _imp = fit_full(
        recipe, sites=None, target_col=target_col, task=task, progress=True, min_rows=min_rows, pool=pool, **xgb
    )
    # The classifier ships with its own operating points, so deploy.predict.threshold_for_beta finds
    # them straight away and the widget's beta slider works on a freshly trained model.
    ops = (scores.attrs.get("operating_points") or {}).get(name)
    extra = (
        {"beta_table": ops["beta_table"], "base_rate": ops["base_rate"], "beta_table_coverage": ops["coverage"]}
        if ops
        else None
    )
    for f in save_model(model, feat, str(OUT / f"{name}.json"), task=task, target_col=target_col, extra=extra):
        print(f"  wrote {f}")
    if ops:
        print(
            f"  beta table: {len(ops['beta_table'])} operating points, base rate {ops['base_rate']:.3f}, "
            f"from {ops['coverage']:.1%} of pooled rows"
        )


def main():
    parser = argparse.ArgumentParser(description="Train + log the shipped recipe(s).")
    # --false-alarm-rate is gone with recall_at_far, which is no longer a reported column: it was a
    # hindsight maximum over a threshold sweep, and it was never recorded in the log entry either,
    # so runs at different budgets were distinguishable only by their names. The honest operating
    # points come from the beta_table below.
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the trained model: the booster is saved to models/<name>_{REG,CLF}.json and the "
        "run is logged under that name in fulltrain_logs.json. Default: recipe_CLF2, or 'light' with --light.",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Train light_REG / light_CLF instead of the full recipes: the reduced feature set the static "
        "widget can build client-side (basin-mean weather limited to fuel_moisture_1000h + precip_in_1d, "
        "single distance bucket). See src/features/recipes.py.",
    )
    parser.add_argument(
        "--true-lofo",
        type=float,
        nargs="?",
        const=DEFAULT_MAX_HOLDOUT_PCT,
        default=None,
        metavar="MAX_HOLDOUT_PCT",
        help="Score LOFO by holding out ONE basin family at a time and training on every other site, "
        "instead of GroupKFold(5) which holds out about a fifth of the families at once. The value in "
        "(0, 1] caps the share of pooled ROWS a single held-out family may occupy -- a family above it "
        "is still trained on, but never becomes the test set, so one oversized family cannot dominate "
        f"the pooled OOF. Default {DEFAULT_MAX_HOLDOUT_PCT} when the flag is given without a value, "
        "which scores every family in this cohort. NOTE that on THIS cohort it barely moves the "
        "numbers -- the two largest families are 40.9% and 27.8% of rows, too big to share a "
        "fifth-sized fold, so GroupKFold already gives each its own fold and the regimes coincide on "
        "68.7% of the data (measured: lofo_r2 0.3231 vs 0.3706, lofo_prauc 0.6794 vs 0.7004). The "
        "run is still logged with true_lofo=true so it is not compared against GroupKFold runs by "
        "accident. See cook.cook_many for when the distinction does matter.",
    )
    args = parser.parse_args()
    if args.true_lofo is not None and not 0.0 < args.true_lofo <= 1.0:
        parser.error("--true-lofo must be in (0, 1]")

    reg, clf = (light_REG, light_CLF) if args.light else (recipe_REG, recipe_CLF)
    # Separate default name so a --light run cannot overwrite the shipped boosters, and so the two
    # sit side by side in fulltrain_logs.json for comparison.
    name = args.name or ("light" if args.light else "recipe")
    if args.light:
        print("[cfg] training the LIGHT recipes (static-site feature set)")

    # build() takes the tuned n_estimators from the recipe's RECIPE_XGB entry and raises UntunedRecipe if it has none -- there is no early stopping to fall back on.
    lofo_kw = dict(true_lofo=args.true_lofo is not None, max_holdout_pct=args.true_lofo or DEFAULT_MAX_HOLDOUT_PCT)
    if args.true_lofo is not None:
        print(f"[cfg] TRUE LOFO: one family held out per fold, capped at {args.true_lofo:.0%} of pooled rows")

    try:
        build(name + "_REG", reg, target_col="nitrate_con", task="reg", xgb=xgb_for(reg, "reg"), **lofo_kw)
        build(name + "_CLF", clf, target_col="violation", task="clf", xgb=xgb_for(clf, "clf"), **lofo_kw)
    except UntunedRecipe as e:
        raise SystemExit(f"\n{e}")


if __name__ == "__main__":
    main()
