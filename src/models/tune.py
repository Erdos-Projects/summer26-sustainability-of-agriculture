"""tune.py -- grid-search XGBoost hyperparameters against the HONEST LOFO metric.

Tunes any of FOUR recipes: the shipped pair (recipe_REG / recipe_CLF) and the light pair (light_REG / light_CLF) the static widget scores with. They need separate tuning rather than shared numbers -- light_CLF carries 49 columns against recipe_CLF's 51 and a different bucket geometry, so the depth and learning rate that suit one need not suit the other.

Each recipe's pool is the expensive part, so it is built ONCE per recipe and every config re-uses the same X / y / folds. A config therefore costs only its CV fits, and because all configs share one pool the comparison is exactly paired.

THE TREE COUNT IS NOT AN AXIS -- it is resolved per config. Early stopping used to choose it, and chose it badly (see src/eval/cook.py's module docstring), so every config now fits each fold once at a `--ceiling` and scores every PREFIX of that fit on the holdout. Each config reports the tree count it actually wanted (`best_k`), its score there, and `k_frac = best_k / ceiling` -- a k_frac near 1.0 means the ceiling is binding rather than chosen, and should be raised. This costs the same two CV passes scoring already paid, versus ~log2(ceiling) x n_splits fits for a bisection, and unlike a bisection it cannot be fooled by the measured curve, which is not unimodal.

WHICH AXES. `--search` names the axes crossed into the grid; every other axis takes its value from `--fix`, else from the inherited default. Staging is therefore a sequence of INVOCATIONS, not edits. Axes: depth, lr, lam, mcw, subsample, colsample.

    # Stage 1 — capacity x learning rate
    python tune.py --family light --task reg --search depth,lr --ceiling 1500
        -> read best config AND k_frac. k_frac > ~0.8 means the ceiling is binding, not chosen: raise --ceiling and repeat before trusting anything.

    # Stage 2 — the one genuine cartesian, at stage 1's lr
    python tune.py --family light --task reg --search depth,lam --fix lr=0.02 --append

    # Stage 3 — coordinate, each pinning everything settled so far
    python tune.py --family light --task reg --search mcw       --fix depth=4,lr=0.02,lam=0.1 --append
    python tune.py --family light --task reg --search subsample --fix depth=4,lr=0.02,lam=0.1,mcw=0.02 --append

Or run the whole protocol with `python fulltune.py --family light --task reg`.

WHY THE AXES ARE NOT ALL INDEPENDENT. Two regimes, and the grid design follows from them:

- `max_depth` x {`reg_lambda`, `min_child_weight`} are strongly coupled in RAW units, because both regularizers compete against a leaf's accumulated Hessian and that scales as N/2^depth. Fixing lambda at one depth and then optimizing depth would be a genuine mistake -- you would never visit (deep, large-lambda). Hence they are expressed RELATIVELY (below) and crossed with depth anyway.
- `min_child_weight` x `reg_lambda`, and the sampling fractions against everything else, are only weakly coupled -- one blocks splits, the other shrinks the leaves of splits that survive. Coordinate descent is sound for these.

`learning_rate` and `n_estimators` USED to be a third, totally-coupled regime -- with early stopping never firing, every config ran its full budget, so only the product `lr x n_trees` mattered. Resolving the tree count per config dissolves that: lr is a learning rate again and the optimal count simply adapts to it.

RELATIVE UNITS. `--lams` and `--mcws` are given as a FRACTION of the expected per-leaf Hessian, not in raw XGBoost units, and are converted per-config once depth is known. Leaf weight is -G/(H+lambda), so lambda = c x H yields a shrink of 1/(1+c) at EVERY depth: the axis becomes depth-invariant, which is what lets a modest cartesian cover the space. Two consequences worth knowing:

- It exposes how inert the historical values were. reg_lambda=5 at depth 4 is c=0.0009 -- a 0.09% shrink. min_child_weight=10 is 0.2% of a typical leaf. Nothing in a sweep over those was ever going to move.
- It unifies the REG and CLF ladders. REG's Hessian is 1/row; CLF's is p(1-p), ~0.17 at the observed base rate, so the same relative c produces the ~5.7x smaller absolute CLF value automatically. One ladder, both tasks. `--absolute` opts out and passes the numbers through untouched.

OUTPUT. Each recipe upserts one self-describing row into models/lofo_tune.csv keyed on its OWN name, so the four coexist and re-tuning one replaces only its line. train.py::_tuned_iters reads `n_estimators` from that file by recipe name, and REFUSES to fit a deployable model without it. The per-recipe grid also lands in models/tune_<recipe>.csv (`--append` accumulates stages).
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

import src.eval.cook as cook
from src.data.access import get_site_ids

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "models"  # src/models/models -- anchored, so any cwd works (like train.OUT)
LOFO_FILE = OUT / "lofo_tune.csv"  # shared summary: one upserted best-row per recipe

# (family, task) -> (recipe attr on the module, target column).
_RECIPE_SPEC = {
    ("full", "reg"): ("recipe_REG", "nitrate_con"),
    ("full", "clf"): ("recipe_CLF", "violation"),
    ("light", "reg"): ("light_REG", "nitrate_con"),
    ("light", "clf"): ("light_CLF", "violation"),
}

# The metric each task is ranked by -- the same names train.py reports as its headline, so a config
# that wins here is a config that wins there. NOTE this changed with the score rework: CLF used to
# rank on auc, which disagrees with prauc about what "best" means at a ~26% violation base rate.
_HEADLINE = {"reg": "lofo_r2", "clf": "lofo_prauc"}
# The same quantity under the name cook._score returns it by. _score emits bare metric names
# (r2, prauc, auc, rmse); the lofo_/loso_ prefixes are attached later by _cross_metrics, which is
# what the scoreboard reports. The tree-count scan calls _score directly -- one metric, thousands of
# times -- so it needs the bare name.
_CURVE_METRIC = {"reg": "r2", "clf": "prauc"}
# reported next to it, so a config that buys LOFO by widening the leakage gap is visible.
# lofo_between_r2 leads for REG on purpose: depth is bought almost entirely in the between-site
# ranker, which is the known bottleneck.
_ALSO = {
    "reg": ["lofo_between_r2", "lofo_within_r2", "loso_r2", "lofo_rmse"],
    "clf": ["loso_auc", "lofo_auc", "lofo_prauc_lift", "lofo_between_rate_r2"],
}
_INT_KEYS = {"max_depth", "n_estimators"}
# tree-count scan granularity: a peak narrower than _COARSE is invisible to it (see
# _score_pool_curve), and the measured peak spans >150 trees, so 50 leaves ~3x margin
_COARSE, _FINE = 50, 10

# axis name -> (xgb parameter it sets, default ladder, given in relative units?).
# `trees` is NOT an axis: crossing it would cost a refit per value, and one fit at a ceiling already
# yields every prefix for free (see _score_pool_curve).
_AXES = {
    "depth": ("max_depth", (3, 4, 5, 6), False),
    "lr": ("learning_rate", (0.01, 0.02, 0.05), False),
    "lam": ("reg_lambda", (0.002, 0.02, 0.1, 0.3), True),
    "mcw": ("min_child_weight", (0.002, 0.02, 0.1, 0.3), True),
    "subsample": ("subsample", (0.5, 0.7, 0.9), False),
    "colsample": ("colsample_bytree", (0.5, 0.7, 0.9), False),
}

# axis -> (hard lower limit, hard upper limit, floor-is-already-inert). A winner sitting on a HARD
# limit is a real answer; a winner sitting on a ladder end that is not a hard limit means the search
# was truncated.
_AXIS_LIMIT = {
    "depth": (1, None, False),
    "lr": (0.0, None, False),
    "lam": (0.0, None, True),  # c<=0.002 is a <0.2% leaf shrink -- indistinguishable from no L2 at all
    "mcw": (0.0, None, True),  # likewise: a fraction this small never blocks a split
    "subsample": (0.0, 1.0, False),
    "colsample": (0.0, 1.0, False),
}


def _leaf_hessian(depth: int, subsample: float, n_rows: int, train_frac: float, hess_per_row: float) -> float:
    """Expected sum-of-Hessian in one leaf of a depth-`depth` tree.

    (n_rows x train_frac x subsample) / 2^depth rows, each contributing `hess_per_row` -- 1 for squared error, p(1-p) for logistic. Approximate (real trees are unbalanced) but the right SCALE, which is what makes relative lam/mcw depth-invariant.
    """
    return (n_rows * train_frac * subsample) / (2**depth) * hess_per_row


def _materialize(axis_vals: dict, ctx: dict, absolute: bool) -> dict:
    """Axis point -> XGBoost override dict. Relative lam/mcw resolve against the leaf Hessian implied by THIS point's depth and subsample, not a grid-wide average."""
    cfg: dict = {}
    for name, v in axis_vals.items():
        param, _ladder, _rel = _AXES[name]
        if param is not None and name not in ("lam", "mcw"):
            cfg[param] = v

    depth = int(cfg.get("max_depth", ctx["default_depth"]))
    subsample = float(cfg.get("subsample", ctx["default_subsample"]))

    if "lam" in axis_vals or "mcw" in axis_vals:
        h = _leaf_hessian(depth, subsample, ctx["n_rows"], ctx["train_frac"], ctx["hess_per_row"])
        for name, param in (("lam", "reg_lambda"), ("mcw", "min_child_weight")):
            if name in axis_vals:
                cfg[param] = float(axis_vals[name]) if absolute else float(axis_vals[name]) * h

    return {k: (int(v) if k in _INT_KEYS else v) for k, v in cfg.items()}


def _grid(search: list[str], fixed: dict, ladders: dict, ctx: dict, absolute: bool) -> list[tuple[dict, dict]]:
    """(axis point, xgb override) per cell of the cartesian over `search`, with `fixed` merged in. Leads with the inherited default as the reference row."""
    ladders = {a: ladders.get(a) or _AXES[a][1] for a in search}
    out: list[tuple[dict, dict]] = [({}, {})]  # {} = the inherited base config
    for combo in itertools.product(*(ladders[a] for a in search)):
        pt = {**fixed, **dict(zip(search, combo))}
        out.append((pt, _materialize(pt, ctx, absolute)))
    return out


def _label(pt: dict, cfg: dict) -> str:
    if not pt:
        return "DEFAULT (inherited)"
    bits = []
    for k, v in pt.items():
        if k in ("lam", "mcw"):
            param = _AXES[k][0]
            bits.append(f"{k}={v:g}({cfg[param]:.0f})")  # relative(absolute)
        else:
            bits.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(bits)


def _flag(axis: str) -> str:
    """CLI flag for an axis ladder: the axis name, pluralised."""
    return f"{axis}s"


def _predict_prefix(m, X, task: str, k: int) -> np.ndarray:
    """Prediction using only the FIRST k trees of an already-fitted model. `iteration_range` is what makes the whole tree-count search cost one fit instead of one fit per candidate."""
    return m.predict_proba(X, iteration_range=(0, k))[:, 1] if task == "clf" else m.predict(X, iteration_range=(0, k))


def _score_pool_curve(
    pool, feat, target, task, metric, ceiling, n_splits, true_lofo, max_holdout_pct, coarse=_COARSE, fine=_FINE, **cfg
):
    """Score a config, choosing its tree count rather than being told it. -> (metrics_at_best_k, best_k, curve).

    Fit each fold once at `ceiling`, then score every PREFIX on the holdout via iteration_range=(0, k) -- no refits, so the whole curve plus final metrics cost the same two CV passes a plain score already paid. Scanned rather than bisected because the curve is not unimodal. Coarse-then-fine since a prefix costs O(k); `metric` is a bare cook._score key (r2/prauc), not a lofo_-prefixed one.
    """
    X, y = pool[feat], cook._target(pool, target, task)
    fam = cook.basin_groups(pool["site"])
    kw = dict(n_estimators=ceiling, **cfg)

    if true_lofo:
        fam_models = [(m, te) for te, m in cook._fold_models(X, y, cook.folds_true_lofo(fam, max_holdout_pct), task, **kw)]
    else:
        fam_models = [(m, te) for te, m in cook._grouped_models(X, y, fam, task, n_splits, **kw)]

    def curve_over(ks):
        out = {}
        for k in ks:
            oof = np.full(len(y), np.nan)
            for m, te in fam_models:
                oof[te] = _predict_prefix(m, X.iloc[te], task, k)
            out[k] = (cook._score(y, oof, task)[metric], oof)
        return out

    scan = curve_over(range(coarse, ceiling + 1, coarse))
    # Refine around the top TWO coarse points rather than only the argmax. The curve is bimodal --
    # on the measured fold it peaked +0.350 at 175 then recovered to a +0.317 plateau by 1375, a
    # margin of only ~0.03 -- so when both basins are visible to the coarse grid but rank in the
    # wrong order, the second window recovers the true peak.
    #
    # What this does NOT fix: a peak NARROWER than `coarse` is invisible to the coarse grid entirely
    # (both flanking samples read the base level), so the top-2 coarse points both sit in the other
    # basin and neither window covers it. The only real protection there is `coarse` itself, which
    # is why it is 50 rather than 100: the observed peak spans >150 trees, so 50 leaves a 3x margin.
    ranked = sorted(scan, key=lambda k: scan[k][0], reverse=True)
    windows = set()
    for k0 in ranked[:2]:
        windows.update(range(max(fine, k0 - coarse), min(ceiling, k0 + coarse) + 1, fine))
    scan.update(curve_over(sorted(windows - set(scan))))
    best_k = max(scan, key=lambda k: scan[k][0])
    oof_family = scan[best_k][1]

    # the LOSO pass, prefixed at the SAME k, so loso_* reports the leakage gap for the model chosen
    site_models = [(m, te) for te, m in cook._grouped_models(X, y, pool["site"], task, n_splits, **kw)]
    oof_site = np.full(len(y), np.nan)
    for m, te in site_models:
        oof_site[te] = _predict_prefix(m, X.iloc[te], task, best_k)

    metrics = dict(
        n_sites=pool["site"].nunique(),
        n_families=int(pd.Series(fam).nunique()),
        n_rows=len(pool),
        n_feat=len(feat),
        **cook._cross_metrics(pool, np.asarray(y), oof_site, oof_family, task),
    )
    return metrics, best_k, {k: v[0] for k, v in sorted(scan.items())}


def _parse_pairs(s: str | None) -> dict:
    """'depth=4,lam=0.1' -> {'depth': 4.0, 'lam': 0.1}. Axis names only -- a raw XGBoost name like reg_lambda=0.1 is rejected, since it would mean thousands of times less than the axis of the same name."""
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in _AXES:
            raise SystemExit(f"--fix: {k!r} is not an axis. Axes: {', '.join(_AXES)}")
        out[k] = float(v)
    return out


def _parse_list(s: str | None, cast=float):
    return None if not s else tuple(cast(x) for x in s.split(","))


def _upsert_lofo(row):
    """Write `row` into models/lofo_tune.csv, replacing any existing row for the same recipe.

    Keyed on the 'recipe' column: tuning a recipe again overwrites its line; a new recipe name appends a line. `n_estimators` is the field train.py reads -- it is the TUNED tree count, not an early-stopping best_iteration, and train.py refuses to ship a model without it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if LOFO_FILE.exists():
        old = pd.read_csv(LOFO_FILE)
        old = old[old["recipe"] != row["recipe"]]  # drop the prior row for this recipe
        out = pd.concat([old, new], ignore_index=True)
        cols = list(row) + [c for c in old.columns if c not in row]  # keep any legacy extra cols
        out = out.reindex(columns=cols)
    else:
        out = new
    out = out.sort_values(["task", "recipe"]).reset_index(drop=True)
    out.to_csv(LOFO_FILE, index=False)
    return out


def _emit_config(df: pd.DataFrame, search: list[str], recipe_name: str, base: dict, fixed: dict | None = None) -> None:
    """Print the winner as a paste-ready RECIPE_XGB entry, plus which axes are still at inherited values.

    Only the keys that MOVED against `base` -- echoing the whole config would bury the two or three swept keys that changed among nine that did not. n_estimators is excluded on purpose: train.py reads the tree count from lofo_tune.csv, not from RECIPE_XGB.

    An axis counts as RESOLVED if this run searched it OR `fixed` pins it, because fulltune.py threads each stage's winner into the next as --fix: without that, every stage after the first would report the axes it had already settled as untuned, and the final confirmation stage -- which pins all six -- would report the completed config as tuning nothing at all.
    """
    resolved = set(search) | set(fixed or {})
    best = df.iloc[0]
    changed = {}
    for axis in _AXES:
        param = _AXES[axis][0]
        if param not in best or pd.isna(best.get(param)):
            continue
        v = best[param]
        v = v.item() if hasattr(v, "item") else v  # numpy scalar -> native, else repr() prints np.float64
        v = int(v) if param in _INT_KEYS else round(float(v), 6)
        if base.get(param) != v:
            changed[param] = v

    print(f"\n  n_estimators={int(best['best_k'])} is written to {LOFO_FILE.name} automatically (train.py reads it).")
    if changed:
        print(f"  To adopt the rest, put these in train.RECIPE_XGB[{recipe_name!r}]:")
        print(f'      "{recipe_name}": {{' + ", ".join(f"{k!r}: {v!r}" for k, v in changed.items()) + "},")
    else:
        print(f"  Swept parameters match the base config -- nothing to add to train.RECIPE_XGB[{recipe_name!r}].")

    untouched = [a for a in _AXES if a not in resolved]
    if untouched:
        print(f"  [partial] NOT resolved: {', '.join(untouched)} -- still at inherited values. "
              f"lam/mcw in particular are inert at the inherited reg_lambda=5 / min_child_weight=10. "
              f"`python fulltune.py` resolves all six.")


def _edge_report(df: pd.DataFrame, search: list[str], ladders: dict, ceiling: int) -> None:
    """Warn when the winning config sits on the edge of a searched ladder, or on the tree ceiling.

    An edge winner means the search was TRUNCATED, not that an optimum was found -- unless the edge is a hard limit (subsample/colsample at 1.0) or an already-inert floor (lam/mcw at ~0, which is indistinguishable from no regularization). Prints the ladder to try next; extending is left to the caller, since auto-widening can run away.
    """
    best = df.iloc[0]
    notes = []
    if pd.notna(best.get("best_k")) and best["best_k"] >= 0.8 * ceiling:
        notes.append(f"best_k={int(best['best_k'])} is >=80% of the {ceiling} ceiling -- the ceiling is binding, "
                     f"not chosen. Re-run with a larger --ceiling before trusting this winner.")
    for ax in search:
        v = best.get(f"ax_{ax}")
        lad = sorted(ladders.get(ax) or _AXES[ax][1])
        if pd.isna(v) or len(lad) < 2:
            continue
        lo_lim, hi_lim, inert_floor = _AXIS_LIMIT[ax]
        if v >= lad[-1] and (hi_lim is None or lad[-1] < hi_lim):
            step = lad[-1] - lad[-2]
            nxt = [round(lad[-1] + step * i, 6) for i in (1, 2)]
            if hi_lim is not None:
                nxt = [x for x in nxt if x <= hi_lim] or [hi_lim]
            notes.append(f"{ax}={v:g} is the TOP of its ladder {lad} -- untruncated optimum unknown. "
                         f"Try --{_flag(ax)} {','.join(f'{x:g}' for x in lad[-2:] + nxt)}")
        elif v <= lad[0]:
            if inert_floor:
                notes.append(f"{ax}={v:g} is the bottom of its ladder, and that low it is already inert -- "
                             f"read as 'no {ax} helps', not as a truncated search.")
            elif lad[0] > lo_lim:
                step = lad[1] - lad[0]
                nxt = [round(lad[0] - step * i, 6) for i in (1, 2)]
                nxt = [x for x in nxt if x > lo_lim]
                if nxt:
                    notes.append(f"{ax}={v:g} is the BOTTOM of its ladder {lad} -- untruncated optimum unknown. "
                                 f"Try --{_flag(ax)} {','.join(f'{x:g}' for x in sorted(nxt) + lad[:2])}")
    for n in notes:
        print(f"  [edge] {n}")


def tune(
    task: str,
    recipe,
    target_col: str,
    sites,
    search: list[str],
    fixed: dict,
    ladders: dict,
    absolute: bool,
    ceiling: int,
    seeds: tuple[int, ...],
    append: bool,
    true_lofo: bool,
    max_holdout_pct: float = 0.2,
    n_splits: int = 5,
    min_rows: int = 500,
    pool: pd.DataFrame | None = None,
):
    """Sweep one recipe. `pool` lets fulltune.py share one pooling pass across all six stages."""
    from src.models.train import xgb_for  # local: train imports cook, and cook is heavy

    name = getattr(recipe, "__name__", str(recipe))
    head = _HEADLINE[task]
    base = xgb_for(recipe, task)  # the inherited config the DEFAULT row measures, and _emit_config diffs against

    if pool is None:
        pool = cook._pool(recipe, sites, target_col, min_rows=min_rows, progress_label=name)
    feat = cook._features(pool, target_col)
    y = cook._target(pool, target_col, task)
    fam = cook.basin_groups(pool["site"])
    n_fam = int(pd.Series(fam).nunique())

    # Hessian per row: exactly 1 for squared error; p(1-p) for logistic, at the pool's own base rate.
    # This is the entire reason the same relative ladder yields different absolute values per task.
    p = float(np.mean(y)) if task == "clf" else float("nan")
    hess_per_row = p * (1 - p) if task == "clf" else 1.0
    # one family held out per fold under true LOFO, else GroupKFold's (k-1)/k
    train_frac = (1 - 1 / n_fam) if true_lofo else (1 - 1 / min(n_splits, n_fam))
    ctx = dict(
        n_rows=len(pool),
        train_frac=train_frac,
        hess_per_row=hess_per_row,
        ceiling=ceiling,
        default_depth=base["max_depth"],
        default_subsample=base["subsample"],
    )

    grid = _grid(search, fixed, ladders, ctx, absolute)
    print(
        f"\n=== {name} ({task.upper()})  pool {pool.shape}  |  {len(feat)} features  |  "
        f"{pool['site'].nunique()} sites, {n_fam} families  |  {len(grid)} configs x {len(seeds)} seed(s) ==="
    )
    print(f"    ranking on {head}" + (f"   [true LOFO, cap {max_holdout_pct:.0%}]" if true_lofo else ""))
    print(f"    searching {search or ['(none)']}" + (f", fixed {fixed}" if fixed else ""))
    if not absolute:
        eff = ctx["n_rows"] * train_frac * ctx["default_subsample"]
        scale = "  ".join(
            f"d{d}:{_leaf_hessian(d, ctx['default_subsample'], ctx['n_rows'], train_frac, hess_per_row):,.0f}"
            for d in (3, 4, 5, 6)
        )
        print(
            f"    leaf Hessian ({eff:,.0f} rows/tree, {hess_per_row:.4f}/row) -> {scale}   "
            f"[relative units; c=0.1 means a 9% leaf shrink]"
        )
    print(f"    tree count: scanned to a {ceiling} ceiling per config, coarse {_COARSE} then fine {_FINE}")
    print()

    rows, t0 = [], time.time()
    for i, (pt, cfg) in enumerate(grid, 1):
        t1 = time.time()
        for seed in seeds:
            kw = {**base, **cfg, "random_state": seed}
            kw.pop("n_estimators", None)  # the scan supplies the ceiling; a base value would fight it
            r, best_k, curve = _score_pool_curve(
                pool, feat, target_col, task, _CURVE_METRIC[task], ceiling, n_splits,
                true_lofo, max_holdout_pct, **kw,
            )
            rows.append(
                {
                    "config": _label(pt, cfg),
                    # `searched` identifies which run a row came from. Under --append the file
                    # accumulates stages and sorting by headline interleaves them, so without it a
                    # stage-1 row and a stage-3 row are distinguishable only by guessing from which
                    # ax_* happen to be non-NaN.
                    "searched": ",".join(search) or "none",
                    "seed": seed,
                    head: r[head],
                    **{k: r.get(k) for k in _ALSO[task]},
                    "best_k": best_k,  # trees the config actually wanted
                    "k_frac": round(best_k / ceiling, 3),  # ~1.0 means the ceiling is binding -- raise it
                    "at_ceiling": round(curve[max(curve)], 4),
                    # the EFFECTIVE config, not just the overrides, so each row says what was fit
                    # rather than what differed
                    **{f"ax_{k}": v for k, v in pt.items()},
                    **{**kw, "n_estimators": best_k},
                }
            )
        got = [x[head] for x in rows[-len(seeds):]]
        ks = [x["best_k"] for x in rows[-len(seeds):]]
        seed_note = f"  [{len(seeds)} seeds, sd {np.std(got):.4f}]" if len(seeds) > 1 else ""
        print(
            f"  [{i}/{len(grid)}] {_label(pt, cfg):<50.50} {head}={np.mean(got):.4f}  k={int(np.mean(ks)):>5}"
            f"  (vs {np.mean([x['at_ceiling'] for x in rows[-len(seeds):]]):+.4f} at {max(curve)})"
            f"{seed_note}  ({time.time() - t1:.0f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    if len(seeds) > 1:
        # Rank on the seed MEAN: picking the max over a grid of noisy single-seed scores is
        # optimistically biased, and subsample<1 makes every fit seed-dependent. Averaging only the
        # numeric columns on purpose -- a config column that is None/str would make .mean() raise
        # here, i.e. AFTER every fit in the sweep has been paid for.
        num = [c for c in df.columns if c != "config" and pd.api.types.is_numeric_dtype(df[c])]
        rest = [c for c in df.columns if c != "config" and c not in num]
        df = df.groupby("config", as_index=False).agg({**{c: "mean" for c in num}, **{c: "first" for c in rest}})
        df["n_seeds"] = len(seeds)
    df = df.sort_values(head, ascending=False).reset_index(drop=True)
    ref = df.loc[df["config"] == "DEFAULT (inherited)", head]
    if len(ref):
        df.insert(2, "delta_vs_default", (df[head] - float(ref.iloc[0])).round(4))

    OUT.mkdir(parents=True, exist_ok=True)
    # Keyed on the RECIPE, not the task: tune_reg.csv would have had light_REG silently overwrite
    # recipe_REG's grid on any run that covered both.
    grid_fp = OUT / f"tune_{name}.csv"
    if append and grid_fp.exists():
        df_out = pd.concat([pd.read_csv(grid_fp), df], ignore_index=True).sort_values(head, ascending=False)
    else:
        df_out = df
    df_out.to_csv(grid_fp, index=False)

    best = df.iloc[0]
    print(f"\n  best: {best['config']}  ({head}={best[head]:.4f}, {int(best['best_k'])} trees)")
    _edge_report(df, search, ladders, ceiling)

    # one self-describing best-row per recipe, upserted into the shared summary file
    winner = {k: best[k] for k in base if k in best.index}
    winner["n_estimators"] = int(best["best_k"])
    summary = {
        "recipe": name,
        "task": task,
        "metric": head,
        "score": round(float(best[head]), 6),
        "searched": best["searched"],
        "ceiling": ceiling,
        "k_frac": float(best["k_frac"]),
        "true_lofo": bool(true_lofo),
        **{k: (int(v) if k in _INT_KEYS else v) for k, v in winner.items()},
    }
    _upsert_lofo(summary)
    _emit_config(df, search, name, base, fixed)
    print(f"  upserted best row (recipe={name!r}) -> {LOFO_FILE}")
    print(f"  wrote {grid_fp}  ({len(df_out)} rows, {time.time() - t0:.0f}s)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["reg", "clf", "both"], default="both")
    ap.add_argument(
        "--family",
        default="both",
        choices=["full", "light", "both"],
        help="which recipe family: 'full' = recipe_REG/recipe_CLF (shipped), 'light' = light_REG/light_CLF "
        "(the static widget's set), 'both' (default). Combined with --task, so the default tunes all four.",
    )
    ap.add_argument("--search", default="depth,lr", help=f"axes to cross: {', '.join(_AXES)}")
    ap.add_argument("--fix", default=None, help="pin axes not searched, e.g. 'depth=4,lam=0.1'")
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="read --lams/--mcws as raw XGBoost units instead of fractions of the per-leaf Hessian",
    )
    ap.add_argument(
        "--ceiling",
        type=int,
        default=1500,
        help="upper bound on trees; each config is scanned to it and reports the count it actually wanted",
    )
    ap.add_argument("--seeds", default="42", help="comma-separated random_state values; >1 ranks on the seed mean")
    ap.add_argument(
        "--append", action="store_true", help="append to tune_<recipe>.csv instead of overwriting, so stages accumulate"
    )
    ap.add_argument("--true-lofo", action="store_true", help="score with a TRUE leave-one-family-out holdout")
    ap.add_argument(
        "--max-holdout-pct",
        type=float,
        default=0.5,
        help="with --true-lofo, a family above this share of pooled rows is never held out (it still trains)",
    )
    ap.add_argument(
        "--module",
        default="src.features.recipes",
        help="module to import the recipes from (default: src.features.recipes)",
    )
    ap.add_argument(
        "--sites",
        type=int,
        default=None,
        help="cap to the first N sites (SMOKE ONLY -- it changes the family structure and so the LOFO "
        "difficulty; never pick values on a subset)",
    )
    ap.add_argument("--splits", type=int, default=5, help="GroupKFold splits for LOSO/LOFO (default 5)")
    ap.add_argument("--min-rows", type=int, default=500, help="per-site inclusion floor (default 500)")
    for axis in _AXES:
        ap.add_argument(f"--{_flag(axis)}", default=None, help=f"ladder for the {axis} axis (default {_AXES[axis][1]})")
    a = ap.parse_args()

    search = [s.strip() for s in a.search.split(",") if s.strip()]
    for s in search:
        if s not in _AXES:
            raise SystemExit(f"--search: {s!r} is not an axis. Axes: {', '.join(_AXES)}")
    fixed = _parse_pairs(a.fix)
    if set(fixed) & set(search):
        raise SystemExit(f"--fix and --search overlap on {sorted(set(fixed) & set(search))}")
    ladders = {axis: _parse_list(getattr(a, _flag(axis)), int if axis == "depth" else float) for axis in _AXES}
    seeds = _parse_list(a.seeds, int)

    mod = importlib.import_module(a.module)
    sites = [str(s) for s in get_site_ids()]
    if a.sites:
        sites = sites[: a.sites]

    tasks = ["reg", "clf"] if a.task == "both" else [a.task]
    families = ["full", "light"] if a.family == "both" else [a.family]
    jobs = [(f, t) for f in families for t in tasks]
    missing = [_RECIPE_SPEC[j][0] for j in jobs if not hasattr(mod, _RECIPE_SPEC[j][0])]
    if missing:
        ap.error(f"{a.module} has no {', '.join(missing)} -- use --family/--task to narrow the run")

    print(f"tuning {len(jobs)} recipe(s) on {len(sites)} sites: " + ", ".join(_RECIPE_SPEC[j][0] for j in jobs))
    for family, task in jobs:
        recipe_attr, target_col = _RECIPE_SPEC[(family, task)]
        tune(
            task, getattr(mod, recipe_attr), target_col, sites, search, fixed, ladders, a.absolute,
            a.ceiling, seeds, a.append, a.true_lofo, a.max_holdout_pct, a.splits, a.min_rows,
        )


if __name__ == "__main__":
    main()
