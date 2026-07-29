"""fulltune.py -- run tune.py's staged workflow end to end and print the finished config.

`python tune.py` searches ONE stage and leaves the other axes inherited. This chains the whole protocol, threading each stage's winner into the next as --fix, and ends with a confirmation run at the settled config to resolve its final tree count.

    python fulltune.py                                  # all four recipes
    python fulltune.py --family light                   # light_REG + light_CLF
    python fulltune.py --family light --task reg        # just light_REG
    python fulltune.py --sites 25 --ceiling 400         # fast shakedown

STAGE ORDER follows axis coupling, not importance -- see tune.py's module docstring:

    1  depth x lr      largest effects, and everything downstream needs a depth to sit at
    2  depth x lam     re-crosses depth on purpose: relative units are only APPROXIMATELY depth-invariant, and this is the pair where fixing one then the other would miss (deep, large-lambda)
    3  mcw             weakly coupled to lam (one blocks splits, the other shrinks surviving leaves) -> coordinate is sound
    4  subsample
    5  colsample
    final              nothing searched, everything fixed: resolves best_k for a combination the coordinate stages may never have scored together

THE CEILING ADAPTS. Stage 1 runs at `--ceiling` (default 1500) and reports what every config actually wanted; stages 2-5 then run at the largest best_k stage 1 produced, since scanning above it is pure prediction cost. The FINAL run returns to the original ceiling, because the settled combination is new and may want more trees than any single-axis winner did. A stage whose winner lands at >=80% of its ceiling is retried once at double, because a binding ceiling is not a result.

One pool is built and shared across every stage (tune.tune(pool=...)); pooling the cohort is the expensive part and repeating it six times would dominate the run.

WHAT IT LEAVES BEHIND. Each stage upserts into models/lofo_tune.csv, so the FINAL stage's row is the one that survives. Nothing reads that file: the settled config reaches a model only when the block the final stage prints is pasted into train.RECIPE_XGB, tree count and all. The per-stage grids accumulate in models/tune_<recipe>.csv (stage 1 overwrites, the rest append).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

import src.eval.cook as cook
import src.models.tune as T
from src.data.access import get_site_ids

_STAGES = [["depth", "lam"], ["mcw"], ["subsample"], ["colsample"]]  # stage 1 is separate: it sets the ceiling
_BIND = 0.8  # k_frac at or above this means the ceiling chose the tree count, not the data


def _winner_axes(df: pd.DataFrame) -> dict:
    """Axis-space values of the top-ranked row, for threading into the next stage's --fix.

    Only axes the row actually carries. A DEFAULT (inherited) winner contributes nothing, which is right: it means the inherited value beat every ladder rung, and leaving it unpinned inherits exactly that.
    """
    best = df.iloc[0]
    return {a: float(best[f"ax_{a}"]) for a in T._AXES if f"ax_{a}" in df.columns and pd.notna(best.get(f"ax_{a}"))}


def _stage(label, task, recipe, target_col, sites, search, fixed, ceiling, seeds, true_lofo, mhp, pool, append, splits, min_rows):
    """Run one stage, retrying once at double the ceiling if the winner comes back ceiling-bound."""
    print(f"\n{'=' * 78}\n{label}  search={search or ['(none)']}  fix={fixed or '{}'}  ceiling={ceiling}\n{'=' * 78}")
    fixed = {k: v for k, v in fixed.items() if k not in search}  # search wins over fix; strip so tune() never sees both
    df = T.tune(task, recipe, target_col, sites, search, fixed, {}, False, ceiling, seeds, append,
                true_lofo, mhp, splits, min_rows, pool=pool)
    if float(df.iloc[0].get("k_frac", 0)) >= _BIND:
        print(f"\n  [retry] winner used {int(df.iloc[0]['best_k'])}/{ceiling} trees -- ceiling-bound, "
              f"redoing at {ceiling * 2}")
        df = T.tune(task, recipe, target_col, sites, search, fixed, {}, False, ceiling * 2, seeds, True,
                    true_lofo, mhp, splits, min_rows, pool=pool)
    return df


def full(task, recipe, target_col, sites, ceiling0, seeds, true_lofo, mhp, splits=5, min_rows=500):
    name = getattr(recipe, "__name__", str(recipe))
    pool = cook._pool(recipe, sites, target_col, min_rows=min_rows, progress_label=name)

    t0 = time.time()
    df = _stage("STAGE 1/6  capacity x learning rate", task, recipe, target_col, sites, ["depth", "lr"], {},
                ceiling0, seeds, true_lofo, mhp, pool, False, splits, min_rows)
    fixed = _winner_axes(df)

    # every stage-1 config reported the tree count it wanted; nothing downstream needs to look above
    # the largest of them
    ceiling = int(np.ceil(df["best_k"].max() / T._COARSE) * T._COARSE)
    print(f"\n  >> stage 1 best_k ranged {int(df.best_k.min())}-{int(df.best_k.max())}; "
          f"stages 2-5 will scan to {ceiling} (was {ceiling0})")

    for i, search in enumerate(_STAGES, start=2):
        df = _stage(f"STAGE {i}/6", task, recipe, target_col, sites, search, fixed, ceiling, seeds,
                    true_lofo, mhp, pool, True, splits, min_rows)
        fixed.update(_winner_axes(df))

    # the settled combination may never have been scored as a unit by the coordinate stages --
    # resolve its own best_k, at the ORIGINAL ceiling since it can want more trees than any
    # single-axis winner did
    df = _stage("STAGE 6/6  confirmation at the settled config", task, recipe, target_col, sites, [], fixed,
                ceiling0, seeds, true_lofo, mhp, pool, True, splits, min_rows)

    print(f"\n{'=' * 78}\nFULLTUNE {name} done in {(time.time() - t0) / 60:.0f} min   settled axes: {fixed}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["reg", "clf", "both"], default="both")
    ap.add_argument("--family", choices=["full", "light", "both"], default="both")
    ap.add_argument("--ceiling", type=int, default=1500,
                    help="stage-1 and confirmation ceiling; stages 2-5 use stage 1's max best_k")
    ap.add_argument("--sites", type=int, default=None, help="cap to the first N sites (shakedown only)")
    ap.add_argument("--seeds", default="42", help="comma-separated random_state values; >1 ranks on the seed mean")
    ap.add_argument("--true-lofo", action="store_true")
    ap.add_argument("--max-holdout-pct", type=float, default=0.5)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--module", default="src.features.recipes")
    a = ap.parse_args()

    mod = importlib.import_module(a.module)
    sites = [str(s) for s in get_site_ids()]
    if a.sites:
        sites = sites[: a.sites]
    seeds = tuple(int(x) for x in a.seeds.split(","))

    tasks = ["reg", "clf"] if a.task == "both" else [a.task]
    families = ["full", "light"] if a.family == "both" else [a.family]
    jobs = [(f, t) for f in families for t in tasks]
    missing = [T._RECIPE_SPEC[j][0] for j in jobs if not hasattr(mod, T._RECIPE_SPEC[j][0])]
    if missing:
        ap.error(f"{a.module} has no {', '.join(missing)} -- use --family/--task to narrow the run")

    print(f"fulltune {len(jobs)} recipe(s) on {len(sites)} sites: " + ", ".join(T._RECIPE_SPEC[j][0] for j in jobs))
    for family, task in jobs:
        recipe_attr, target_col = T._RECIPE_SPEC[(family, task)]
        full(task, getattr(mod, recipe_attr), target_col, sites, a.ceiling, seeds,
             a.true_lofo, a.max_holdout_pct, a.splits, a.min_rows)


if __name__ == "__main__":
    main()
