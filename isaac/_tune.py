"""_tune.py -- grid-search XGBoost hyperparameters against the HONEST cross-site metric.

Sweeps a small grid via cook_many and ranks configs by the leave-one-family-out score
(lofo_r2 for regression, lofo_auc for classification) -- the metric that reflects transfer to
an UNSEEN basin, not memorization. Prints the ranked table and the winning config as a
copy-pasteable dict (drop it into build_model.REAL_XGB, or fit_full(**winner)).

Tuning uses early stopping (n_estimators adapts per config) and skips the slow permutation
importance. lofo needs >=2 basin families among the chosen sites, so don't tune on too few.

Run:
    python isaac/_tune.py                 # regressor, 10 sites (--med)
    python isaac/_tune.py --clf --full    # classifier, all filtered sites
"""

import argparse
import itertools
import sys

sys.path.insert(0, "../")

from cook import cook_many
from recipes2 import recipe_REG, recipe_CLF
from data.features import daily_nitrate
from data import get_site_ids

# Fixed (non-swept) settings; the grid overrides these per run.
BASE = dict(
    n_estimators=4000,  # high ceiling; early stopping picks the real count per config
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.0,
    early_stopping_rounds=50,
    random_state=42,
)

# The search grid. 4 x 3 = 12 configs; uncomment the extra knobs to widen it (product grows
# fast -- adding both lines below makes it 12 x 2 x 2 = 48).
GRID = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.05],
    # "min_child_weight": [5, 10],
    # "reg_lambda": [1.0, 5.0],
}

MODES = {"test": 5, "med": 10, "full": None}  # how many filtered sites to tune on


def _combos(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def build_sites():
    """Filtered sites (>=1500 daily nitrate obs)."""
    return [s for s in get_site_ids() if daily_nitrate(s).dropna().shape[0] >= 1500]


def main():
    ap = argparse.ArgumentParser(description="Grid-search XGBoost on the lofo metric.")
    ap.add_argument("--clf", action="store_true", help="tune the classifier (default: regressor)")
    g = ap.add_mutually_exclusive_group()
    for m in MODES:
        g.add_argument(f"--{m}", dest="mode", action="store_const", const=m)
    ap.set_defaults(mode="med")
    args = ap.parse_args()

    recipe = recipe_CLF if args.clf else recipe_REG
    target_col = "violation" if args.clf else "nitrate_con"
    task = "clf" if args.clf else "reg"
    lofo_key = "lofo_auc" if args.clf else "lofo_r2"
    loso_key = "loso_auc" if args.clf else "loso_r2"

    sites = build_sites()
    n = MODES[args.mode]
    sites = sites if n is None else sites[:n]

    combos = list(_combos(GRID))
    print(f"tuning {task} on {len(sites)} sites: {len(combos)} configs, ranked by {lofo_key}\n")

    results = []
    for i, combo in enumerate(combos, 1):
        cfg = {**BASE, **combo}
        r = cook_many(recipe, sites, target_col=target_col, task=task, extra_importance_test=False, progress=False, **cfg)
        lofo, loso = r[lofo_key], r[loso_key]
        results.append((lofo, loso, combo))
        print(f"  [{i:>2}/{len(combos)}] {lofo_key}={lofo:+.4f}  {loso_key}={loso:+.4f}  {combo}")

    # best lofo first; NaN (e.g. <2 basin families) ranks last
    results.sort(key=lambda t: (t[0] != t[0], -(t[0] if t[0] == t[0] else 0)))
    best_lofo, best_loso, best = results[0]

    print(f"\n=== ranked by {lofo_key} ===")
    for lofo, loso, combo in results:
        print(f"  {lofo_key}={lofo:+.4f}  {loso_key}={loso:+.4f}  {combo}")

    if best_lofo != best_lofo:  # NaN
        print(f"\n[!] every config returned NaN {lofo_key} -- likely <2 basin families in these "
              f"{len(sites)} sites. Re-run with --full (or more sites) so lofo is defined.")
        return

    winner = {**BASE, **best}
    print(f"\n=== winner: {lofo_key}={best_lofo:+.4f} ({loso_key}={best_loso:+.4f}) ===")
    print("REAL_XGB = dict(")
    for k, v in winner.items():
        print(f"    {k}={v!r},")
    print(")")


if __name__ == "__main__":
    main()
