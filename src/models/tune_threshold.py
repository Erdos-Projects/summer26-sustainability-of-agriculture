"""Post-hoc class-imbalance threshold tuning for an ALREADY-TRAINED violation classifier.

The deployed booster is frozen -- it emits P(violation) per day and is never retrained. "Tuning"
here just picks the decision cutoff tau via an F-beta operating point: beta is the recall/precision
emphasis (beta=2 => recall weighted 4x precision => catch more violations, tolerate more false
alarms). For a grid of beta we find tau maximising F_beta on the honest LOFO out-of-fold
predictions, and record the operating point there -- recall (catch rate) and FDR (false-discovery
rate = 1 - precision = the share of our alarms that are false), the two numbers that measure real
usefulness. The whole table is patched into the model's <path>.meta.json; the booster file is left
byte-for-byte unchanged. The widget reads the table and lets the user dial beta live.

FDR is prevalence-dependent, so we also store the pooled base_rate -- the widget reports FDR as
"expected at ~base-rate prevalence".

NORMALLY YOU DO NOT NEED THIS. src/models/train.py now derives the same table from the LOFO
out-of-fold vector its cross-validation already produced, and writes it into both the run log and
the model sidecar -- so a freshly trained classifier ships with its thresholds already attached.
This script exists for the case it was built for: re-tuning an ALREADY-DEPLOYED, frozen booster to
a different operating point without retraining. That costs a full grouped CV here (to rebuild the
OOF that training had and discarded), which is precisely why the training path stopped relying on
it.

Usage
-----
    python -m src.models.tune_threshold isaac_CLF2                    # recipe_CLF, patch deploy meta
    python -m src.models.tune_threshold light_CLF --recipe light_CLF  # the static widget's pair
    python -m src.models.tune_threshold isaac_CLF2 --recipe recipe_CLF --n-splits 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]  # repo root/
sys.path.insert(0, str(_ROOT))

from src.eval.cook import _pool, _features, _target, basin_groups, _grouped_oof, BETA_GRID, beta_operating_points
from src.features.recipes import recipe_CLF, light_CLF
from src.data.access import get_site_ids
from src.models.train import xgb_for

# Re-exported for callers that used to import these from here. The definitions live in cook.py now,
# beside the rest of the scoring, because src/models/train.py derives the same table from the CV's
# own LOFO out-of-fold vector -- and two copies of "the operating point at beta" would eventually
# disagree about which one a shipped model was tuned under.
BETA = list(BETA_GRID)
_beta_operating_points = beta_operating_points

_RECIPES = {"recipe_CLF": recipe_CLF, "light_CLF": light_CLF}


def build_beta_table(recipe, sites=None, n_splits: int = 5, min_rows: int = 500, xgb: dict = None):
    """Rebuild the honest LOFO out-of-fold predictions for `recipe` (same pooling + grouping the CV
    used) and return (beta_table, base_rate). xgb defaults to the recipe's own effective config so
    the OOF probability scale matches the deployed model."""
    xgb = xgb_for(recipe, "clf") if xgb is None else xgb
    target = "violation"
    pool = _pool(recipe, sites or get_site_ids(), target, min_rows=min_rows, progress_label=getattr(recipe, "__name__", "recipe"))
    feat = _features(pool, target)
    X, y = pool[feat], _target(pool, target, "clf")
    groups = basin_groups(pool["site"])  # LOFO: hold out whole basin families
    oof = _grouped_oof(X, y, groups, "clf", n_splits, **xgb)
    return _beta_operating_points(np.asarray(y), oof), float(np.asarray(y, dtype=float).mean())


def _meta_paths(model: str) -> list[Path]:
    """The <path>.meta.json sidecars to patch for a model stem: the deploy copy (what the widget
    loads) and, if present, the src/models/models training copy."""
    candidates = [
        _ROOT / "deploy" / "models" / f"{model}.json.meta.json",
        Path(__file__).resolve().parent / "models" / f"{model}.json.meta.json",
    ]
    return [p for p in candidates if p.parent.exists()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Model stem to patch, e.g. isaac_CLF2 (patches <stem>.json.meta.json).")
    p.add_argument("--recipe", default="recipe_CLF", choices=list(_RECIPES), help="Recipe the model was trained on.")
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold folds for the LOFO OOF (match training).")
    p.add_argument("--min-rows", type=int, default=500, help="Per-site inclusion floor (match training).")
    args = p.parse_args()

    print(f"Rebuilding honest LOFO OOF for {args.recipe} (this runs a {args.n_splits}-fold grouped CV)...")
    beta_table, base_rate = build_beta_table(_RECIPES[args.recipe], n_splits=args.n_splits, min_rows=args.min_rows)

    print(f"\nbase rate (pooled violation prevalence): {base_rate:.3f}\n")
    print(pd.DataFrame(beta_table).round(4).to_string(index=False))

    metas = _meta_paths(args.model)
    if not metas:
        raise FileNotFoundError(f"No {args.model}.json.meta.json under deploy/models or src/models/models.")
    for path in metas:
        meta = json.loads(path.read_text()) if path.exists() else {}
        meta["beta_table"] = beta_table
        meta["base_rate"] = base_rate
        path.write_text(json.dumps(meta, indent=2))
        print(f"\npatched {path.relative_to(_ROOT)} (booster file untouched)")


if __name__ == "__main__":
    main()
