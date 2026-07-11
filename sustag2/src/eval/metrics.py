"""Public metric surface for the nitrate models (mirrors kpis.md).

The full evaluation harness is cook.py (ported whole from isaac/cook.py). This module re-exports
the scoring primitives so callers can `from src.eval.metrics import _score, _cross_metrics` without
depending on the whole harness; a future refactor may hoist the implementations here.

  _score          -- pooled AUC / PR-AUC / best-F1 / Brier / base (clf) or RMSE / MAE / R2 (reg)
  _best_f1        -- max-F1 operating point over the threshold sweep, with its P(violation) cutoff
  _imbalance_suite -- class-imbalance suite: prauc_lift, best-F2, best-MCC, recall @ fixed FAR
  _cross_metrics  -- LOSO/LOFO decomposition (auc, prauc, f1 + the imbalance suite for both) + ...
  _per_site_score -- one-site AUC (clf) or R2 (reg)
  _persistence_pred / _persistence_skill -- "predict yesterday" baseline + skill score
  _spearman       -- scale-free rank correlation of pred vs actual
"""

from src.eval.cook import (  # noqa: F401
    _cross_metrics,
    _imbalance_suite,
    _per_site_score,
    _persistence_pred,
    _persistence_skill,
    _score,
    _spearman,
)
