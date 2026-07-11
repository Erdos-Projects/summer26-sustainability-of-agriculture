# KPIs

Primary metric is **LOFO** (leave-one-basin-family-out) — the honest generalization number. LOSO (leave-one-site-out) is reported too but is optimistic because nested basins leak. All predictions are scored as raw model outputs; the 10 mg/L threshold defines the *target*, not a decision cutoff. Definitions mirror `src/eval/metrics.py` (single source of truth).

## Classification (violation ≥ 10 mg/L)

Headline pair is **`lofo_prauc`** + **`lofo_f1`** — base-rate-aware discrimination and best-case exceedance-decision quality, both with the leaking-basin family held out. LOSO twins are reported but optimistic.

| KPI | Definition | Direction |
|---|---|---|
| `lofo_auc` / `loso_auc` | ROC-AUC on family / site OOF | ↑ (0.5 = chance) |
| `lofo_prauc` / `loso_prauc` | Average precision (baseline ≈ base rate), family / site OOF | ↑ |
| `lofo_f1` / `loso_f1` | Best-F1 over the threshold sweep (max-F1 operating point) | ↑ |
| `brier` | Mean squared error of P(violation); calibration | ↓ (0 best) |
| `macro_auc` | Median per-site AUC (equal site weight) | ↑ |
| `between_rate_r2` | R² of per-site mean P vs per-site violation rate | ↑ |
| `base` | Violation rate (reference for prauc/brier) | — |
| `persist_skill` | Brier skill vs predict-yesterday | ↑; **gauged-site only** — N/A for a virtual site (no own history to persist) |

**Class-imbalance suite** (`_imbalance_suite`, reported `lofo_*` + `loso_*`) — the honest picture when violations are the rare class:

| KPI | Definition | Direction |
|---|---|---|
| `lofo_prauc_lift` / `loso_prauc_lift` | PR-AUC ÷ base rate — × better than a random ranker (imbalance-normalised, threshold-free) | ↑ (1 = chance) |
| `lofo_f2` / `loso_f2` | Best F2 over the sweep (recall weighted 2× precision — false-negative-averse) | ↑ |
| `lofo_mcc` / `loso_mcc` | Best Matthews correlation (all 4 confusion cells; robust to skew) | ↑ (0 = chance) |
| `lofo_recall_at_far` / `loso_recall_at_far` | Recall achievable at a false-alarm rate ≤ `_FAR_BUDGET` (10%) | ↑ |

`_score` also returns `f1_thresh`, the P(violation) cutoff that achieves the best F1 — the operating point to ship in a decision UI. The best-over-sweep entries (`f1`, `f2`, `mcc`) tune their threshold on the eval rows, so read them as mildly optimistic and always beside the threshold-free `prauc` / `prauc_lift`.

## Regression (nitrate concentration, mg/L)

| KPI | Definition | Direction |
|---|---|---|
| `lofo_r2` / `loso_r2` | R² on family / site OOF | ↑ |
| `rmse` | Root mean squared error | ↓ |
| `persist_skill` | 1 − MSE(model)/MSE(predict-yesterday) | ↑ |
| `spearman` | Rank corr of pred vs actual (scale-free) | ↑ |
| `between_r2` / `within_r2` | Cross-site level / daily-anomaly R² | ↑ |
| `macro_r2` | Median per-site R² | ↑ |
