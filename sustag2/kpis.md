# KPIs

Primary metric is **LOFO** (leave-one-basin-family-out) — the honest generalization number. LOSO (leave-one-site-out) is reported too but is optimistic because nested basins leak. CV outputs are scored as raw model probabilities/values; the 10 mg/L threshold defines the *target*, not a scoring cutoff. **Deployment** then layers one decision cutoff on top: a β-derived threshold τ picked post-hoc on the frozen classifier (see *Deployed operating point* below). Scoring definitions mirror `src/eval/cook.py` (`_score`, `_imbalance_suite`, `_persistence_skill`); the β-table lives in `src/models/tune_threshold.py` and the shipped model's `<name>.meta.json`.

## Headline results (final deployed models)

LOFO unless noted; `recipe_CLF2` / `recipe_REG2`, 80 sites · 19 basin families · 160,074 site-days (`logs/fulltrain_logs.json`). These are the numbers in `notebooks/fulldemo.ipynb` and the presentation transcript.

**Classification — violation ≥ 10 mg/L** (base rate 0.263):

| ROC-AUC | PR-AUC | PR-AUC lift | macro-AUC | F1 | F2 | MCC | recall @10% FAR | Brier |
|---|---|---|---|---|---|---|---|---|
| **0.82** | **0.63** | **2.4×** | **0.90** | 0.61 | 0.71 | 0.47 | 0.52 | **0.137** |

LOSO twins (optimistic): AUC 0.84 · PR-AUC 0.68 · lift 2.6× · recall@FAR 0.57.

**Regression — nitrate concentration (mg/L):**

| R² (LOFO) | R² (LOSO) | RMSE | Spearman | within-site R² | between-site R² | macro-R² |
|---|---|---|---|---|---|---|
| **0.33** | 0.38 | **4.36** | 0.65 | 0.42 | 0.21 | 0.24 |

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

`_score` also returns `f1_thresh`, the P(violation) cutoff that achieves the best F1. The best-over-sweep entries (`f1`, `f2`, `mcc`) tune their threshold on the eval rows, so read them as mildly optimistic and always beside the threshold-free `prauc` / `prauc_lift`.

### Deployed operating point (β-table)

What actually ships in the widget is a **recall-emphasis knob β**, not a fixed F1 cutoff. For each β, `src/models/tune_threshold.py` sweeps the honest LOFO OOF, picks τ = argmax F_β, and records the recall/precision there; the table + base rate are written into the model's `meta.json` and read at serve time. The frozen booster is never retrained — this is pure cutoff selection.

| KPI | Definition | Direction |
|---|---|---|
| `tau` | Decision threshold on P(violation): alarm when P ≥ τ. τ = threshold that maximises F_β on the LOFO OOF | — (↓ as β ↑) |
| `recall` | Catch rate at τ(β): TP/(TP+FN) — share of true violation days flagged | ↑ |
| `fdr` | False-discovery rate at τ(β): FP/(TP+FP) = 1 − precision — share of alarms that are false | ↓; **prevalence-dependent**, quoted at `base_rate` |
| `base_rate` | Pooled violation prevalence (the reference at which `fdr` is stated) | — |

β weights recall β²× precision, so higher β ⇒ lower τ ⇒ more alarms (more caught, more false). Deployed β-table (`isaac_CLF2`, base rate 0.263), **default β = 2**:

| β | τ | recall | FDR |
|---|---|---|---|
| 0.5 | 0.56 | 0.50 | 0.34 |
| 1.0 | 0.32 | 0.65 | 0.43 |
| 1.5 | 0.11 | 0.80 | 0.53 |
| **2.0** | **0.06** | **0.86** | **0.59** |
| 2.5 | 0.02 | 0.94 | 0.65 |
| 3.0 | 0.01 | 0.96 | 0.67 |
| 4.0 | 0.01 | 0.99 | 0.70 |

## Regression (nitrate concentration, mg/L)

| KPI | Definition | Direction |
|---|---|---|
| `lofo_r2` / `loso_r2` | R² on family / site OOF | ↑ |
| `rmse` | Root mean squared error | ↓ |
| `persist_skill` | 1 − MSE(model)/MSE(predict-yesterday) | ↑; **gauged-site only** (predict-yesterday needs local history) |
| `spearman` | Rank corr of pred vs actual (scale-free) | ↑ |
| `between_r2` / `within_r2` | Cross-site level / daily-anomaly R² | ↑ |
| `macro_r2` | Median per-site R² | ↑ |
