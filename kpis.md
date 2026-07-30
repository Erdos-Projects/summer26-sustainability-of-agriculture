# KPIs

Primary KPI: **can we flag a nitrate-violation day (≥ 10 mg/L) at a location with no sensor?** This is the classification (CLF) task.

Secondary KPI: Same question but with a regression target, **can we predict nitrate concentration (mg/L) timeseries at unseen sites?** This is the regression (REG) task.

Here are our best models, with metrics explained below. The `light` model in each case is a version of the model trained specifically for deployment in our static-site widget. Surprisingly, it does nearly as well in both cases.

**Classification — violation ≥ 10 mg/L** (base rate 0.258):

| Recipe | Model | Date | PR-AUC (AP) | ROC-AUC | Lift vs base | Recall @ β=2 | FDR @ β=2 | Brier | between-rate R² | LOSO AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| `recipe_CLF` | recipe_0729_CLF | 2026-07-29 | **0.6954** | 0.8605 | 2.70× | 0.888 | 0.545 | 0.1288 | 0.4339 | 0.8785 |
| `light_CLF` | light6_0729_CLF | 2026-07-29 | **0.6893** | 0.8581 | 2.67× | 0.880 | 0.533 | 0.1285 | 0.4236 | 0.8749 |

**Regression — nitrate concentration (mg/L):**

| Recipe | Model | Date | R² | RMSE | between-site R² | within-site R² | macro-R² | LOSO R² |
|---|---|---|---|---|---|---|---|---|
| `recipe_REG` | recipe_0729_REG | 2026-07-29 | **0.3799** | **4.386** | 0.2581 | 0.4034 | 0.3278 | 0.4479 |
| `light_REG` | light6_0729_REG | 2026-07-29 | **0.3713** | 4.416 | 0.2042 | 0.4129 | 0.3056 | 0.4517 |


# Cross Validation

We evaluate our methods using various scores (see below) using a few different cross validation metrics:
- **LOFO -- Leave One FAMILY Out:** primary CV technique, splits train/test based on hydrological connection. A hydrologically connected family of sites is always kept together across the split. Most robust to data leakage. Is a conservative metric. All metrics listed are this unless otherwise specified with a prefix. We actually use GroupKFold with $k = 5$ for this, 85/15.
- **LOSO -- Leave One Site Out:** secondary CV technique, performs train/test split without reference to hydrological connection. It still never splits data from a site across train/test; sites are highly autocorrelated so this essentially trivializes the problem -- you train a model to memorize each site and then learn to identify which site it is looking at from the static geographic features. Also GroupKFold with $k = 5$, attempt to meet, 85/15.
- **True_LOFO:** Same thing as LOFO but we actually hold one whole family out for testing rather than doing GroupKFold.
- **LODO_d -- Leave One Distance family Out:** A mix between True_LOFO and True_LOSO, hold out one site for testing, train on everything which is either not connected to the site OR is at least $d$ meteres away by flow distance.

# Metrics for CLF (violation ≥ 10 mg/L)

Headline pair is **`lofo_prauc_lift`** + **`lofo_auc`**. The former is **`prauc`** below divided by the base-violation rate — how much the CLF model outperforms coin-flipping. Formulas and descriptions are given below. Note these definitions are independent of cv technique; for most we only track `lofo` values, but we track `loso_auc` as a comparison point.

Take the pooled table of (truth y, prediction p, site g) over all 158,215 rows and 81 sites, in 20 basin families.

`prauc` — "of the rows the model flags hardest, how many are real?"
Average precision: sweep every threshold and integrate precision against the recall it buys, so a rare positive class is never rewarded for the true negatives it gets for free.
$$\mathrm{AP} = \sum_k \big(R_k - R_{k-1}\big), P_k, \qquad P_k = \mathrm{Prec}(\tau_k),\ R_k = \mathrm{TPR}(\tau_k)$$

`prauc_lift` — "how much better than guessing?"
Average precision divided by the base rate, which is what a random ranker scores. Imbalance-normalised, so it reads the same way at any prevalence — but bounded above by $1/\pi$ (3.88 here), so it is not comparable across cohorts whose base rates differ.
$$\text{lift} = \mathrm{AP}/\pi, \qquad \pi = \tfrac{1}{N}\sum_i y_i$$

`auc` — "does a random violation outrank a random non-violation?"
ROC-AUC: the probability that a randomly drawn positive row scores above a randomly drawn negative one, integrated over the entire ranking including thresholds nobody would deploy.
$$\mathrm{AUC} = \Pr\big(p_i > p_j ,\big|, y_i = 1,\ y_j = 0\big) = \int_0^1 \mathrm{TPR}, d(\mathrm{FPR})$$

`br2 (between_rate_r2)` — "which basins are the bad ones?"
Collapse each site to its observed violation rate and its mean predicted probability, then take R² over those 81 points, unweighted by row count.
$$R^2_{\text{between-rate}} = 1 - \frac{\sum_{s\in\mathcal S}\big(\bar y_s - \bar p_s\big)^2}{\sum_{s\in\mathcal S}\big(\bar y_s - \bar{\bar y}\big)^2}, \qquad \bar y_s = \tfrac{1}{n_s}\!\!\sum_{i:,s(i)=s}\!\! y_i$$

`macro_auc` — "how well does it time violations at a typical site?"
ROC-AUC computed separately on each site's own rows, then the median across sites — one site one vote, so unlike `auc` it is not dominated by long-record sites. Sites where AUC is undefined (no violations, or no non-violations) drop out of the median rather than counting as zero.
$$\text{macro-AUC} = \operatorname*{median}_{s} \mathrm{AUC}_s$$

`f2 (recall_at_f2)` — "at the shipped alarm setting, what share of violations do we catch?"
Pick the threshold that maximizes $F_\beta$ at $\beta = 2$ (recall weighted $4\times$ precision, the deployed operating point) and read off the true-positive rate there.
$$\tau_2 = \operatorname*{arg,max}_\tau \frac{5\,\mathrm{Prec}(\tau)\,\mathrm{TPR}(\tau)}{4\,\mathrm{Prec}(\tau) + \mathrm{TPR}(\tau)}\, \qquad \text{recall@}F_2 = \mathrm{TPR}(\tau_2)$$

`fdr_at_f2` — "at that same setting, what share of the alarms are false?"
The complement of precision at the identical $\tau_2$, which is why it must always be quoted beside f2 — recall bought by lowering the threshold shows up here as cost.
$$\mathrm{FDR@}F_2 = 1 - \mathrm{Prec}(\tau_2) = \frac{\mathrm{FP}(\tau_2)}{\mathrm{TP}(\tau_2) + \mathrm{FP}(\tau_2)}$$

`brier` — "when it says 30%, does it happen 30% of the time?"
Mean squared error of the predicted probability against the 0/1 outcome — a calibration score, not a ranking one, so unlike prauc and auc it moves when the probabilities are shifted even if the ordering is untouched. Lower is better and $0$ is perfect, but the number is unreadable alone: the reference point is the climatology forecast $p_i \equiv \pi$, which scores $\pi(1-\pi) = 0.258 \times 0.742 = 0.1914$ on this cohort. That gives the skill score $\mathrm{BSS} = 1 - \mathrm{Brier}/\pi(1-\pi)$, so `recipe_CLF`'s 0.1288 is a BSS of 0.33 (`light_CLF`'s 0.1285 likewise).

$$\mathrm{Brier} = \frac{1}{N}\sum_i \big(p_i - y_i\big)^2$$

`base` — "how often does a violation happen at all?"
The pooled violation rate. Not a score in itself: it is the reference every other CLF number is read against — the floor `prauc` is divided by, the $\pi(1-\pi)$ in the Brier skill score, and the prevalence at which `fdr` is quoted.
$$\pi = \tfrac{1}{N}\sum_i y_i = 0.258$$

`loso_auc` — "how much does hydrological leakage flatter us?"
The same ROC-AUC on the LOSO out-of-fold vector instead of the LOFO one. Always the higher of the two, because nested basins put near-copies of a held-out site into training; the gap is what the blocking buys. For the deployed classifier: 0.8749 LOSO against 0.8581 LOFO.

# Metrics for REG (nitrate mg/L)
These are more self explanatory. The primary one is `lofo_r2` which captures the same thing as `rmse`. The triple `between_r2`, `within_r2` and `macro_r2` are for measuring inter-site dynamics in various ways.

`lofor2` — "how much of the total nitrate variance is explained?"
Ordinary R² over all 158,215 pooled rows against the global mean, so it is row-weighted and long-record sites dominate it.
$$R^2 = 1 - \frac{\sum_i (y_i - p_i)^2}{\sum_i (y_i - \bar y)^2}$$

`loso_r2` — "how much does hydrological leakage flatter us?"
The same R² on the LOSO out-of-fold vector instead of the LOFO one, and for the same reason always the higher of the two. For the deployed regressor: 0.452 LOSO against 0.371 LOFO.

`rmse` — "how far off is a typical prediction, in mg/L?"
The same error as lofor2 but left in the target's units instead of normalized by variance, which makes it comparable across cohorts where $R^2$ is not.
$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (y_i - p_i)^2}$$

`between_r2` — "which basins are bad?"
Collapse every site to two numbers, its mean truth and its mean prediction, then take R² over those 81 points, unweighted by row count:
$$R^2_{\text{between}} = 1 - \frac{\sum_s (\bar y_s - \bar p_s)^2}{\sum_s (\bar y_s - \bar{\bar y})^2}$$

`within_r2` — "when does it spike?"
Broadcast each site's mean back to its rows and subtract it from both truth and prediction, then take R² on what's left:
$$R^2_{\text{within}} = 1 - \frac{\sum_i \big[(y_i - \bar y_{s(i)}) - (p_i - \bar p_{s(i)})\big]^2}{\sum_i (y_i - \bar y_{s(i)})^2}$$

`macro_r2` — "how does the typical site do?"
Compute R² separately on each site's own rows, then take the median across sites. Sites where R² is undefined (a single row, or zero variance in the truth) drop out of the median rather than counting as zero:
$$\text{macro-}R^2 = \operatorname*{median}_{s} R^2_s$$

## Some disclaimers
- **`recall`/`FDR` at β=2 are mildly optimistic.** τ is chosen on the same out-of-fold rows those rates are then measured on, so read them as "best achievable at this operating point", not a forward estimate. PR-AUC, ROC-AUC and the R² columns are threshold-free.
- **`between_r2` carries a wide error bar** — an R² over ~81 site means, sampling SD ≈ 0.08. Read 0.2581 and 0.2042 as indistinguishable.

## Deployed operating point (β-table)

What actually ships in the widget is a **recall-emphasis knob β**, not a fixed cutoff. For each β, `src/models/tune_threshold.py` sweeps the honest LOFO OOF, picks τ = argmax F_β, and records the recall/precision there; the table + base rate are written into the model's `meta.json` and read at serve time. The frozen booster is never retrained — this is pure cutoff selection.

| KPI | Definition | Direction |
|---|---|---|
| `tau` | Decision threshold on P(violation): alarm when P ≥ τ. τ = threshold that maximises F_β on the LOFO OOF | — (↓ as β ↑) |
| `recall` | Catch rate at τ(β): TP/(TP+FN) — share of true violation days flagged | ↑ |
| `fdr` | False-discovery rate at τ(β): FP/(TP+FP) = 1 − precision — share of alarms that are false | ↓; **prevalence-dependent**, quoted at `base_rate` |
| `base_rate` | Pooled violation prevalence (the reference at which `fdr` is stated) | — |

β weights recall β²× precision, so higher β ⇒ lower τ ⇒ more alarms (more caught, more false). β-table for the best classifier (`recipe_0729_CLF`, base rate 0.258), **default β = 2**:

| β | τ | recall | FDR |
|---|---|---|---|
| 0.5 | 0.562 | 0.491 | 0.263 |
| 1.0 | 0.261 | 0.698 | 0.386 |
| 1.5 | 0.096 | 0.840 | 0.498 |
| **2.0** | **0.056** | **0.888** | **0.545** |
| 2.5 | 0.030 | 0.933 | 0.598 |
| 3.0 | 0.025 | 0.945 | 0.613 |
| 3.5 | 0.017 | 0.965 | 0.642 |
| 4.0 | 0.012 | 0.976 | 0.662 |

The widget ships the LIGHT classifier, whose thresholds differ: at β = 2 it alarms at τ = 0.095 for recall 0.880 / FDR 0.533. τ is not comparable between the two models — it is a cutoff on each one's own probability scale, so only the recall/FDR pair is.

## Regression (nitrate concentration, mg/L)

| KPI | Definition | Direction |
|---|---|---|
| `lofo_r2` / `loso_r2` | R² on family / site OOF (LOFO honest, LOSO optimistic) | ↑ |
| `rmse` | Root mean squared error (mg/L) | ↓ |
| `between_r2` | R² of predicted vs actual per-site means — ranks site levels | ↑ |
| `within_r2` | R² after removing each site's mean — tracks daily movement within a site | ↑ |
| `macro_r2` | Median per-site R² (equal site weight) | ↑ |
