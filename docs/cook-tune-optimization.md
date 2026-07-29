# The 2026-07-28 cook/tune speed pass

Companion to the working change in `src/eval/cook.py`, `src/models/train.py` and `src/models/tune.py`. Four separate optimizations, measured individually because they are worth very different amounts and one of them is worth much less than its complexity suggests.

Everything below was measured in the `sustag` env (xgboost 3.3.0) on `light_REG` / `light_CLF` over the first 5-8 sites, both arms in one process. Site subsets change the family structure and so the LOFO difficulty — these are PARITY and COST measurements, not scores, and no number here is a hyperparameter choice.

---

## 1. float32 features in `_pool` — free, exact

`_pool` now casts every float64 column except the target to float32. XGBoost's `DMatrix` casts to float32 internally regardless, so this is exact rather than approximate, and the target keeps float64 so every metric is still computed there.

| | result |
|---|---|
| agreement | **13,270/13,270 predictions bit-identical** (max diff 0.0), both tasks, fit on the cast frame vs the uncast one |
| pool memory | 5.7 MB → 3.1 MB (REG), 5.6 MB → 3.0 MB (CLF) — **~46%** |

## 2. Pool once per training run — the big one

`cook_many` and `fit_full` take a prebuilt `pool=`, and `train.build` pools once and passes the same frame to both. A training run used to build the identical frame twice.

Pooling costs ~3.6 s/site on this cohort, so on the full 83 sites that is ~5 minutes given back per `train.py` invocation — by a wide margin the largest saving in this change. `compare_many` refuses a prebuilt pool for more than one recipe, since `pool` rides through `**kw` and every recipe after the first would otherwise be silently scored on the FIRST recipe's columns.

| | result |
|---|---|
| `cook_many(pool=)` vs re-pooling | 10 REG / 14 CLF numeric score columns, **0 differ** |
| `fit_full(pool=)` vs re-pooling | feature lists equal, **11,181/11,181 predictions bit-identical** |
| `compare_many(pool=, 2 recipes)` | raises, as intended |

## 3. LOSO off per config — the big one for sweeps

The site-grouped pass is a full second CV whose only product is the leakage-gap column, and nothing ranks on it. It is now off by default in `tune.py` (`--loso` restores it) and optional in `cook_many` (`loso=`), with one site pass run for the winning config instead, at the winner's own tree count rather than the ceiling.

That halves the fits in a sweep — fitting is ~95% of a config's cost (see 4) — for one extra fit set per sweep. With `loso=False` the column set is unchanged and `loso_r2` / `loso_auc` is NaN, verified, so no consumer has to care.

**One defect found and fixed here.** The winner's gap was being written to `f"loso_{_CURVE_METRIC[task]}"`, which is `loso_r2` for REG but **`loso_prauc` for CLF** — a column no other row carries and nothing reads, while the `loso_auc` every consumer does read stayed empty. CLF ranks on prauc but its surviving LOSO column is auc. Now keyed on `_LOSO_METRIC` and the gap is printed against its own twin (`loso_auc` vs `lofo_auc`), since a prauc-vs-auc subtraction is not a leakage gap. Confirmed on an 8-site `light_CLF` shakedown: `loso_auc` populated on the winner row only, no stray column, `lofo_tune.csv` upserted correctly. Both shakedown CSVs were deleted rather than left for `train.py` to read.

## 4. Additive prefix margins — correct, and much less valuable than expected

`_prefix_oofs` replaces the per-k walk: boosting margins are additive and `iteration_range=(a, b)` costs O(b − a), so the margin at each k is the running sum of slice predictions rather than a fresh walk from tree 0 — O(max k) per fold instead of O(sum k). At ceiling 1500 with `_COARSE=50` the arithmetic says 23,250 tree-evaluations per fold become 1,500, i.e. ~15x.

**Measured, it is 2.4-2.5x, on a step that is ~5% of the cost.** Prediction is not purely proportional to tree count — per-call overhead and the fold slice dominate at this pool size — and the fits the scan reads from are far more expensive than the scan itself:

| | REG | CLF |
|---|---|---|
| 4 fold fits at 1500 trees | 28 s | 27 s |
| scan over 30 prefixes, old | 1.6 s | 1.6 s |
| scan over 30 prefixes, new | 0.6 s | 0.7 s |
| speedup on the scan | 2.5x | 2.4x |
| share of the config's total cost | ~5% | ~5% |

It is also the one piece here that is **not exact**. Summing float32 slice margins accumulates rounding a single walk does not:

| | REG | CLF |
|---|---|---|
| max abs prediction drift | 8.39e-05 (at k=1450) | 1.65e-06 (at k=1500) |
| max abs metric drift | 1.07e-06 (r2) | 1.58e-06 (prauc) |
| argmax k over the coarse grid | identical | identical |

The drift is ~4 orders of magnitude below any curve difference that decides a `best_k`, so it is harmless. But the trade is real and worth stating plainly: **~3% off a sweep, in exchange for approximate prefixes and two traps that produce plausible wrong numbers rather than errors** — `base_score` is stored in PROBABILITY space for `binary:logistic` and must be logit-ed before it can be subtracted from a margin, and `iteration_range=(0, 0)` means ALL trees rather than none. Both are handled and documented in `_base_margin` / `_prefix_oofs`. Reverting to the per-k walk would cost ~1 s per config and restore exactness; keeping it is defensible, but not on the strength of the O(sum k) arithmetic alone.

---

## Verification status

`src/selftest.py`: **7 passed, 0 failed**. End-to-end `tune.py` shakedown (8 sites, `light_CLF`, 3 configs, ceiling 300) completed in 21 s including the winner LOSO pass, wrote both CSVs correctly, and printed the expected ceiling-bound and ladder-top edge reports.

Not verified end to end: `train.py build()`, which needs a real tuning run to supply `n_estimators` first. Its pool threading is covered by the `cook_many` and `fit_full` parity above, so what remains untested there is only the wiring.

No model was retrained, and `models/lofo_tune.csv` is absent again — the four shipped boosters remain stale in the sense the migration doc records, and `fulltune.py --family light` is still the next step before any retrain.
