# What the 2026-07-28 cook/tune migration broke

Companion to the change itself. Written as the work went, so the list is what was actually observed rather than what was predicted.

The migration did three things: ported the new repo's pooling optimizations, rewrote `src/eval/cook.py` around `lean_cook.py` (single-site path gone, early stopping gone, every aggregate recomputed from the LOFO out-of-fold vector), and replaced `src/models/tune.py` with the six-axis staged tuner that resolves the tree count by a prefix scan.

Nothing in `deploy/`, `widget/` or `logs/` broke. Everything below is experiment or notebook code, plus two things about the LOG rather than the code.

---

## 1. Code that no longer imports

**`notebooks/demo_model.py`** — was broken (`ImportError: cannot import name 'FAST_XGB'`), **FIXED**.

`FAST_XGB` now lives in `demo_model.py` itself: "fast enough to demo" is a notebook concern, and a shared constant in `cook.py` invites it being mistaken for a config something ships with. Its `_grouped_models` call turned out to be unaffected — it passes arguments positionally, and only the trailing `seed` parameter was dropped. `run_cv` verified end to end on 10 sites; it returns the new 10-column CLF score set.

**`experiments/isaac/`** — 21 files, none of which are on any production path. **Left broken by decision**, except `_tune.py`.

- `_tune.py` — **DELETED**. It is the tuner this migration replaces, and leaving it beside the new `src/models/tune.py` invites someone running the wrong one against the same `lofo_tune.csv`, where the two disagree about what the tree-count column means.
- `build_model.py` — `compare_many, save_comparison, fit_full, save_model`. Only `fit_full` changed signature (`final_fit` removed).
- `experiments/_experiment{6,6c,7,7c,8,8c}.py` — `from cook import *`, so they pick up whatever is gone.
- `experiments/_experiment{9,9c,10,10c,11,11c,12,12c,13,14,15,16,17,18,19}.py` — `FAST_XGB` and, in `_experiment16`, `compare_fleet`.

They also import `from cook import ...` (bare, not `src.eval.cook`), so they already depended on being run from a particular directory. Repairing them was judged not worth the churn: the score columns they print no longer exist regardless of whether the imports are fixed.

**`experiments/*/*.ipynb`** — stored OUTPUTS reference old score columns (`persist_skill`, `spearman`, `rmse`, `loso_prauc`, `between_r2`). No code change needed; the cells would need re-running to produce a current table.

---

## 2. Pre-existing breakage found and fixed in passing

**`src/selftest.py::test_features`** — was FAILING before this migration, on all 3 sampled sites × both tasks. It asserted a `precip_in_1d` feature family that `recipes.WEATHER_KEEP` had already cut down to `fuel_moisture_1000h` in earlier work. Fixed to key the assertion on `WEATHER_KEEP` rather than a literal column name, so it tracks the recipe instead of contradicting it. All 7 selftests now pass.

Worth recording because a red selftest is exactly what would have hidden a real regression from this migration.

---

## 3. Files whose schema changed

**`src/models/models/lofo_tune.csv`** — the column `best_iteration` is now `n_estimators`, and the row carries `score` / `searched` / `ceiling` / `k_frac` / `true_lofo` instead of `lofo` / `loso` / `gap`. The rename is not cosmetic: the old field was an early-stopping artifact, the new one is a tuned hyperparameter. Old rows describe a regime that no longer exists — delete rather than migrate. (The file was already absent when the migration started.)

**`logs/fulltrain_logs.json`** — historical entries keep their wide, LOSO-based score dict and render fine. New entries carry `"cv_schema": 2`. Entries WITHOUT that field were produced under a different holdout for every aggregate AND a different training regime, so they are not comparable to new ones; nothing in the numbers themselves says so, which is what the field is for.

**Score columns, for reference.** CLF went from 21 columns to 10, REG from 8 to 6:

| | kept | dropped |
|---|---|---|
| CLF | `loso_auc`, `lofo_prauc`, `lofo_auc`, `lofo_prauc_lift`, `lofo_recall_at_beta`, `lofo_fdr_at_beta`, `lofo_brier`, `base`, `lofo_between_rate_r2`, `lofo_macro_auc` | `loso_prauc`, `loso_f1`, `lofo_f1`, `loso_*` imbalance suite, `lofo_f2`, `lofo_mcc`, `lofo_recall_at_far`, `persist_skill`, `brier` (LOSO) |
| REG | `loso_r2`, `lofo_r2`, `lofo_rmse`, `lofo_between_r2`, `lofo_within_r2`, `lofo_macro_r2` | `rmse` (LOSO), `persist_skill`, `spearman`, LOSO-computed `between_r2`/`within_r2`/`macro_r2` |

`logs/render_logs.py` needed no change — `_score_table` renders whatever keys the dict carries.

---

## 4. CLI changes

- **`train.py --false-alarm-rate` is gone**, with `recall_at_far`. It set a module-level `cook._FAR_BUDGET` that was never recorded in the log entry, so runs at different budgets (`recipe_CLF2` / `recipe_CLF_far_20` / `recipe_CLF_far_30` in the current log) are distinguishable only by their names.
- **`train.py` now raises `UntunedRecipe`** when `models/lofo_tune.csv` has no `n_estimators` for the recipe, instead of falling back to `fit_full(final_fit=True)`. There is no early stopping to fall back on. The message names the exact `fulltune.py` command to run.
- **`tune.py`'s CLI is entirely different**: `--search`/`--fix`/`--ceiling`/`--seeds`/`--append` and per-axis ladders, replacing the positional `task` plus `--depths`/`--lrs`/`--regular`/`--fast`. `--family` and `--task` survive with the same meaning.
- **`tune.py`'s CLF ranking metric changed from `auc` to `lofo_prauc`** — the two disagree about "best" at a ~26% base rate, and prauc is the headline train.py reports.

---

## 5. What was measured

Recorded here because these numbers are not reproducible once the pre-optimization implementations are gone, and because the source repo's figures were measured on a different cohort (122 sites, 17 weather variables) and do not transfer by assertion.

**Aggregation** — the vectorized weighted groupby against the per-group implementation it replaces, both run in one process, 3 sites × 3 agg dicts × bucketed and unbucketed, plus a NaN-bearing case:

| | result |
|---|---|
| agreement | equal to float32 epsilon (max relative deviation 5.96e-08), NaN masks identical |
| speedup | **38–49x** on aggregation alone; 272–880x on small-cell sites where per-call overhead dominates, 4–15x on a 2118-cell basin |

**Flow field** — level-set BFS + hop-length table vs the per-cell geodesic walk, on basins spanning 96 to 255,841 cells:

| | result |
|---|---|
| agreement | NaN mask identical; max deviation 2.3e-07 m (sub-micrometre, from pyproj rounding at different longitudes) |
| speedup | **13.7x** on the largest basin |
| flow accumulation | 2.73 s → 0.01 s once cached to disk (this raster is 1057×1741; the source repo's 38 s figure is for a 2880×4320 one) |

**Per-site grids** — `get_grid` from artifact vs computed live, all 83 sites: 1.1 s vs 25.5 s = **22x**. Disk copy verified equal to the live build column-for-column including geometry, and the live fallback verified by hiding the artifact.

**End to end** — `cook._pool(recipe_REG, ...)` over 15 sites, both arms in one process, pools verified identical:

| | result |
|---|---|
| before | 342–363 s |
| after | 52–70 s |
| speedup | **~6x** |

The optimized pool is now dominated by **weather parquet I/O — 21.5 s of 38.2 s (56%)**, reading 19 yearly files per site; aggregation is down to 5.6 s. That is the next bottleneck and this migration does not address it.

**The early-stopping defect, reproduced on this repo's data.** A 12-site `light_REG` sweep at a 300-tree ceiling: the winning config scored `lofo_r2` 0.2255 at its chosen 70 trees against **0.1415 at the ceiling**. Running to the ceiling would have given back ~0.08 R². The source repo measured 0.0324 on a different cohort; the direction and rough magnitude replicate.

The same shakedown also confirmed the relative lam/mcw units matter: `fulltune` settled on `reg_lambda=342` and `min_child_weight=114` against the inherited 5 and 10, and climbed `lofo_r2` 0.2227 → 0.2945 across the six stages. At the inherited values those axes were a 0.09% leaf shrink — inert, exactly as predicted.

**Caveat on all tuning numbers above:** they come from 12-site shakedown runs, which change the family structure and so the LOFO difficulty. They demonstrate that the machinery works and that the defect is real. They are NOT hyperparameter choices, and the artifacts they wrote (`lofo_tune.csv`, `tune_light_REG.csv`) were deleted rather than left for `train.py` to pick up.

---

## 6. Models

The four shipped boosters are now stale relative to the TRAINING REGIME, not just the recipe: they were fitted with early stopping under the old CV. `deploy/models/light_{REG,CLF}.json` were already pending a retrain against current `recipes.py`; that retrain now needs a tuning run first (`fulltune.py --family light`). When it happens, `widget/static/build_bundle.py --group models` must repack the new boosters for the static site.

No model was retrained as part of this migration.
