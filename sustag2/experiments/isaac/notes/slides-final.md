# 5-Minute Presentation — Slide Deck (final, numbers-filled)

*Slide-by-slide companion to [transcript-final.md](transcript-final.md). 5 content slides + title +
appendix, ~5:00 total. On-screen bullets are condensed; full wording lives in the transcript. Figures
are real files in [`images/`](images/). `---` marks slide breaks (Marp/reveal-compatible). All model
numbers are honest leave-one-basin-family-out CV from the deployed `recipe_CLF2` / `recipe_REG2`.*

---

## Slide 1 — Title

# Predicting Iowa's Nitrate Risk
### A virtual sensor from weather + land-use data

*Team names · date*

**Visual:** full-bleed `images/erin_surplus_2005.png` (Iowa N-surplus hotspots — the agricultural-intensity story at a glance).

**Speaker notes (~15s):** *TL;DR up front* — "We can flag nitrate-violation risk at locations with **no sensor**, catching ~86% of true violation days and ranking risk ~2.4× better than chance on watersheds the model has never seen. That turns a shrinking sensor network into near-statewide coverage. Here's how."

---

## Slide 2 — The Problem & The Data

- Iowa = heart of U.S. row-crop agriculture; runs on **nitrogen fertilizer**
- Excess N leaches into water as **nitrate** → linked to elevated **cancer risk**; **10 mg/L** federal limit
- Monitoring is **sparse, expensive, and being cut** — most waterways ungauged
- **Data:** 85 curated sensors (from 162), 2008–2025, **~160k site-days**
- Predictors = **what's upstream**: drainage basin (**verified 3 ways**) × satellite crops · N-surplus · daily weather

**Visual:** `images/basin_over_crops.png` — a real basin tiled by weather-grid cells, each colored by dominant crop, draining to the starred sensor (the clearest "what's upstream" visual).

**Speaker notes (~70s):** Iowa's agriculture runs on nitrogen fertilizer; the excess leaches into water as nitrate, linked to cancer risk, and Iowa's water often exceeds the 10 mg/L federal limit — Iowa is one of only a couple of states where cancer rates are still climbing. But sensors are expensive and coverage is being cut, so most waterways are blind. Our unit is a sensor site-day. We built from 85 curated sensors — about 160,000 site-days — and for each computed its drainage basin, verified three independent ways, then layered satellite crops, soil-nitrogen surplus, and daily weather onto a common grid. The insight: only what's *upstream* matters.

---

## Slide 3 — Modeling & Honest Validation

- **XGBoost** for both tasks — response is nonlinear (rain × fertilizer), data is messy
- Beat the **classical baselines** (persistence, linear, exponential smoothing)
- **Leakage-aware validation is the crucial choice:**
  - Correlated sensors on the same river would **leak** across a naive split
  - Hold out **entire families of overlapping basins** → scores = transfer to **unseen** watersheds

**Visual:** `images/jay_xgb_transfer_allsites.png` (cross-site transfer) or `images/jay_holtwinters_monthly_vs_weekly.png` (classical baseline struggles).

**Speaker notes (~45s):** We use XGBoost — the weather–fertilizer–nitrate relationship is nonlinear and the data messy, and trees beat the classical baselines we tried. But the most important decision was validation: sensors near each other on the same river report correlated values, so a naive split leaks. We hold out entire families of overlapping basins, so every number reflects transfer to genuinely new locations, not memorization.

---

## Slide 4 — Results

- **Flagging violations (core KPI):** AUC **0.82** · PR-AUC **0.63** vs 26% base → **2.4× lift**
- Tuned for recall: catches **~86% of violation days** (~60% of alarms false — one knob dials it)
- **Per-site AUC 0.90** — nails *when* nitrate spikes · calibrated (**Brier 0.137**)
- **Concentration is harder:** R² **≈ 0.33**, ~4 mg/L error · within-basin 0.42 / between-basin 0.21
- → strong **"when & roughly how bad,"** not a calibrated meter

**Visual:** `images/preet_pr_curve_spatial_cv.png` (spatial-CV precision–recall — one line per held-out basin family, above the base-rate floor). Secondary: `images/preet_feature_importance.png`.

**Speaker notes (~65s):** On watersheds it's never seen, the classifier hits AUC 0.82 and PR-AUC 0.63 against a 26% base rate — a 2.4× lift over chance. Tuned to prioritize catching violations, it flags about 86% of real violation days; the cost is that ~60% of alarms are false, and a single recall-vs-precision knob dials that. Within any basin, per-site AUC is 0.90 — it's excellent at timing. Predicting exact concentration is harder: about a third of the variance, ~4 mg/L error, and much better at tracking movement within a basin than ranking a brand-new basin's absolute level.

---

## Slide 5 — Interpretability, Limits & Impact

- **Learned signals are physically sensible** (gain + permutation agree):
  - Statewide nitrate today · **location/distance geometry** · **near-field corn** · multi-week rain lag · spring peak
- **Limits:** novel far-from-training sites; 85 sites ≠ full diversity; annual data misses within-season shifts
- **Next:** **point-source (manure) data** — biggest expected gain · graph models of the river network
- **Who acts sooner:** utilities (treat early) · agencies (target monitoring $) · farmers (time fertilizer)
- Where the alternative is *no data at all*, imperfect prediction is a major upgrade

**Visual:** `images/preet_feature_importance_breach.png` (ranked importance for the violation model) or `images/erin_ftest_diagnostics.png` (seasonality).

**Speaker notes (~55s):** The model's behavior is sensible: land near the sensor beats distant basin area, rain lag beats same-day rain, clear spring spike — and permutation importance confirms it leans on what's immediately upstream. Limits: it may fail at genuinely novel sites, our 85 don't capture all of Iowa, and annual data misses within-season change. The biggest next gain is point-source manure data, which location is proxying for now. Bottom line: utilities, agencies, and farmers can all act sooner — and where budget cuts mean the alternative is no information at all, even imperfect prediction is a real upgrade. Thank you.

---

## Appendix — figure reference

| Slide | Figure | Shows |
|---|---|---|
| 1 | `erin_surplus_2005.png` | Iowa N-surplus hotspots (250 m) |
| 2 | `basin_over_crops.png` | Real basin, grid cells by dominant crop, sensor pin |
| 2 | `preet_crop_map_2017.png` | Corn–soy land-use footprint (alt) |
| 3 | `jay_xgb_transfer_allsites.png` | Cross-site transfer |
| 3 | `jay_holtwinters_monthly_vs_weekly.png` | Classical baseline struggles (alt) |
| 4 | `preet_pr_curve_spatial_cv.png` | Classifier PR under spatial CV |
| 4 | `preet_feature_importance.png` | Ranked feature importance (regression) |
| 5 | `preet_feature_importance_breach.png` | Ranked importance (violation model) |
| 5 | `erin_ftest_diagnostics.png` | Seasonal structure (alt) |

### Numbers cheat-sheet (honest LOFO CV)

| Metric | Value |
|---|---|
| Classifier ROC-AUC | 0.82 |
| Classifier PR-AUC (base 0.26) | 0.63 → **2.4× lift** |
| Recall @ β=2 operating point | ~86% (FDR ~60%) |
| Per-site (macro) AUC | 0.90 |
| Brier (vs 0.19 base floor) | 0.137 |
| Regressor R² (unseen basin) | ~0.33 |
| Regressor RMSE | ~4 mg/L |
| Within / between-basin R² | 0.42 / 0.21 |
