# 5-Minute Presentation — Slide Deck

*Slide-by-slide version of [transcript-draft.md](transcript-draft.md). One slide per section, ~40s
each, ~5:00 total. Each slide lists condensed on-screen bullets, a suggested visual (figures already
in [`images/`](images/)), and speaker notes. `---` marks slide breaks (Marp/reveal-compatible).*

---

## Slide 1 — Title

# Predicting Iowa's Nitrate Risk
### A virtual sensor from weather + land-use data

*Team names · date*

**Visual:** full-bleed Iowa nitrogen-surplus map — `images/erin_surplus_2005.png` (immediately shows the agricultural-intensity story).

**Speaker notes (~10s):** Title card. Let the map sit while you open.

---

## Slide 2 — The Problem

- Iowa = heart of U.S. row-crop agriculture, powered by **nitrogen fertilizer**
- Excess nitrogen leaches into water as **nitrate**
- Above **10 mg/L** (federal limit) → health hazard ("blue baby syndrome"), costly to treat
- Monitoring is **sparse & expensive** — most waterways ungauged

**Visual:** `images/erin_surplus_2005.png` (surplus hotspots) or `images/preet_crop_map_2017.png` (corn–soy footprint).

**Speaker notes (~40s):** Iowa grows a huge share of America's corn and soybeans, and that runs on nitrogen fertilizer. The excess leaches into streams and groundwater as nitrate — and above the federal limit of 10 mg/L it's a real health hazard, linked to blue-baby syndrome in infants. Cities spend heavily to treat it. But we can't see the problem clearly: sensors are expensive, so only a few hundred exist statewide. Most communities are flying blind.

---

## Slide 3 — Research Question

> **Can we predict nitrate risk at a location from public weather + land-use data alone — with no sensor there?**

- A **virtual sensor**
- Two framings:
  - **Regression** — how much nitrate (daily max, mg/L)
  - **Classification** — will it exceed 10 mg/L? *(the deployable target)*

**Visual:** `images/virtual_sensor_flow.png` — the drop-a-pin → weather/crops/surplus → XGBoost → P(violation) flow.

**Speaker notes (~25s):** Our question was simple to state: can we predict nitrate risk at a location using only publicly available weather and land-use data — no physical sensor there? A virtual sensor. We attacked it two ways: predicting the actual concentration, and — more importantly — predicting whether a day will exceed the 10 mg/L limit.

---

## Slide 4 — The Data

- **85 sensors** (quality-filtered from 162), 2008–2025, state (IWQIS) + federal (USGS)
- **Outcomes:** daily-max nitrate · violation flag (~28% of days)
- **Predictors = what's upstream:** compute each sensor's **drainage basin**, layer 3 gridded datasets:
  - Crop type (satellite CDL) · Nitrogen surplus (gTREND) · Daily weather
  - Reconciled across 30 m / 250 m / 4 km grids
- **Top predictors:** near-sensor corn, nitrogen surplus, antecedent rain, location, distance-to-sensor

**Visual:** `images/basin_over_crops.png` — a real basin tiled by weather-grid cells, each colored by dominant crop, draining to the starred sensor (the clearest "what's upstream" visual). Secondary: `images/isaac_basin_audit.png` (basin-scale spread across the fleet).

**Speaker notes (~70s):** We started from 162 real sensors and kept the 85 best after quality-filtering, from state and federal networks. Two outcomes: daily maximum nitrate, and a yes/no violation flag that fires about 28% of days. The key insight for predictors is that only what's *upstream* of a sensor matters — so for each site we computed its drainage basin, the land that drains to it, and layered three datasets over it: crop type from satellite, a model of soil nitrogen surplus, and daily weather. These come at very different resolutions, so a lot of the work was stitching them onto a common grid. The predictors that mattered most were the crop mix near the sensor, nitrogen surplus, recent rainfall, and location itself.

---

## Slide 5 — Modeling Approach

- **Gradient-boosted trees (XGBoost)** for both tasks
- Why: nonlinear response + interactions (rain × fertilizer); robust to messy/missing data; **beat classical time-series** (Holt-Winters / Prophet) head-to-head
- **Leakage-aware validation** — the crucial choice:
  - Leave-one-**site**-out and leave-one-**basin-family**-out
  - Scores reflect real generalization to **unseen places**, not memorization

**Visual:** `images/jay_holtwinters_monthly_vs_weekly.png` (classical baseline struggles) or `images/jay_xgb_transfer_allsites.png` (transfer across 82 sites).

**Speaker notes (~55s):** We used gradient-boosted decision trees — XGBoost — for both tasks, chosen deliberately: the relationship between rain, fertilizer, and nitrate is nonlinear and full of interactions, the data is messy, and in direct comparisons trees decisively beat classical time-series forecasters. But the most important decision wasn't the algorithm — it was validation. Because the goal is prediction at *unseen* locations, we hold out entire sites, and even entire families of nested basins, so our accuracy reflects true generalization, not memorization.

---

## Slide 6 — Findings

- **Nitrate has strong memory** — "predict yesterday" is a tough baseline (gauged sites)
- **Classification generalizes to unseen basins** from land-use + weather + seasonality alone — **ROC-AUC ≈ 0.80** → the virtual-sensor win
- **Regression is harder** — transfer R² ≈ 0.25–0.35
- Signals are **physically sensible**:
  - Near-field (0–2 km **riparian**) land cover dominates
  - **Multi-week antecedent rain** > today's rain (slow subsurface transport)
  - Clear **spring/early-summer** seasonal peak; location a strong proxy
- Acting on the riparian finding **measurably improved the classifier**

**Visual (pick 1–2):** `images/isaac_autocorrelation_rain.png` (memory) · `images/preet_pr_curve_final.png` (classifier works) · `images/erin_ftest_diagnostics.png` (seasonality).

**Speaker notes (~75s):** A few things stood out. First, nitrate has strong memory — "predict yesterday" is a remarkably good baseline where you already have a sensor. But the exciting result is classification: using land-use, weather, and seasonality alone, we flag violation risk at a completely unmonitored basin with an ROC-AUC around 0.80 — the virtual-sensor payoff. Predicting exact concentration is much harder, in the 0.25-to-0.35 range. And the learned signals are physically sensible: the land within about two kilometers of the sensor matters far more than the far basin; rain from the previous few weeks matters more than today's, consistent with slow movement through soil and tile drains; and there's a clear spring-to-early-summer peak. When we acted on the near-field finding and added a fine-grained riparian feature, the classifier measurably improved.

---

## Slide 7 — Next Steps

- **Point-source pollution** data (feedlots, treatment plants) — biggest expected gain
- **Deploy the virtual sensor** — interactive drop-a-pin forecast
- **Prioritize catching violations** — tune classifier for costly false negatives
- **Fill sensor gaps** + **spike/event prediction**
- Infrastructure makes each a *fast next step*, not a fresh start

**Visual:** `images/jay_xgb_transfer_4sites.png` (transfer to new sites) or a simple roadmap list.

**Speaker notes (~40s):** Where next? Our single biggest expected gain is adding point-source pollution data — feedlots and treatment plants — which our location variable currently only proxies for. Beyond that: deploy the virtual sensor as a drop-a-pin tool; tune the classifier to prioritize catching violations, since a missed exceedance is far costlier than a false alarm; and extend to filling sensor gaps and predicting spikes. The infrastructure we built makes each of these a fast next step rather than a fresh start. Thank you.

---

## Appendix — figure reference

| Slide | Figure | Shows |
|---|---|---|
| 3 | `virtual_sensor_flow.png` | Pin → inputs → XGBoost → P(violation) schematic |
| 4 | `basin_over_crops.png` | Real basin, grid cells by dominant crop, sensor pin |
| 1–2 | `erin_surplus_2005.png` | Iowa N-surplus hotspots (250 m) |
| 2, 4 | `preet_crop_map_2017.png` | Corn–soy land-use footprint |
| 4 | `isaac_basin_audit.png` | Basin-size heterogeneity |
| 5 | `jay_holtwinters_monthly_vs_weekly.png` | Classical baseline struggles |
| 5, 7 | `jay_xgb_transfer_{4sites,allsites}.png` | Cross-site transfer |
| 6 | `isaac_autocorrelation_rain.png` | Nitrate autocorrelation (memory) |
| 6 | `preet_pr_curve_final.png` | Classifier PR performance |
| 6 | `erin_ftest_diagnostics.png` | Seasonal structure |
