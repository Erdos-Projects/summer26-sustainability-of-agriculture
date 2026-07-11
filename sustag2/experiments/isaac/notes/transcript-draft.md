# 5-Minute Presentation — Transcript Draft

*Iowa waterborne-nitrate prediction project. Audience: mixed (non-technical stakeholders, hydrology
& land-use policy advisors, data scientists). Target ~5:00 at ~150 wpm (~800 words). Sourced from
`writeup.md` and `team_eda_survey.md`.*

---

## Outline

**1. The real-world problem** *(~40s)*
- Iowa is the center of U.S. row-crop agriculture; fertilizer drives the economy.
- Excess soil nitrogen leaches into streams and drinking water as nitrate.
- Above the federal limit (10 mg/L) it's a health hazard — infant "blue baby syndrome," treatment costs for utilities.
- Monitoring is sparse and expensive: a few hundred sensors, most waterways and wells ungauged.

**2. Research question** *(~25s)*
- Can we predict nitrate risk at a location from public weather + land-use data alone — no physical sensor there? A "virtual sensor."
- Two framings: regression (how much nitrate) and classification (will it exceed 10 mg/L).

**3. The data** *(~70s)*
- 85 quality-filtered sensors (from 162), 2008–2025, state (IWQIS) + federal (USGS), 5–15-minute readings.
- Outcomes: daily-max nitrate; binary violation flag (~28% of days).
- Predictors are "what's upstream": compute each sensor's drainage basin, then layer three gridded datasets — crop type (satellite CDL), modeled nitrogen surplus (gTREND), daily weather — reconciled across resolutions (30 m / 250 m / 4 km).
- Key predictors: near-sensor crop mix (corn), nitrogen surplus, antecedent rainfall, location, distance-to-sensor.

**4. Modeling approach** *(~55s)*
- Gradient-boosted trees (XGBoost) for both tasks.
- Why: nonlinear response and interactions (rain × fertilizer); robust to messy/missing data; beat classical time-series (Holt-Winters/Prophet) decisively in head-to-head tests.
- Leakage-aware validation (leave-one-site-out and leave-one-basin-family-out) so scores reflect true generalization to unseen places.

**5. Findings** *(~75s)*
- Strong autocorrelation — "predict yesterday" is a very strong baseline for gauged sites.
- Classification generalizes to unseen basins from land-use + weather + seasonality alone (ROC-AUC ≈ 0.80) — the virtual-sensor result.
- Regression is harder (transfer R² ≈ 0.25–0.35).
- Physically sensible signals: near-field (0–2 km riparian) land cover dominates; multi-week antecedent rain matters more than today's rain (slow subsurface/tile-drainage transport); clear spring/early-summer seasonal peak; location is a strong proxy (likely for unmeasured point sources).
- Acting on the riparian finding measurably improved the classifier.

**6. Next steps** *(~40s)*
- Add point-source pollution data — our single biggest expected gain.
- Deploy the virtual sensor as an interactive drop-a-pin tool.
- Tune the classifier to prioritize catching violations (costly false negatives).
- Fill gaps in the lower-quality sensors; add spike/event prediction.

---

## Transcript

**[The problem]**
Iowa grows a huge share of America's corn and soybeans, and that agriculture runs on nitrogen fertilizer. But not all of that nitrogen stays in the soil. The excess leaches into streams, rivers, and groundwater as *nitrate* — and when nitrate in drinking water climbs above the federal safety limit of 10 milligrams per liter, it becomes a genuine health hazard, linked to "blue baby syndrome" in infants and other risks. Cities across the state spend heavily to treat it. The catch is that we can't see the problem clearly: nitrate sensors are expensive, so only a few hundred exist across all of Iowa. Most streams, and most communities drawing from them, are flying blind.

**[Research question]**
So our question was simple to state: *Can we predict nitrate risk at a location using only publicly available weather and land-use data — without a physical sensor there?* In effect, can we build a **virtual sensor**? We attacked it two ways: predicting the actual nitrate concentration, and — more importantly — predicting whether a given day will *exceed* the 10 mg/L limit.

**[The data]**
We started from 162 real sensors and, after quality-filtering for gaps and bad records, kept the 85 best, spanning 2008 to 2025, from both state and federal monitoring networks. From those we built two outcomes: the daily maximum nitrate level, and a yes/no *violation* flag — which fires on about 28% of days. For predictors, the key insight is that only what's *upstream* of a sensor matters. So for each site we computed its drainage basin — the land that actually drains to it — and then layered three datasets over that basin: crop type from satellite imagery, a model of nitrogen surplus in the soil, and daily weather like rainfall, temperature, and evapotranspiration. These come at wildly different resolutions, so a big part of the work was stitching them onto a common grid. The predictors that carried the most weight were the crop mix near the sensor, nitrogen surplus, recent rainfall, and location itself.

**[Modeling approach]**
For the models, we used gradient-boosted decision trees — XGBoost — for both tasks. We chose them deliberately: the relationship between rain, fertilizer, and nitrate is nonlinear and full of interactions, the data is messy, and in direct comparisons trees decisively beat classical time-series forecasters. But the most important modeling decision wasn't the algorithm — it was how we *validated*. Because the whole goal is prediction at *unseen* locations, we used leakage-aware cross-validation: we hold out entire sites, and even entire families of geographically nested basins, so our reported accuracy reflects true generalization to new places, not memorization of ones we've already seen.

**[Findings]**
A few things stood out. First, nitrate has strong memory — simply "predicting yesterday's value" is a remarkably good baseline where you already have a sensor. But the exciting result is the classification task: using land-use, weather, and seasonality *alone*, we can flag violation risk at a completely unmonitored basin with an ROC-AUC around 0.80. That's the virtual-sensor payoff. Predicting the exact concentration is much harder — transfer performance lands in the 0.25-to-0.35 range. Encouragingly, the signals the model learned are physically sensible: the land immediately around the sensor — the riparian zone, within about two kilometers — matters far more than the far reaches of the basin; rainfall from the *previous few weeks* matters more than today's, consistent with slow movement through soil and tile drains; and there's a clear spring-to-early-summer seasonal peak. When we acted on the near-field finding and added a fine-grained riparian feature, the classifier measurably improved.

**[Next steps]**
Where do we go next? Our single biggest expected gain is adding **point-source pollution** data — feedlots and treatment plants — which our location variable is currently only proxying for. Beyond that, we want to deploy the virtual sensor as an interactive tool where you drop a pin and get a forecast; tune the classifier to prioritize *catching* violations, since a missed exceedance is far costlier than a false alarm; and extend the work to fill data gaps in lower-quality sensors and to predict sudden spikes. The data infrastructure we built makes each of these a fast next step rather than a fresh start.

---

*Delivery notes: at ~150 wpm this runs ~5:00. If long, the tightest cuts are the resolution detail
in Data and the regression numbers in Findings.*
