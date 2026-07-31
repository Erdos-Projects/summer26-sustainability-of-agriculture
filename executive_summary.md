# Executive Summary

**Project members:** Erin Bevilacqua, Xiaoying He, Rajpreet Kaur, Jay Lee, Isaac Martin

**Github Repo:** <https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture>

**Deployed Widget:** <https://erdos-projects.github.io/summer26-sustainability-of-agriculture/>

**Project Objective:** Iowa grows 2.5 billion bushels of corn a year, which demands heavy nitrogen fertilizer; the excess leaches into rivers as nitrate, a Group 2A carcinogen. Iowa's drinking water routinely exceeds the 10 mg/L federal limit — far above the 1–2 mg/L background level and the 3 mg/L threshold now linked to elevated cancer risk — a likely contributor to the state's rising cancer rate. Real-time nitrate sensors are sparse, costly, and shrinking under budget cuts, leaving utilities, regulators, and farmers blind across most of the state. We ask: **can we predict nitrate concentration and violation risk at any Iowa location from publicly available weather and land-use data alone, with no physical sensor?**

**Data preparation and computing workflow:** 
We filtered 162 candidate sensors to 83 high-quality sites, of which 81 carry enough history to train on (158,215 daily nitrate records). For each, we computed and hand-validated a drainage basin using one of three distinct methods, then stitched three gridded layers onto it: annual satellite crop cover, annual soil nitrogen surplus, and daily weather. Feature engineering was a major effort, with three findings driving the design:

- Proximity dominates — land near a sensor matters far more than the basin's outer reaches, so we bucketed features by distance and weighted them by exponential decay toward the sensor.
- Geography is a strong proxy — latitude/longitude and basin geometry proved surprisingly predictive, likely standing in for unmeasured local factors like nearby livestock operations.
- Seasonality is real, leakage is a trap — violations are seasonal beyond weather alone (sine/cosine terms helped), and because upstream basins nest inside downstream ones, we held whole basin families together across the train/test split to prevent spatial leakage.

**Statistical Validation and KPIs:** 
Two early dead ends shaped our approach. First, classical time-series forecasting (e.g., Holt-Winters) is nonsensical for our target; it needs a site's own history, which a virtual site lacks. Second, per-site models underperformed even a dummy-mean baseline (negative test R²) for want of data. Pooling all sites into one XGBoost model was the unlock — untuned, it already reached 0.24 R² (regression) and 0.51 PR-AUC (classification) on unseen sites, the benchmark the tuned pipeline was measured against.

Final tuned results below, reported using CV designed to prevent leakage from nested basins: 

- **Classification (violation flagging):** ROC-AUC 0.87, PR-AUC 0.71 against a 25.8% base rate (2.75× better than chance), and per-site AUC 0.90 — excellent at timing when a basin spikes. A Brier score of 0.123 (vs. a 0.191 base-rate floor, a 36% skill improvement) confirms well-calibrated probabilities. Tuned toward recall, it catches 90% of violations at a 54% false-discovery rate, with a knob to trade recall against precision.

- **Regression (concentration):** R² ≈ 0.43 (up from the 0.24 baseline), ~4.2 mg/L typical error. To account for the high variability in average site nitrate values we report two mean-normalized scores: within-basin R² (0.42) calculates R² after subtracting a site's mean from both the prediction and the actual, and between-basin R² (0.40) compares the predicted and actual site means. The second was long the weaker of the two — the regressor could tell you when a basin would spike but not how bad an unseen basin was in absolute terms. Adding long-run land-use composition roughly doubled it, from 0.20, which is what moved the headline R² as well.

These numbers are the deployed `light_CLF` and `light_REG` models — the pair restricted to the feature set a browser can assemble at an arbitrary point, and therefore exactly what the widget runs. They were retrained on 2026-07-30 with the long-run composition block; the full `recipe_*` models have not been, so they currently score *below* the light pair (PR-AUC 0.695, R² 0.380). See [`kpis.md`](kpis.md) for both.

**Statistical tests and deployment implications:** 
An F-test confirmed nitrate's seasonality survives controlling for rainfall seasonality, ruling out a naive "high in June" shortcut. Gain- and permutation-based importance agree on the backbone: statewide daily nitrate, sensor location/geometry, and near-field corn. 

Key limitations: a sparse sensor fleet, annual crop/soil snapshots that miss within-season change, and extrapolation risk at unfamiliar basins — and nitrate ultimately depends on more than weather and crops. Natural next steps: point-source pollution data (livestock manure, which location may be proxying for), soil/tile-drainage data, fertilizer timing, and graph models exploiting hydrological network structure.

The work is deployed as a public web widget at <https://erdos-projects.github.io/summer26-sustainability-of-agriculture/>: drop a pin anywhere in Iowa and it snaps to the nearest mapped stream reach, whose drainage basin and ~50-feature set were computed in advance. It then scores the models entirely in browser to return predicted-nitrate and violation-probability curves — no local history needed — with a tunable recall/precision knob.

**Executive Impact:**
Nitrate predictions let three groups act sooner and more precisely than the current sparse sensor network allows. Water utilities can begin treatment before a spike hits rather than reacting after the fact. Regulatory agencies can target monitoring and remediation dollars toward high-risk unmonitored locations rather than spreading resources evenly. Farmers can time fertilizer application around forecasted rain to reduce runoff. The model is not a substitute for physical sensors in regulatory compliance contexts, but in a state where budget cuts mean the realistic alternative at most locations is no information at all, an imperfect, well-calibrated prediction represents a substantial upgrade in situational awareness.