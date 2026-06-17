# Nitrate Exceedance Forecasting — Implementation Summary

A basin/sensor-level tool: given an area of Iowa and a crop or nitrogen-surplus
scenario over it, forecast for each downstream IWQIS stream sensor the
probability of exceeding 10 mg/L nitrate over a spring window. Groundwater is
deferred for now.

---

## Definitions

### Pollutant and threshold

- **Nitrate (as N)** — a soluble form of nitrogen that washes off farmland into
  water. "As N" means the concentration is reported as the mass of nitrogen
  atoms per liter, which is the convention the regulatory limit uses.
- **mg/L** — milligrams per liter, the concentration unit.
- **MCL (Maximum Contaminant Level)** — a regulatory ceiling for a contaminant
  in drinking water. For nitrate it is 10 mg/L as N, set by the
  **EPA (Environmental Protection Agency)**. This is the threshold your yes/no
  target is built around.

### Agriculture / the input side

- **Nitrogen surplus** — applied nitrogen minus the nitrogen crops take up, per
  unit area (e.g. kg N/ha). It is the excess left in the soil that can leach to
  water; it is the quantity your model treats as the manipulable cause.
- **CDL (Cropland Data Layer)** — a yearly raster from the
  **USDA (U.S. Department of Agriculture)** giving the crop type in each ~30 m
  pixel across the country. You use it to know what was grown where, each year.
- **Per-crop nitrogen budget** — a lookup mapping each crop to a typical surplus
  (corn high, soybean near zero or negative because it fixes its own nitrogen).
  A standard source is Iowa State's **MRTN (Maximum Return to Nitrogen)** rate
  guidance. This is the function that turns a crop map into a surplus number.

### Hydrology / the river network

- **Drainage basin (= watershed = contributing area)** — for a given point on a
  stream, the set of all land whose surface water eventually flows through that
  point. "Upstream of X" means inside X's basin; "downstream of X" means X's
  water reaches it.
- **Catchment** — the smallest local drainage unit in the digital river
  network: the strip of land draining directly to one stream segment.
  Catchments tile the landscape with no gaps.
- **HUC12 (Hydrologic Unit Code, 12-digit)** — a standardized nested watershed
  boundary; the 12-digit level is a small subwatershed (tens of km²). An
  alternative, coarser tiling to catchments.
- **NHDPlus (V2)** — the **USGS (U.S. Geological Survey)** national digital
  river network plus its catchments; the data layer everything hydrologic is
  built on.
- **NLDI (Network-Linked Data Index)** — a USGS web service that walks the
  NHDPlus network. It's what you already used to compute each sensor's upstream
  basin.
- **Tile drainage** — buried perforated pipes under Iowa farm fields that drain
  excess water. They are the dominant fast route carrying nitrate to streams,
  which is why your stream signal responds strongly to farmland loading.
- **Spring nitrate flush** — the seasonal peak (roughly April–June) when rain,
  snowmelt, and tile flow are high and crops haven't yet taken up much nitrogen,
  so stream nitrate spikes. Your model targets this window.
- **Discharge (streamflow)** — the volume of water passing a point per unit
  time. Concentration is roughly load ÷ discharge, so discharge is the dilution
  term.
- **Load** — mass of nitrate per unit time (concentration × discharge).
- **Attenuation** — the gradual loss of nitrate as water travels downstream
  (microbes convert it to nitrogen gas, plants absorb it), so far-upstream land
  contributes less at a sensor than nearby land.

### Data sources / APIs

- **IWQIS (Iowa Water Quality Information System)** — the network of
  high-frequency in-stream nitrate sensors that produce your outcome data
  (surface water, i.e. streams).
- **NWIS (National Water Information System)** — USGS service for streamflow and
  water data; your discharge source.
- **NASA POWER** — a NASA meteorological dataset (precipitation, temperature,
  etc.); your weather source.

### Modeling terms

- **Influence matrix (M)** — a precomputed sparse matrix where M[unit, sensor]
  is the attenuation-weighted fraction of that unit's nitrogen load reaching
  that sensor. Multiplying a per-unit loading vector by M gives the loading
  arriving at each sensor.
- **Counterfactual** — a prediction under a hypothesized intervention ("set this
  land's surplus to X"), as opposed to passively forecasting what will happen.
- **Baseline** — the observed recent loading on land the user did *not* change,
  held fixed during a scenario.
- **Random intercept (hierarchical / mixed model)** — a per-group offset; here
  one per sensor, absorbing that sensor's stable baseline level so the shared
  predictors capture variation *around* it. It lets each sensor sit at its own
  level without free parameters that would otherwise soak up the loading signal.
- **Calibration** — whether predicted probabilities match observed frequencies
  (of the cases you call 30%-risk, about 30% should actually exceed). Distinct
  from accuracy.
- **GBM (gradient-boosted machine)** — a tree-ensemble regressor; strong on raw
  fit but poor at extrapolating beyond the training range.
- **Overlap fraction** — the share of a sensor's basin covered by the user's
  selection; it bounds how much any scenario can move that sensor.

---

## Implementation steps

**1. Fix the spatial units (precompute once).**
Partition the landscape into a gap-free tiling of drainage units — NHDPlus
catchments or HUC12 subwatersheds. Tag each with its agricultural fraction from
CDL, and mark predominantly non-agricultural units (urban, water) as
non-manipulable: the user can't change crops there, but they stay in the
accounting because they still contribute discharge. For each IWQIS sensor you
already have its upstream basin; record which units fall in it and each unit's
flow-distance to the sensor.

**2. Build the influence matrix (precompute once).**
For every (unit → downstream sensor) pair, compute an attenuation weight. Start
simple — area-weighted, optionally with a single fixed decay in flow-distance —
and assemble these into the sparse matrix M. Later you can fit the decay if
residuals demand it.

**3. Construct the training table.**
One row per (sensor, year), for the fixed spring window.
- *Target:* the spring-season maximum (or a high quantile) of daily-mean nitrate
  at the sensor, on a log scale (nitrate is roughly log-normal).
- *Loading feature:* reconstruct each year's per-unit nitrogen surplus from that
  year's CDL crops × the per-crop budget, then map to each sensor through M to
  get that sensor-year's basin loading.
- *Weather/dilution features:* spring-aggregate weather (total precipitation,
  temperature/growing-degree days) from NASA POWER, plus that spring's observed
  discharge from NWIS.
- Do **not** hand-engineer soil/basin covariates; a per-sensor random intercept
  will absorb them in the next step.

**4. Fit the model.**
A hierarchical regression predicting log spring-max nitrate from basin loading +
spring weather + discharge as shared (fixed) effects, with a per-sensor random
intercept for stable basin character. Loading's effect is identified from
within-sensor, across-year variation (crop rotation flips surplus year to year),
so the intercept absorbs the level without eating the loading lever. Predict
mean and variance, then convert to P(exceed) = P(predicted log-nitrate > log 10).
Benchmark a GBM for raw accuracy, but deploy the parametric form, because
scenarios can push loading past the historical range where a GBM extrapolates
unreliably.

**5. Validate, with two go/no-go baselines.**
Cross-validate by holding out whole years (tests unseen weather) and whole
nested sensor clusters (so overlapping basins can't leak between train and
test). Check calibration, not just accuracy, since you ship probabilities.
Compare against (a) each sensor's historical exceedance rate and (b) a
loading-blind model. If loading doesn't beat the loading-blind model, the
counterfactual lever isn't real and the rest isn't worth building — run this
comparison early, on a hydrologically similar subset of basins (e.g. the heavily
tile-drained north-central region) where the loading signal is cleanest.

**6. Run scenarios in the interactive tool.**
The user selects manipulable units and sets a management scenario over them —
either a crop mix (converted to surplus via the per-crop budget) or a surplus
value directly — optionally via worst / observed-recent / low presets. For each
sensor that has any selected unit in its basin, total loading = scenario loading
over the selected units + observed-recent baseline over the rest of that basin,
all routed through M.

**7. Marginalize over weather and report.**
Reuse historical springs as the weather ensemble: for each past spring, take its
observed weather and discharge, predict P(exceed) under the scenario loading,
and average across springs. Report, per (scenario, sensor): the exceedance
probability with an uncertainty band (weather spread + model uncertainty), the
baseline probability for contrast, the overlap fraction, the spring window the
number is conditional on, and an extrapolation flag when scenario loading
exceeds the historical range.

---

## Deferred for now

Groundwater and its lag (and the lagged-loading feature that would proxy it),
the fitted attenuation kernel, the crop-distribution presentation layer if you
start in surplus space, and free-draw area selection (v1 selects units
directly).