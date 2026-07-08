# Plan: cell-feature → 7-day nitrate-violation probability

Hypothesis-testing plan for predicting the probability of observing
`nitrate_con > 10 mg/L` over a 7-day period at a monitoring site, and relating
that probability to per-cell features (rain, surplus, distance to sensor,
fraction of cell in basin).

## Feasibility (grounding)

| candidate site | 7-day windows w/ data | violation rate (>10) | lifespan |
|---|---|---|---|
| WQS0071 | 245 | 21% | 6.4 yr |
| WQS0102 | 137 | 28% | 9.9 yr |
| WQS0070 | 183 | 21% | 5.3 yr |
| USGS-06603750 | 55 | **91%** | 1.4 yr |

So ~130–250 windows and a ~20–28% base rate at good sites — workable, but
**~40–50 violation events means ≤4 predictors** before you're overfitting. Avoid
degenerate sites like USGS-06603750 (violates 91% of the time → no variance to
explain).

## The core structural insight (reframe the hypotheses)

The **outcome lives at the (site, 7-day window) grain**, but the **features live
at the (cell, window-or-static) grain**. Within one fixed basin:

- **rain** varies in time *and* space ✅
- **surplus** is annual and slow → effectively *constant in time*
- **dist_to_sensor, frac_cell_in_basin** are *static* (geometry)

So H1 is the only hypothesis with clean temporal variation in a single basin.
**H3 and H4 aren't about levels — they're about how to *weight* each cell's rain
contribution. And H2's pure effect isn't identifiable from one basin's time
series** (surplus barely moves), so within one basin H2 becomes "does rain over
high-surplus cells matter more" (a rain×surplus interaction). All four collapse
into **one weighted rain-loading index**:

$$L_t = \sum_{cells} \left(\frac{\text{frac area}_c^a \cdot \text{surplus}_c^d}{\text{dist}_c^b}\right)\cdot \text{rain}_{c, t-\tau(\text{dist}_c)}$$

$$\text{logit} P(\text{violation}_t) = \alpha + \gamma\cdot L_t$$

- H1 ⇔ γ > 0
- H2 ⇔ d > 0
- H3 ⇔ b > 0 (and/or lag τ grows with distance)
- H4 ⇔ a > 0

That index is the destination. The plan builds up to it.

### Why this form — component by component

The loading index splits into a **static spatial weight** times a **time-varying
driver**, summed over cells:

$$L_t = \sum_{c\,\in\,\text{cells}} \underbrace{\frac{\operatorname{frac}_c^{\,a}\,\operatorname{surplus}_c^{\,d}}{\operatorname{dist}_c^{\,b}}}_{w_c\ =\ \text{static weight}} \cdot \underbrace{\operatorname{rain}_{c,\;t-\tau(\operatorname{dist}_c)}}_{\text{time-varying driver}}$$

**Overall shape — $\sum_c(\text{weight})\times(\text{driver})$.** Each cell emits a
nitrate pulse into the stream this window: (mobilizing rain, time-shifted) ×
(a static factor for how much this cell matters to the sensor). Summing over
cells is a mass-balance / superposition idea — the load reaching the gauge is the
sum of per-source contributions. $L_t$ is the single scalar "nitrate loading
arriving in window $t$."

**Driver — $\operatorname{rain}_{c,\,t-\tau}$.** Rain is the transport mechanism:
precipitation drives leaching and tile/overland flow that flushes soil nitrate to
the stream. It is the only feature with real temporal variation, so it carries
the time-dependence of $L_t$ and enters linearly (twice the rain ⇒ ~twice the
flush, holding the cell fixed). This is what H1 ($\gamma>0$) tests.

**Transport lag — $\tau(\operatorname{dist}_c)$.** $\operatorname{dist}_c$ is
*flow* distance, so water from a far cell physically arrives later. $\tau$ shifts
each cell's rain back by its travel time so the sum aligns contributions that
genuinely co-arrive in window $t$. This is the mechanistic reading of H3 ($\tau$
increasing in distance).

**Distance attenuation — $1/\operatorname{dist}_c^{\,b}$.** Beyond timing, a longer
flow path means more in-stream denitrification, dilution, and storage, so a unit
of mobilized N from a far cell contributes less to the concentration peak. $b>0$
⇔ attenuation ⇔ H3; $b=0$ ⇒ distance doesn't down-weight.

**In-basin area correction — $\operatorname{frac}_c^{\,a}$.** `frac_cell_in_basin`
is the fraction of the (~4 km) cell actually inside the basin; only that portion
drains to this sensor. A cell straddling the divide should count proportionally
less. $a=1$ is the literal area-proportional correction; a free $a$ lets the data
say sub- or super-linear. $a>0$ ⇔ H4.

**Nitrogen availability — $\operatorname{surplus}_c^{\,d}$.** `surplus_kgha` is the
cell's reactive-N pool (applied − removed). Rain only delivers nitrate if there's
N to mobilize: the same rain over a high-surplus corn cell yields more than over
low-surplus pasture. So surplus **multiplies** rain's effect — an interaction,
not an additive term. This is exactly why, in a single basin where surplus is
time-static, H2 can only surface *as a weight on rain* ($d>0$).

**Why powers $a,b,d$.** They give a positive, scale-free, sign-constrained knob
per feature, and crucially **exponent $=0$ makes the factor $1$** ("this feature
doesn't matter") — so each hypothesis is a clean nested test of "exponent $>0$."
Equivalently the weight is log-linear,

$$\log w_c = a\log\operatorname{frac}_c + d\log\operatorname{surplus}_c - b\log\operatorname{dist}_c,$$

a standard interpretable parameterization (the weight is linear in the
log-features).

**Dose–response slope — $\gamma$.** Maps loading to log-odds; $\gamma>0$ means more
loading raises violation odds (the model's premise, and H1). Its size is the
loading→risk sensitivity on the log-odds scale.

**Baseline — $\alpha$.** Log-odds of violation at $L_t=0$: the basin's background
propensity to violate (baseline concentration / dilution regime).

**Why logistic — $\operatorname{logit}$.** The outcome is binary (violated in
window $t$ or not). The logit link keeps $P\in(0,1)$, is linear in the index, and
makes $\gamma$ a log-odds-ratio per unit loading. The whole thing is a
single-index GLM: collapse all cell features into one scalar $L_t$, then push it
through a sigmoid.

**Assumptions the form bakes in (it is a simplification):**

- *Linear superposition in rain* — ignores thresholds / antecedent-moisture
  nonlinearity in real transport.
- *Separable, multiplicative weight* — assumes frac, surplus, distance act
  independently.
- *One scalar index* — no cell-specific residual dynamics.
- *Identifiability* — $\alpha,\gamma,a,b,d$ plus the lag shape from only ~50
  violation events is a lot; fit it parsimoniously (fix $\tau$'s shape,
  grid-search $a,b,d$) or it will overfit.

---

## Plan

### 0. Fix the basin
Pick one balanced-base-rate site with all layers built (rain_grid w/ dist+frac,
rain timeseries, surplus_grid). **WQS0071** is a good first pick (245 windows,
21%). Confirm it has a rain grid and surplus_grid before committing.

### 1. Build the analysis table
- **Outcome (window grain):** `Y_t = 1 if max(nitrate_con) over the 7-day window > 10`.
  Use **non-overlapping** windows for inference (rolling windows inflate N with
  autocorrelation — fine for plots, not for p-values). Test threshold sensitivity
  later (8, 12 mg/L).
- **Cell features (cell × window):** `rain_{c,t}` = sum of `precip_in_1d` over the
  window (join rain↔grid on `node_id`); `surplus_c` for the window's year; static
  `dist_c`, `frac_c`.
- **Window-level aggregates** under several weighting schemes (this is where
  H3/H4 get tested).

### 2. EDA / sanity (do this first — it catches the biggest confound)
- Violation rate by **month**, and mean basin rain by month, on the same axis.
  Rain and nitrate both peak in spring/early summer → **seasonality is the #1
  confound.** Everything downstream must control for it (month dummies or a
  sin/cos harmonic) or you'll attribute season's effect to rain.
- Maps/distributions of dist, frac, surplus across cells.
- Correlation of simple basin-mean rain at lags t, t−1, t−2 with Y (transport delay).

### 3. Hypothesis-by-hypothesis

**H1 — rain ↑ → P ↑** *(start here; the clean one)*
- Aggregate basin-mean rain at lags 0/1/2.
- Tests: Mann–Whitney U (rain in violation vs non-violation windows); logistic
  `Y ~ rain + lags + season`, **one-sided** Wald test β_rain > 0; violation rate
  by rain decile → Cochran–Armitage trend test.

**H2 — surplus ↑ → P ↑** *(can't test on time alone in one basin)*
- *Within-basin proxy:* build a surplus-weighted rain index
  `Σ surplus_c·rain_{c,t}`, compare CV log-loss/AUC vs unweighted mean rain; test
  the rain×surplus interaction sign.
- *Proper test (cross-site):* across many basins, regress site violation rate on
  basin-aggregate surplus (pooled logistic, site as unit). This is the honest H2
  test — a multi-basin extension.

**H3 — distance ↓ impact** *(two readings, test both)*
- *Down-weighting:* rain index with `w_c = 1/dist_c^β` (or `exp(−dist/λ)`); fit
  β/λ by maximizing CV log-likelihood; H3 ⇔ β>0 beats uniform weights
  (likelihood-ratio / CV comparison).
- *Travel-time lag (mechanistic — dist is literally flow distance):* lag each
  cell's rain by τ ∝ dist; near-cell rain should correlate with Y at lag 0,
  far-cell rain at longer lag (cross-correlation). Test whether distance-lagged
  rain beats contemporaneous.
- *Stratified check:* split cells near/far at median dist; near-rain-agg should
  correlate with Y more strongly than far-rain-agg.

**H4 — frac ↑ impact**
- Compare frac-weighted rain (`w_c = frac_c`) vs unweighted index fit. (Partly a
  "true in-basin area" correction, so expect a modest but positive improvement.)
  Same stratified/interaction logic as H3.

### 4. Unified model (the payoff)
Fit the weighted-loading logistic above, exponents a,b,d ≥ 0 via grid search or
gradient. Each hypothesis = an LR test that its exponent > 0 (vs the nested model
with it fixed at 0/excluded), plus sign of γ for H1. Judge everything against the
**unweighted mean-rain baseline** with **cross-validated log-loss / AUC**, not
just in-sample p-values.

### 5. Pitfalls to bake in
- **Temporal autocorrelation:** consecutive windows are correlated → block
  bootstrap or HAC/cluster-robust SEs; naive logistic SEs are too small.
- **Seasonality:** control it everywhere (see §2) or your rain effect is inflated.
- **Rare events / separation:** ~50 events → Firth penalized logistic; keep ≤4
  effective predictors.
- **Multiple testing:** 4 directional hypotheses → pre-register one-sided tests or
  Holm-correct.
- **Spatial autocorrelation among cells:** cell features are correlated → don't
  treat cell-pooled rows as independent N; prefer window-level aggregates (or
  mixed models when you go multi-site).

---

**Recommended build order:** §2 (see the seasonality confound) → H1 with seasonal
control → layer H3/H4 as weighting-scheme comparisons on the rain index → H2 as a
weight within-basin, then as the cross-site extension → fold into the §4 unified
index.
