# KPIs

Primary KPI: **can we flag a nitrate-violation day (≥ 10 mg/L) at a location with no sensor?** This is the classification (CLF) task.

Secondary KPI: Same question but with a regression target, **can we predict nitrate concentration (mg/L) timeseries at unseen sites?** This is the regression (REG) task.

We discuss our results below with a comparison to the figures we discuss in our presentation video. Below that are sections describing our cross-validation strategy and the definitions for all the metrics we use to score our models.

# Results

Here are tables of our metrics, presented alongside the figures quoted in our presentationa. The comparison is unfair in favor of the older models in two ways:
- The default cross-validation technique at the time of the video was LOSO (optimistic) whereas the figures we report here default to LOFO (conservative). See the CV section below for a full discussion.
- The figures we report in this widget come from our `light_REG` and `light_CLF` models. These were optimized specifically to run in-browser, but sacrifice some performance over what is possible.

Nonetheless, our light models still vastly outperform the old models:

**Regression — nitrate concentration (mg/L):**

|  | LOSO R² | LOFO R² | between-site R² | within-site R² | macro-R² | site AP | captured |
|:---|---:|---:|---:|---:|---:|---:|---:|
| At time of video: | 0.3800 | 0.3280 | 0.2094 | 0.4187 | 0.2445 | — | — |
| Widget's `light_REG`: | **0.4727** | **0.4251** | **0.3959** | **0.4212** | **0.2768** | **0.493** | **0.351** |
| ∆ improvement: | **+0.0927** | **+0.0971** | **+0.1865** | ≈ equal | **+0.0323** | — | — |

**Classification — violation ≥ 10 mg/L:**

|  | LOSO AUC | LOFO AUC | LOFO PR-AUC | Lift vs base | between-rate R² | macro AUC | site AP | captured |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| At time of video: | 0.8358 | 0.8066 | 0.6249 | 2.38× | 0.2201 | 0.8938 | — | — |
| Widget's `light_CLF`: | **0.8768** | **0.8702** | **0.7105** | **2.75×** | **0.4980** | **0.8983** | **0.614** | **0.531** |
| ∆ improvement: | **+0.0410** | **+0.0636** | **+0.0856** | **+0.38×** | **+0.2779** | ≈ equal | — | — |

Blank cells are metrics that did not exist when that model was scored. Definitions of every metric can be found in the sections below.

These gains were achieved by many small optimizations including
- further improvements to basin accuracy
- more feature engineering
- better XGBoost tuning protocols.

## Interpretation of results
In words:
- If you're willing to tolerate a lot of false alarms, the classifier is pretty good at identifying violation days. 
- The actual predictions the regressor produces shouldn't be trusted, but it can be used to identify trends. 
- Both the classifier and the regressor could be used to identify a shortlist of unmonitored, potentially dangerous sites to then go and field test prior to sensor installation.

In numbers:
- `LOFO-PR AUC at 0.71:` The main metric. This measures the classifier's ability to pick out the days that are true positives. It should be compared to the base violation rate, which is 0.2584 in this case, so the classifier is 2.75× better than chance at identifying violation days (on average across all thresholds). At our default threshold choice, it means our model can be trusted to detect **90% of actual violation days anywhere in Iowa**, but that 54% of the alarms it raises will be false (see the β-table below). At more aggressive tuning, you catch 97% of true positives at a 64% false alarm rate.
- `LOFO R²:` Our regression model explains 43% of day-to-day variation at unseen sites. Don't trust it for precise nitrate values, but it is a good indicator of trends.
- `between-site + site AP + captured:` Both the classifier and the regressor are good at picking out particularly dangerous sites; the classifier is better. You can use them to identify a shortlist of potentially unmonitored dangerous sites.

# Cross Validation

We evaluate our methods using various scores (see below) using a few different cross validation metrics:
- **LOFO -- Leave One FAMILY Out:** primary CV technique, splits train/test based on hydrological connection. A hydrologically connected family of sites is always kept together across the split. Most robust to data leakage. Is a conservative metric. All metrics listed are this unless otherwise specified with a prefix. We actually use GroupKFold with $k = 5$ for this, 85/15.
- **LOSO -- Leave One Site Out:** secondary CV technique, performs train/test split without reference to hydrological connection. It still never splits data from a site across train/test; sites are highly autocorrelated so this essentially trivializes the problem -- you train a model to memorize each site and then learn to identify which site it is looking at from the static geographic features. Also GroupKFold with $k = 5$, attempt to meet, 85/15.
- **True_LOFO:** Same thing as LOFO but we actually hold one whole family out for testing rather than doing GroupKFold.
- **LODO_d -- Leave One Distance family Out:** A mix between True_LOFO and True_LOSO, hold out one site for testing, train on everything which is either not connected to the site OR is at least $d$ meteres away by flow distance.

# Metrics for CLF (violation ≥ 10 mg/L)

Headline pair is **`lofo_prauc_lift`** + **`lofo_auc`**. The former is **`prauc`** below divided by the base-violation rate — how much the CLF model outperforms coin-flipping. Formulas and descriptions are given below. Note these definitions are independent of cv technique; for most we only track `lofo_` values, but we track `loso_auc` as a comparison point to `lofo_auc`.

Take the pooled table of (truth y, prediction p, site g) over all 158,215 rows and 81 sites, in 20 basin families.

##### `prauc` — "of the rows the model flags hardest, how many are real?"
Average precision: sweep every threshold and integrate precision against the recall it buys, so a rare positive class is never rewarded for the true negatives it gets for free.
$$\mathrm{AP} = \sum_k \big(R_k - R_{k-1}\big), P_k, \qquad P_k = \mathrm{Prec}(\tau_k),\ R_k = \mathrm{TPR}(\tau_k)$$

##### `prauc_lift` — "how much better than guessing?"
Average precision divided by the base rate, which is what a random ranker scores. Imbalance-normalised, so it reads the same way at any prevalence — but bounded above by $1/\pi$ (3.88 here), so it is not comparable across cohorts whose base rates differ.
$$\text{lift} = \mathrm{AP}/\pi, \qquad \pi = \tfrac{1}{N}\sum_i y_i$$

##### `auc` — "does a random violation outrank a random non-violation?"
ROC-AUC: the probability that a randomly drawn positive row scores above a randomly drawn negative one, integrated over the entire ranking including thresholds nobody would deploy.
$$\mathrm{AUC} = \Pr\big(p_i > p_j ,\big|, y_i = 1,\ y_j = 0\big) = \int_0^1 \mathrm{TPR}, d(\mathrm{FPR})$$

##### `br2 (between_rate_r2)` — "which basins are the bad ones?"
Collapse each site to its observed violation rate and its mean predicted probability, then take R² over those 81 points, unweighted by row count.
$$R^2_{\text{between-rate}} = 1 - \frac{\sum_{s\in\mathcal S}\big(\bar y_s - \bar p_s\big)^2}{\sum_{s\in\mathcal S}\big(\bar y_s - \bar{\bar y}\big)^2}, \qquad \bar y_s = \tfrac{1}{n_s}\!\!\sum_{i:,s(i)=s}\!\! y_i$$

##### `macro_auc` — "how well does it time violations at a typical site?"
ROC-AUC computed separately on each site's own rows, then the median across sites — one site one vote, so unlike `auc` it is not dominated by long-record sites. Sites where AUC is undefined (no violations, or no non-violations) drop out of the median rather than counting as zero.
$$\text{macro-AUC} = \operatorname*{median}_{s} \mathrm{AUC}_s$$

##### `f2 (recall_at_f2)` — "at the shipped alarm setting, what share of violations do we catch?"
Pick the threshold that maximizes $F_\beta$ at $\beta = 2$ (recall weighted $4\times$ precision, the deployed operating point) and read off the true-positive rate there.
$$\tau_2 = \operatorname*{arg,max}_\tau \frac{5\,\mathrm{Prec}(\tau)\,\mathrm{TPR}(\tau)}{4\,\mathrm{Prec}(\tau) + \mathrm{TPR}(\tau)}\, \qquad \text{recall@}F_2 = \mathrm{TPR}(\tau_2)$$

##### `fdr_at_f2` — "at that same setting, what share of the alarms are false?"
The complement of precision at the identical $\tau_2$, which is why it must always be quoted beside f2 — recall bought by lowering the threshold shows up here as cost.
$$\mathrm{FDR@}F_2 = 1 - \mathrm{Prec}(\tau_2) = \frac{\mathrm{FP}(\tau_2)}{\mathrm{TP}(\tau_2) + \mathrm{FP}(\tau_2)}$$

##### `brier` — "when it says 30%, does it happen 30% of the time?"
Mean squared error of the predicted probability against the 0/1 outcome — a calibration score, not a ranking one, so unlike prauc and auc it moves when the probabilities are shifted even if the ordering is untouched. Lower is better and $0$ is perfect, but the number is unreadable alone: the reference point is the climatology forecast $p_i \equiv \pi$, which scores $\pi(1-\pi) = 0.258 \times 0.742 = 0.1914$ on this cohort. That gives the skill score $\mathrm{BSS} = 1 - \mathrm{Brier}/\pi(1-\pi)$, so `recipe_CLF`'s 0.1288 is a BSS of 0.33 (`light_CLF`'s 0.1285 likewise).

$$\mathrm{Brier} = \frac{1}{N}\sum_i \big(p_i - y_i\big)^2$$

##### `base` — "how often does a violation happen at all?"
The pooled violation rate. Not a score in itself: it is the reference every other CLF number is read against — the floor `prauc` is divided by, the $\pi(1-\pi)$ in the Brier skill score, and the prevalence at which `fdr` is quoted.
$$\pi = \tfrac{1}{N}\sum_i y_i = 0.258$$


##### `site_ap` — "are the worst basins at the top of the list?"
A measure of the model's accuracy in ranking the top 10% worst sites (highest average nitrate). Same as the formula described in detail below for the regression target, but here we rank sites by *violation rate* rather than average nitrate concentration.

##### `captured` — "how much of the achievable badness would a shortlist find?"
Compare the badness of the sites predicted to be worst to the badness of the actual worst sitse. Take the $k$ sites the model ranks worst, call them $\mathcal P$, and let ($\mathcal T$) be the actual $k$ worst sites. Let $\bar y_s$ be the average violation rate of site $s$ (actual, not predicted) and let $m$ be the average violation rate across all sites. Then calculate
$$\text{captured} = \frac{\sum_{s \in \mathcal P} \bar y_s - m}{\sum_{s \in \mathcal T} \bar y_s - m}.$$
0 is a random shortlist, 1 the best achievable.

# Metrics for REG (nitrate mg/L)
These are more self explanatory. The primary one is `lofo_r2` which captures the same thing as `rmse`. The triple `between_r2`, `within_r2` and `macro_r2` are for measuring inter-site dynamics in various ways.

##### `lofo_r2` — "how much of the total nitrate variance is explained?"
Ordinary R² over all 158,215 pooled rows against the global mean, so it is row-weighted and long-record sites dominate it.
$$R^2 = 1 - \frac{\sum_i (y_i - p_i)^2}{\sum_i (y_i - \bar y)^2}$$

##### `loso_r2` — "how much does hydrological leakage flatter us?"
The same R² on the LOSO out-of-fold vector instead of the LOFO one, and for the same reason always the higher of the two. For the deployed regressor: 0.473 LOSO against 0.425 LOFO.

##### `rmse` — "how far off is a typical prediction, in mg/L?"
The same error as lofo_r2 but left in the target's units instead of normalized by variance, which makes it comparable across cohorts where $R^2$ is not. We only record lofo evaluation.
$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (y_i - p_i)^2}$$

##### `between_r2` — "which basins are bad?"
Collapse every site to two numbers, its mean truth and its mean prediction, then take R² over those 81 points, unweighted by row count:
$$R^2_{\text{between}} = 1 - \frac{\sum_s (\bar y_s - \bar p_s)^2}{\sum_s (\bar y_s - \bar{\bar y})^2}$$
This is a measure of the model's ability to rank the magnitude of sites collectively.

##### `within_r2` — "when does it spike?"
Broadcast each site's mean back to its rows and subtract it from both truth and prediction, then take R² on what's left:
$$R^2_{\text{within}} = 1 - \frac{\sum_i \big[(y_i - \bar y_{s(i)}) - (p_i - \bar p_{s(i)})\big]^2}{\sum_i (y_i - \bar y_{s(i)})^2}$$
Measures whether, when the site diverges from its mean, the predictions also diverge from their mean. This is why it's intended to gauge the model's spike detection ability.

##### `macro_r2` — "how does the typical site do?"
Compute R² separately on each site's own rows, then take the median across sites. Sites where R² is undefined (a single row, or zero variance in the truth) drop out of the median rather than counting as zero:
$$\text{macro-}R^2 = \operatorname*{median}_{s} R^2_s$$
So this is same as `lofo_r2` but restricted to the median site.

##### `site_ap` — "are the worst basins at the top of the list?"
A measure of the model's accuracy in ranking the top 10% worst sites (highest average nitrate). Order all sites from worst to best (highest to lowest average nitrate) so that site $s_1$ has the highest nitrate average, site $s_2$ the second highest, etc. Now order them by the predicted values produced during the GroupKFold split, so that for a site $s$, $\ell_{s}$ is its rank as predicted by the model during CV. Then for the actual top offending sites $s_1,...,s_k$, calculate the average:
$$\text{site\_ap} = \frac1k \sum_{i=1}^k \frac{\# \text{worst sites in the predicted top }\ell_{s_i}}{\ell_{s_i}}.$$
This is easier to see in a table using the 79 sites which actually appear in training:
| rank i | site | actual (mg/L) | found so far | precision at i |
|---:|:---|---:|:---:|---:|
| 1 | WQS0104 | 14.90 | 1/8 | 1.000 |
| 3 | WQS0054 | 13.95 | 2/8 | 0.667 |
| 4 | WQS0048 | 12.34 | 3/8 | 0.750 |
| 7 | WQS0055 | 15.82 | 4/8 | 0.571 |
| 14 | WQS0043 | 11.11 | 5/8 | 0.357 |
| 15 | WQS0056 | 16.21 | 6/8 | 0.400 |
| 16 | WQS0114 | 18.93 | 7/8 | 0.438 |
| 17 | USGS-05481000 | 10.76 | 8/8 | 0.471 |

For these values, the average precision is $0.5817$, whereas randomly guessing would give you an average precision of $k/n \approx 0.1$. In this case, the model is 5.7 $\times$ better than chance at predicting the top 10%.

Note this is a conservative estimate: when the model is asked to predict WQS0114, the top offending site at $18.93$ mg/L, the worst it has seen is $16.21$ mg/L, and it fails to extrapolate. Every other top 10 site has WQS0114 available in training and the model is able to successfully identify those sites as worse.

##### `captured` — "how much of the achievable badness would a shortlist find?"
Compare the badness of the sites predicted to be worst to the badness of the actual worst sitse. Take the $k$ sites the model ranks worst, call them $\mathcal P$, and let ($\mathcal T$) be the actual $k$ worst sites. Let $\bar y_s$ be the average nitrate of site $s$ and let $m$ be the average of all averages. Then calculate
$$\text{captured} = \frac{\sum_{s \in \mathcal P} \bar y_s - m}{\sum_{s \in \mathcal T} \bar y_s - m}.$$
0 is a random shortlist, 1 the best achievable.

# Deployed operating point (β-table)

What actually ships in the widget is a **recall-emphasis knob β**, not a fixed cutoff. For each β, `src/models/tune_threshold.py` sweeps the LOFO out-of-fold, picks an optimal threshold $\tau$, and records the recall/precision there; the table + base rate are saved along with the model.

Here is the β-table for the classifier in this widget, `light_CLF`. The value of β determines the threshold τ, as well as the recall rate and the false detection rate (FDR).

| β | τ | recall | FDR |
|---|---|---|---|
| 0.5 | 0.489 | 0.525 | 0.271 |
| 1.0 | 0.251 | 0.736 | 0.388 |
| **2.0** | **0.077** | **0.903** | **0.540** |
| 3.0 | 0.047 | 0.947 | 0.598 |
| 4.0 | 0.030 | 0.970 | 0.642 |

By default we set **β = 2**, but this can be changed in the forecast section.