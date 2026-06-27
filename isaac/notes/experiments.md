# Experiments

Two different targets: nitrate regression and nitrate violation classification.
Two diferenent models: individual site models, general location models.

| | Nitrate Regression | Violation Classification |
|-|-|-|
| Fixed Site       | Baseline "predict yesterday's" value does incredibly well. Might be better to predict "will it spike?" | Idk what to put here, I guess "predict violation state of yesterday" does pretty well too |
| General Location | Can't use past nitrate | Can't use past nitrate |

Sections are roughly 
- Experiment description and results
- Score descriptions (this part was Claude generated because I got lazy at the end)
- Individual feature scores for each experiment

XGBoost provides its own feature scores, I also included one from sklearn, these are the "Col Shuffle" tables. Basically sklearn takes a trained model, tests it on some test data, and then one at a time it shuffles a column and applies the model again. If shuffling a column makes the model way worse, the column is important. Higher score = feature is more important.

## Table of Contents

- [Experiment descriptions and results](#summary-of-results-and-my-interpretation)
  - [Experiment 6: Does weather data improve regression?](#experiment-6-does-weather-data-improve-regression)
  - [Experiment 6c: Does weather data improve classification?](#experiment-6c-does-weather-data-improve-classification)
  - [Experiment 7: Which recipes perform best on individual site modeling?](#experiment-7-which-recipes-perform-best-on-individual-site-modeling)
  - [Experiment 7c:](#experiment-7c)
  - [Experiment 8:](#experiment-8)
  - [Experiment 8c:](#experiment-8c)
  - [Experiment 9:](#experiment-9)
  - [Experiment 9c:](#experiment-9c)
- [Regression score descriptions](#regression-score-table-columns-eg-_experiment6csv)
- [Classification score descriptions](#classification-score-table-columns-eg-_experiment6ccsv)
- [Experiment 6/6c: Does weather or lagged weather improve performance on multisite regression/classification?](#experiment-66c-does-weather-or-lagged-weather-improve-performance-on-multisite-regressionclassification)
  - [Exp 6 Feature Importance XGBoost](#exp-6-feature-importance-xgboost)
  - [Exp 6 Feature Importance Col Shuffle](#exp-6-feature-importance-col-shuffle)
  - [Exp 6c Feature Importance XGBoost](#exp-6c-feature-importance-xgboost)
  - [Exp 6c Feature Importance Col Shuffle](#exp-6c-feature-importance-col-shuffle)
  - [Exp 7 Feature Importance XGBoost](#exp-7-feature-importance-xgboost)
  - [Exp 7 Feature Importance Col Shuffle](#exp-7-feature-importance-col-shuffle)

## List of experiments, results, and my interpretation

### Experiment 6: Does weather data improve regression?
Yes, including the weather data improves results quite a bit. Lagged weather data doesn't help.

* No Weather recipe: basin-aggregated crop, basin-aggregated surplus, pure seasonality signal.
* 0_Lags recipe: No Weather + Daily weather
* 1_Lags recipe: 0_Lags + copy of every weather column lagged by 1 day
* 2_Lags recipe: 1_Lags + copy of every weather column lagged by 3 days
* 3_Lags recipe: 2_Lags + copy of every weather column lagged by 7 days
* 4_Lags recipe: 3_Lags + copy of every weather column lagged by 14 days

|recipe|n_sites|n_rows|n_feat|loso_r2|lofo_r2|rmse|between_r2|within_r2|macro_r2|
|-|-|-|-|-|-|-|-|-|-|
|0_Lags|20|64807|21|0.1164678371864057|0.0741736213315035|4.440714726959775|-1.2546736707196118|0.336055673316052|0.3069999264615034|
|1_Lags|20|64807|30|0.109123270218675|0.0778797931991754|4.4591337659720525|-1.2139896945653694|0.3277890879598506|0.3212673604491273|
|2_Lags|20|64807|39|0.019145698572732|-0.0103089765575898|4.67890194182973|-1.9288825942157324|0.3272861771521207|0.3312467989672863|
|3_Lags|20|64807|48|0.0177012602334372|0.0060428559603717|4.682345826928637|-1.852240292684313|0.3212811996832708|0.2972105647806638|
|4_Lags|20|64807|57|0.0181876415601706|0.0378634212708504|4.681186460910328|-2.0082399167252043|0.3474406507939748|0.3355258189738003|
|no_weather|20|64807|12|0.0027809021309981|-0.1116734923202105|4.717772414091567|-1.1519610195217935|0.1871658598103492|0.1075552824438559|

### Experiment 6c: Does weather data improve classification?
Yes, including the weather data helps a bit, but not as drastically as it does in the regression. Lagged weather data also doesn't matter.

|recipe|n_sites|n_rows|n_feat|loso_auc|lofo_auc|prauc|brier|base|between_rate_r2|macro_auc|
|-|-|-|-|-|-|-|-|-|-|-|
|0_Lags|20|64807|21|0.7561793445986376|0.7323667246839052|0.4527968126560898|0.1590138926823411|0.2153316771336429|-0.7812331983393002|0.8834327738124312|
|1_Lags|20|64807|30|0.7526718088163223|0.7326660991579867|0.4529097160124932|0.160107686300132|0.2153316771336429|-0.7942746928463804|0.8753571367504475|
|2_Lags|20|64807|39|0.7578605316393957|0.7420639243866387|0.4693743481886722|0.1574280289107578|0.2153316771336429|-0.6871497627600684|0.8836210556751647|
|3_Lags|20|64807|48|0.7554664454633215|0.7296330119148076|0.4627973346236435|0.1586526027332993|0.2153316771336429|-0.7535949773474595|0.8811211928644664|
|4_Lags|20|64807|57|0.7583821675073797|0.733585953327355|0.4657674644637612|0.1575754864309926|0.2153316771336429|-0.7634776327046993|0.8904589704312502|
|no_weather|20|64807|12|0.730527804068899|0.7031861043392078|0.3873952475733466|0.1698916732440958|0.2153316771336429|-1.0699301193160946|0.8349317073438671|

### Experiment 7: Which recipes perform best on individual site modeling?

Tested 4 different recipes on 20 sites, each site gets its own model, no cross-site generalization tested for.

* `A_static`: Just the covariates + static features (lat/lon, basin area). This is crop and surplus exponential-decay aggregation across entire basin, the weather time series, and a pure cos/sin signal for a seasonality reference.
* `B_static`: `A_static` + a site's OWN past nitrate. This only makes sense for individual site monitoring, if you wanted to setup a virtual site you obviously wouldn't have access to past nitrate readings because there aren't past nitrate readings.
* `C_static`: `A_static` + "cross-site climatology". Specifically, the rolling average of all sites on 3D, 7D, 14D, 30D windows, the average daily nitrate of all sensors across the state, the day of year, week of year, month of year signals, and 3 copies of every single other sites's nitrate one for 1,3,7 day lags. A big row set, all basically other nitrate data. Exhibits leakage because a sensor's own nitrate today is the target and is included in the daily average across the site.
* `D_static`: `A_static` + toned down version of C protected from leakage. Rolling average of all sites (up to yesterday, so today is not included) and then the average daily nitrate of all sensors EXCEPT this one at 1,2,3,7,14 day lags. 

|recipe|n_sites|median_r2|mean_r2|
|-|-|-|-|
|A_static|20|-0.0248873320155595|-0.641397738975395|
|B_static|20|0.8779130870020047|0.7513493106700608|
|C_static|20|0.5618503174004563|0.4247365245935888|
|D_static|20|0.4546353715630746|0.2854733279150639|

### Experiment 7c:



### Experiment 8:



### Experiment 8c:



### Experiment 9:



### Experiment 9c:



## Regression score table columns (e.g. `_experiment6.csv`)

- **recipe** — The name of the feature recipe being evaluated. Each row is one recipe scored across the same pooled set of sites, so rows are directly comparable.

- **n_sites** — The number of distinct sites pooled into the cross-site model. Every site contributes rows to training and is held out once for scoring.

- **n_rows** — The total number of usable daily observations pooled across all sites after merging and dropping NaNs. This is the actual sample size the model was trained and scored on.

- **n_feat** — The number of feature columns the recipe produced (excluding the target). Higher values mean a richer/larger feature set.

- **loso_r2** — Leave-one-site-out $R^2$: pooled $R^2$ over out-of-fold predictions where each site is held out once and predicted by a model trained on the others. This is the headline cross-site generalization score, row-weighted so big sites dominate.

- **lofo_r2** — Leave-one-family-out $R^2$: the same idea but folds are whole basin families (hydrologically connected sites) instead of single sites. It is stricter than LOSO because it prevents a site leaking through its connected neighbors, so it is the more honest transfer estimate (and is NaN when there are fewer than 2 families).

- **rmse** — Root mean squared error of the LOSO out-of-fold predictions, in the target's units (mg/L nitrate). Unlike $R^2$ it is an absolute error scale, so lower is better and it is comparable across recipes but not across different targets.

- **between_r2** — $R^2$ of per-site *mean* predicted vs per-site *mean* actual, i.e. does the model rank which sites are high- vs low-nitrate. It collapses each site to one point, so it isolates cross-site *level* prediction — and goes negative when the model cannot place an unseen site's baseline.

- **within_r2** — $R^2$ after subtracting each site's own mean from both actual and predicted, i.e. does the model track day-to-day movement *within* a site. It isolates temporal dynamics from level, so it can be healthy even when `between_r2` is negative.

- **macro_r2** — The median of the per-site $R^2$ scores, giving every site equal weight regardless of how many rows it has. It complements the row-weighted `loso_r2` by showing the typical-site performance rather than the big-site-dominated average.


## Classification score table columns (e.g. `_experiment6c.csv`)

- **recipe** — The name of the feature recipe being evaluated. Each row is one recipe scored across the same pooled set of sites, so rows are directly comparable.

- **n_sites** — The number of distinct sites pooled into the cross-site model. Every site contributes rows to training and is held out once for scoring.

- **n_rows** — The total number of usable daily observations pooled across all sites after merging and dropping NaNs. This is the actual sample size the model was trained and scored on.

- **n_feat** — The number of feature columns the recipe produced (excluding the target). Higher values mean a richer/larger feature set.

- **loso_auc** — Leave-one-site-out ROC-AUC: the area under the ROC curve over out-of-fold predictions where each site is held out once and scored by a model trained on the others. This is the headline cross-site discrimination score; 0.5 is chance and 1.0 is perfect ranking of violation vs non-violation days.

- **lofo_auc** — Leave-one-family-out ROC-AUC: the same metric but folds are whole basin families instead of single sites. It is stricter than LOSO because it blocks leakage through hydrologically connected neighbors, and is NaN when there are fewer than 2 families.

- **prauc** — Area under the precision-recall curve (average precision) over the LOSO out-of-fold predictions. Unlike AUC it focuses on the positive (violation) class and is more informative when violations are rare, so compare it against the `base` rate rather than against 0.5.

- **brier** — Brier score: the mean squared error between predicted violation probabilities and the 0/1 outcomes. It rewards calibrated probabilities (not just correct ranking), and lower is better.




#### Exp 6 Feature Importance XGBoost
(Top 12 kept for each column)
|features|0_Lags|1_Lags|2_Lags|3_Lags|4_Lags|no_weather|
|-|-|-|-|-|-|-|
|fuel_moisture_1000h_lag3|nan|nan|0.09712658|0.098019585|0.10781417|nan|
|doy_sin|0.09290817|0.09214246|0.09167559|0.08208793|0.08068488|0.09952179|
|Nonag|0.08088158|0.07536088|0.06607219|0.062715456|0.051392205|0.113766|
|total_kg_N|0.073514685|0.06465008|0.06512599|0.05171442|0.05207532|0.12804028|
|Corn|0.08401893|0.07050376|0.06004118|0.056709748|0.057434462|0.082810886|
|surplus_kgha|0.082539245|0.071847424|0.046560567|0.052611142|0.03974146|0.08989596|
|Soybeans|0.06746831|0.060537916|0.057062685|0.04394568|0.043140613|0.10668262|
|fuel_moisture_1000h_lag1|nan|0.10214138|0.06717348|0.043242693|0.037622146|nan|
|Hay_Pasture|0.059579182|0.057180524|0.054013006|0.041426335|0.038198642|0.08511412|
|Other|0.063617244|0.04797598|0.04568292|0.040519517|0.039081983|0.08713636|
|doy_cos|0.051155467|0.04707795|0.04968781|0.0518251|0.055019338|0.051087994|
|fuel_moisture_1000h_lag7|nan|nan|nan|0.044832967|0.041841693|nan|
|fuel_moisture_1000h|0.09409456|0.0431456|0.01889941|0.020837158|0.012155788|nan|
|Alfalfa|0.04332433|0.03514571|0.033315077|0.03099246|0.027790826|0.056380738|
|Fallow|0.038318302|0.035257336|0.030979414|0.029379675|0.026178163|0.05545817|
|Small_Grains|0.030720394|0.026078057|0.021491129|0.019850662|0.016358066|0.044105086|

#### Exp 6 Feature Importance Col Shuffle
(Top 12 kept for each column)
|features|0_Lags|1_Lags|2_Lags|3_Lags|4_Lags|no_weather|
|-|-|-|-|-|-|-|
|doy_sin|0.3593945514851149|0.3672423104395232|0.3472458529528761|0.3250966717594633|0.3188081025501089|0.3572083014503673|
|Other|0.1142898932609007|0.1173485951148471|0.1598586497736824|0.1102918915189224|0.122428618939746|0.1437259833432333|
|Nonag|0.1365813651764996|0.1492743006060904|0.1075910038725865|0.0990970854986728|0.0932103200755432|0.0778178788898644|
|fuel_moisture_1000h_lag3|nan|nan|0.1505306193201978|0.0826438707567954|0.0817853446661953|nan|
|fuel_moisture_1000h_lag1|nan|0.2030476705473339|0.0312501288994059|0.0259856505954662|0.0197796010503135|nan|
|fuel_moisture_1000h|0.2789770771562246|0.0221536796386901|0.002469785095353|0.0073090973860807|0.0066544601540739|nan|
|doy_cos|0.0594265567783339|0.065714496628656|0.0581529263243914|0.0666380193943778|0.0534924950051047|0.0757660372358925|
|Soybeans|0.0398292602829377|0.0424154358238511|0.0697760285720772|0.0344916429761099|0.0340582246798368|0.0446617101851452|
|total_kg_N|0.0337089856384133|0.0303061621334292|0.0627548215094185|0.0274358564904616|0.0392617494253979|0.0389655898634836|
|fuel_moisture_1000h_lag14|nan|nan|nan|nan|0.0206334522592045|nan|
|fuel_moisture_1000h_lag7|nan|nan|nan|0.0274310016086009|0.0126015978237032|nan|
|Fallow|0.0128955844757915|0.0164425914737909|0.020027272251566|0.0190332058444249|0.0147198201349285|0.0060709236694688|
|surplus_kgha|0.0151874980358521|0.0118461922926512|0.0081806637385055|0.0099261397892212|0.015754808079111|0.0138152349657474|
|Alfalfa|0.036515146286488|-0.0056315011207578|0.0157348824082163|0.0067056032255863|-0.0075882658187286|0.0231909230143183|
|solar_rad_lag1|nan|0.0027323322778949|0.0186729692570697|0.0034210453972124|0.0025183927765076|nan|
|evapotranspiration|-0.0012525799851264|0.0066886185657659|0.0031735716061153|0.0061408004873455|4.412796427414102e-05|nan|
|min_rel_humidity|0.0054771221632862|-0.0042505860529074|-0.0014080323079748|-0.0013504808261905|-0.0030506191804701|nan|
|Hay_Pasture|-0.0008933011618146|0.0054114115841452|0.0030613653247627|0.0152012984914344|-0.0143529386562673|-0.0194464982149448|
|Small_Grains|0.0073489369712802|-0.0057219086366308|-0.0086056721726674|-0.0053414647488487|-0.0007520853715911|-0.0171485095302967|
|Corn|-0.1251217546768561|-0.0966045200170464|-0.2046682761099332|-0.1718306205931269|-0.1859933055769538|-0.0739007835279343|

#### Exp 6c Feature Importance XGBoost
(Top 12 kept for each column)
|features|Unnamed: 0|0_Lags|1_Lags|2_Lags|3_Lags|4_Lags|no_weather|
|-|-|-|-|-|-|-|-|
|Nonag|0|0.09468572|0.0831418|0.07042374|0.066074595|0.05484773|0.12401171|
|doy_sin|1|0.08345747|0.079189755|0.076342806|0.06973873|0.0692484|0.092456296|
|total_kg_N|2|0.08020317|0.067146525|0.06798269|0.053961147|0.051431693|0.1197592|
|surplus_kgha|3|0.07550649|0.06665498|0.04887391|0.048806585|0.04218494|0.08870725|
|Hay_Pasture|4|0.0633678|0.058301084|0.052689828|0.043255396|0.042495996|0.09179037|
|Soybeans|5|0.06263459|0.05284707|0.050637484|0.04079685|0.03701479|0.09208213|
|doy_cos|6|0.05650569|0.0514865|0.053082835|0.04832495|0.04831312|0.063263185|
|Corn|7|0.06593952|0.05095928|0.047032543|0.040772505|0.04102787|0.07452269|
|fuel_moisture_1000h_lag3|8|nan|nan|0.050892036|0.049625553|0.05080244|nan|
|Other|9|0.055047076|0.04402904|0.04047782|0.037816368|0.03307319|0.079205394|
|Alfalfa|10|0.056419414|0.044180907|0.03694437|0.036011457|0.030701166|0.06607531|
|Fallow|11|0.04389419|0.0406595|0.033586018|0.03242729|0.028626952|0.055981863|
|fuel_moisture_1000h_lag1|12|nan|0.052105486|0.03612382|0.030139428|0.027013132|nan|
|Small_Grains|13|0.038376726|0.028276775|0.026881004|0.02282863|0.019342657|0.052144635|
|solar_rad|14|0.043347456|0.035484884|0.021169951|0.02358069|0.029651115|nan|
|fuel_moisture_1000h|15|0.057451647|0.035155915|0.02071123|0.016969522|0.014103323|nan|
|precip_in_1d_lag7|45|nan|nan|nan|0.008148538|0.0073835403|nan|
|solar_rad_lag14|46|nan|nan|nan|nan|0.007666557|nan|
|max_rel_humidity|47|0.009991204|0.008460573|0.007462754|0.006714066|0.005662727|nan|
|vpd_lag3|48|nan|nan|0.008501311|0.007618467|0.0066943034|nan|
|max_rel_humidity_lag1|49|nan|0.009017865|0.0073507004|0.006839791|0.0060401484|nan|
|max_rel_humidity_lag3|50|nan|nan|0.008002107|0.0068779127|0.0061378935|nan|
|min_rel_humidity_lag7|51|nan|nan|nan|0.0072424775|0.006624463|nan|
|vpd_lag14|52|nan|nan|nan|nan|0.0067651286|nan|
|vpd_lag7|53|nan|nan|nan|0.007174758|0.006110198|nan|
|max_rel_humidity_lag7|54|nan|nan|nan|0.0068970197|0.0061919526|nan|
|precip_in_1d_lag14|55|nan|nan|nan|nan|0.00645068|nan|
|max_rel_humidity_lag14|56|nan|nan|nan|nan|0.006302339|nan|

#### Exp 6c Feature Importance Col Shuffle
(Top 12 kept for each column)
|features|Unnamed: 0|0_Lags|1_Lags|2_Lags|3_Lags|4_Lags|no_weather|
|-|-|-|-|-|-|-|-|
|doy_sin|0|0.1688336589477665|0.159522564312282|0.1535127380037519|0.1424574581519601|0.1374875338469586|0.169318253954048|
|doy_cos|1|0.0527785408046917|0.053802014362051|0.0516045781036574|0.0546211818999811|0.0465524591099127|0.0624154956038404|
|Nonag|2|0.0342315971830991|0.0368705387890034|0.0364691727786232|0.0426347651814691|0.0378884099254678|0.0347593893582717|
|fuel_moisture_1000h_lag3|3|nan|nan|0.0325937444474892|0.0218673691917766|0.0221789234835819|nan|
|fuel_moisture_1000h_lag1|4|nan|0.0425495998642698|0.0093310649418224|0.0098620778399053|0.0099552640005746|nan|
|fuel_moisture_1000h|5|0.0659282029927861|0.0084883073142739|0.0032548381465989|0.0027117718067149|0.0029618242301159|nan|
|Other|6|0.01150457069479|0.0117403108907531|0.0118706274968324|0.0091101741163292|0.0101105234658884|0.0165741158973964|
|total_kg_N|7|0.0096498939183611|0.0093377464560854|0.0117575403804749|0.0104972106706912|0.0095594152571277|0.0095353047796944|
|Fallow|8|0.0066688009976835|0.0054825097152609|0.0046849860901052|0.0051109747997889|0.004864241564527|0.0034702783441865|
|fuel_moisture_1000h_lag7|9|nan|nan|nan|0.0037606960748088|0.0040964484892471|nan|
|surplus_kgha|10|0.003614102334877|0.0022414405422432|0.001152092615321|0.0019827984351261|0.0015132598743958|0.005594356020136|
|fuel_moisture_1000h_lag14|11|nan|nan|nan|nan|0.0022024151832431|nan|
|min_temp|12|0.0026617464824238|0.0012927594991567|0.0015981601043327|0.0015362371002448|0.0010664230576667|nan|
|min_temp_lag3|13|nan|nan|0.0014241577210178|0.0010563194792215|0.0016216009655996|nan|
|min_temp_lag1|14|nan|0.0011434403452375|0.0008465976728057|0.0012782458639221|0.0005923151152291|nan|
|evapotranspiration|15|0.0011539419829313|0.0007032201164715|0.0007251014361008|0.0008812114532583|0.0005600919240975|nan|
|min_rel_humidity|16|0.0018062638625035|0.0006171317256742|0.0004267597860263|0.00045925067875|0.0005188337362562|nan|
|min_rel_humidity_lag1|27|nan|0.0010633917983154|0.0002847796766792|0.0003156257370327|0.0002784053020598|nan|
|max_rel_humidity_lag14|45|nan|nan|nan|nan|6.478904493609328e-05|nan|
|vpd_lag7|46|nan|nan|nan|-5.429182013478417e-05|3.4400035265100826e-05|nan|
|max_temp|47|-0.0002978040527082|0.000159655187691|-0.0001699732868882|0.0001910398368321|6.605337086293072e-05|nan|
|max_temp_lag3|48|nan|nan|-0.0003494754325107|8.022620225889732e-05|5.043776444054514e-05|nan|
|Small_Grains|49|0.0005386543596665|-0.0013890081410942|-0.0005377571424831|0.0005255863178056|0.000438802428419|-0.0002853140017638|
|max_temp_lag14|50|nan|nan|nan|nan|-0.000294282163121|nan|
|max_temp_lag1|51|nan|-0.0006213337735231|-0.0004759799631027|-0.0001549869692342|-1.2352644420650092e-05|nan|
|evapotranspiration_lag14|52|nan|nan|nan|nan|-0.0003255811198371|nan|
|Soybeans|53|-0.0008558160628797|-0.000374766873608|-0.0013778313797386|0.0001334133022011|-0.0018662926054308|-0.0014069203606807|
|Hay_Pasture|54|-0.0059645225876572|-0.0071963035505977|-0.0069425973028914|-0.0090984813003123|-0.0080962381807597|-0.0018017697371701|
|Alfalfa|55|-0.0070879422448348|-0.008300154414145|-0.0080094396130849|-0.0072444451933062|-0.0088042905196391|-0.0046613022565605|
|Corn|56|-0.0080615977543277|-0.0085043580044253|-0.0078788813162411|-0.0085922477976384|-0.0089876485082905|-0.0105465172553859|


#### Exp 7 Feature Importance XGBoost
(Top 12 kept for each column)
|features|A_static|A_static_pct|B_static|B_static_pct|C_static|C_static_pct|D_static|D_static_pct|
|-|-|-|-|-|-|-|-|-|
|USGS-05482300_lag2|nan|nan|0.28880247|2.0698202|nan|nan|nan|nan|
|USGS-05482500_lag2|nan|nan|0.28468436|2.040306|nan|nan|nan|nan|
|USGS-05464420_lag2|nan|nan|0.28327346|2.0301943|nan|nan|nan|nan|
|USGS-05484500_lag2|nan|nan|0.27970624|2.0046284|nan|nan|nan|nan|
|USGS-05484500_lag1|nan|nan|0.40633783|2.912185|0.016541358|1.6290021|nan|nan|
|USGS-05412500_lag1|nan|nan|0.35250717|2.5263858|0.00757526|0.7460158|nan|nan|
|USGS-05482500_lag1|nan|nan|0.3345037|2.3973567|0.010669898|1.0507774|nan|nan|
|WQS0024_lag1|nan|nan|0.331601|2.3765533|0.0029850875|0.29397306|nan|nan|
|USGS-05464420_lag1|nan|nan|0.31589395|2.2639823|0.016140407|1.5895163|nan|nan|
|USGS-05482300_lag1|nan|nan|0.3135966|2.2475176|0.014191818|1.397618|nan|nan|
|USGS-06817000_lag1|nan|nan|0.31939977|2.2891083|0.0060694492|0.5977227|nan|nan|
|WQS0001_lag1|nan|nan|0.27516732|1.9720984|0.004801431|0.4728476|nan|nan|
|nroll_3|nan|nan|nan|nan|0.081672534|8.043157|0.18294498|18.294498|
|USGS-05482300_lag3|nan|nan|0.24827269|1.7793471|0.013154388|1.2954515|nan|nan|
|USGS-05465500_lag1|nan|nan|0.23360078|1.6741949|0.0120652|1.1881875|nan|nan|
|WQS0002_lag1|nan|nan|0.19481342|1.3962095|0.009229467|0.9089231|nan|nan|
|USGS-05484500_lag3|nan|nan|0.17305169|1.2402452|0.015259736|1.5027874|nan|nan|
|rest_of_state_nitrate_lag1|nan|nan|nan|nan|nan|nan|0.092237934|9.223793|
|nroll_7|nan|nan|nan|nan|0.057946116|5.7065654|0.10000422|10.000422|
|rest_of_state_nitrate_lag2|nan|nan|nan|nan|nan|nan|0.0747907|7.47907|
|ncal_d|nan|nan|nan|nan|0.068880156|6.783356|nan|nan|
|rest_of_state_nitrate_lag3|nan|nan|nan|nan|nan|nan|0.047874127|4.7874126|
|doy_sin|0.10643844|10.643846|0.022231292|0.1593296|0.0043683043|0.43019304|0.026104242|2.610424|
|surplus_kgha|0.1137893|11.378931|0.010402508|0.074553795|0.0029444548|0.28997153|0.025335114|2.5335114|
|nroll_14|nan|nan|nan|nan|0.012756688|1.2562857|0.058183413|5.8183413|
|total_kg_N|0.085173935|8.517394|0.011666088|0.08360976|0.00373976|0.3682936|0.02529652|2.529652|
|Alfalfa|0.06666087|6.6660876|0.020604353|0.14766946|0.005691094|0.56046206|0.030842016|3.0842016|
|Hay_Pasture|0.06058821|6.0588217|0.019822653|0.1420671|0.0026443847|0.26042047|0.019541705|1.9541705|
|fuel_moisture_1000h|0.058760464|5.8760467|0.014141674|0.10135205|0.0047568236|0.46845463|0.020451318|2.045132|
|Other|0.052164875|5.216488|0.017233443|0.123510465|0.0029946275|0.29491255|0.024355661|2.4355662|
|Soybeans|0.05095636|5.0956364|0.018138194|0.12999474|0.003925384|0.386574|0.023633458|2.3633459|
|Corn|0.04943018|4.9430184|0.019208152|0.13766302|0.0042997836|0.42344508|0.022560341|2.2560341|
|doy_cos|0.05420669|5.4206696|0.017587088|0.12604502|0.0034374776|0.3385247|0.01942717|1.942717|
|Small_Grains|0.050285827|5.028583|0.015177026|0.10877232|0.0034273681|0.3375291|0.024833808|2.4833808|
|Fallow|0.046109453|4.6109457|0.014724992|0.10553264|0.0034223176|0.33703172|0.02322285|2.3222852|

#### Exp 7 Feature Importance Col Shuffle
(Top 12 kept for each column)
|features|A_static|A_static_pct|B_static|B_static_pct|C_static|C_static_pct|D_static|D_static_pct|
|-|-|-|-|-|-|-|-|-|
|WQS0001_lag1|nan|nan|1.245127261395803|7.292844877228724|0.0035867843718655|1.960472108593224|nan|nan|
|USGS-05464420_lag1|nan|nan|1.089126108620167|6.379129273101474|0.025047694616573|13.690621344714751|nan|nan|
|WQS0010_lag1|nan|nan|1.08803273436815|6.372725261993584|0.0017668238310147|0.9657142672617912|nan|nan|
|USGS-05412500_lag1|nan|nan|1.0331691821992297|6.05138351020096|0.0013989184234647|0.7646237596305275|nan|nan|
|USGS-06817000_lag1|nan|nan|0.9875503508424724|5.78418908688367|0.0018547325964547|1.01376362425626|nan|nan|
|USGS-05484500_lag1|nan|nan|0.9431087519693688|5.52388984139591|-0.0011931597547874|-0.6521597558818341|nan|nan|
|USGS-05482300_lag1|nan|nan|0.9097913416621471|5.328746170050087|0.0144108839404617|7.8767311040654|nan|nan|
|WQS0024_lag1|nan|nan|0.8286587611883542|4.853543881714226|0.0008142169876404|0.4450364251424774|nan|nan|
|WQS0005_lag1|nan|nan|0.810304064547495|4.746038440687804|0.0065406665031114|3.575011185936015|nan|nan|
|WQS0014_lag1|nan|nan|0.802661209633326|4.701273413821621|0.0048717401766727|2.66280594161065|nan|nan|
|USGS-05482500_lag1|nan|nan|0.7678331952245093|4.497281971066882|0.0333935513106001|18.2523171632024|nan|nan|
|USGS-05465500_lag1|nan|nan|0.7945937806181195|4.654021610580239|0.0050397485384648|2.754636303616102|nan|nan|
|WQS0003_lag1|nan|nan|0.7205165299757833|4.220143151232605|0.0061112163669368|3.3402814317289424|nan|nan|
|WQS0002_lag1|nan|nan|0.60693208464189|3.554866784736898|0.0152470810474615|8.333781468848507|nan|nan|
|fuel_moisture_1000h|0.1705331759998874|52.33060335368972|0.0068917369145382|0.0403656476013992|0.014690861343989|8.029761739237424|0.0525056793170013|26.819180364708135|
|nroll_3|nan|nan|nan|nan|0.000409679274437|0.2239233552210485|0.1152965778552888|58.891909544980685|
|doy_sin|0.188608468553879|57.87727167555021|0.0069112675562607|0.0404800406798026|0.0031318755060854|1.7118270686769077|0.0111387380405307|5.689514514051812|
|ncal_d|nan|nan|nan|nan|0.0193624312435368|10.583158191849767|nan|nan|
|nroll_14|nan|nan|nan|nan|0.0030652310261132|1.6754003893364695|0.018849121977091|9.627872805300534|
|WQS0048_lag1|nan|nan|nan|nan|0.0094651971321339|5.173507257765147|nan|nan|
|rest_of_state_nitrate_lag1|nan|nan|nan|nan|nan|nan|0.0082125521194645|4.194858906913835|
|rest_of_state_nitrate_lag7|nan|nan|nan|nan|nan|nan|0.0061700843011884|3.151594377818978|
|USGS-05464420_lag3|nan|nan|0.0067571637832787|0.0395774382340946|0.0045682563593246|2.4969271215769937|nan|nan|
|rest_of_state_nitrate_lag3|nan|nan|nan|nan|nan|nan|0.0049290118274456|2.517671591713728|
|nroll_30|nan|nan|nan|nan|0.0008836279894647|0.4829752357867215|0.0069104737341635|3.5297751384791667|
|Alfalfa|0.0023167399094191|0.7109255812696185|0.0021177403423855|0.0124038339582687|0.0003979612545044|0.2175184953621675|0.0077187394365179|3.9426261659494513|
|Hay_Pasture|0.0114818631574607|3.5233779182932485|0.0001296960832582|0.0007596439703088|-0.0001087772682032|-0.0594557069095385|-0.0028450962517266|-1.4532361169795498|
|min_rel_humidity|0.007851767576946|2.409429908786762|-0.0007246659630276|-0.0042444468288677|-2.8790308312518084e-05|-0.0157362669713914|0.0005862869931265|0.2994673494120851|
|precip_in_1d|0.0039203145543328|1.203005953308622|0.000979417025108|0.0057365513194394|-0.0001239714947167|-0.067760599036641|-0.0006771247251842|-0.3458660162166218|
|min_temp|0.0018066185695847|0.5543873749024203|-0.0008992275614173|-0.005266872967984|0.0004541389715644|0.2482242295246524|-0.0009415190251129|-0.4809149958589548|
|total_kg_N|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|surplus_kgha|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|max_dist_to_sensor|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|lon|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|log_basin_area|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|max_rel_humidity|-0.0015056666105395|-0.4620358573459781|-0.0013585701878117|-0.0079572923521341|-7.587953254804659e-05|-0.0414743937049561|0.001588683590014|0.811477773400467|
|evapotranspiration|-0.0049080909360603|-1.506119606890939|0.0003361905671703|0.0019691044695401|0.0004437087609684|0.2425232631881675|0.0014542360398066|0.7428038100214686|
|nroll_7|nan|nan|nan|nan|-0.0055264808040108|-3.020675443978202|0.0025586890384442|1.3069432433191714|
