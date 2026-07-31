# For CLF

An ablation experiment for the long_run features, means of surplus and corn + standard deviations to try and keep some info. Seems like means do best. Big boost to between_r2, doesn't move the auc scores much.

#### Results
|recipe|n_sites|n_families|n_rows|n_feat|loso_auc|lofo_prauc|lofo_auc|lofo_prauc_lift|lofo_recall_at_beta|lofo_fdr_at_beta|lofo_brier|base|lofo_between_rate_r2|lofo_macro_auc|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|base|79|20|157472|49|0.8732223721104649|0.6930565564805413|0.8581455108180107|2.703619344070746|0.870290088438576|0.5337748168595392|0.1273560313078933|0.256343984962406|0.4139282917388253|0.8968683027642028|
|mean|79|20|157472|76|0.8749133387477057|0.7123491136891882|0.870475576712183|2.7788797688920117|0.8946168900339386|0.5372086168672228|0.1218330331819681|0.256343984962406|0.4883379350912625|0.9027392176274848|
|mean_sd|79|20|157472|103|0.8687649780642309|0.7145163914883653|0.8644769238606563|2.787334337465154|0.8902816657170461|0.5537481529310966|0.1225332961604722|0.256343984962406|0.4777211592768928|0.8996472877753767|

#### Exp experiments/test_results/_experiment20c Feature Importance XGBoost
(Top 12 kept for each column)
|features|base|mean|mean_sd|
|-|-|-|-|
|pct_corn_mean_b2|nan|0.06366623|0.06456449|
|rest_of_state_nitrate_lag1|0.090805784|0.051027633|0.04394559|
|rest_of_state_nitrate_lag3|0.077931955|0.048941888|0.035486072|
|pct_corn_mean_b1|nan|0.0452115|0.04559186|
|pct_corn_b2|0.06000092|0.034585536|0.01831205|
|pct_hay_pasture_mean_b2|nan|0.036051996|0.038533904|
|pct_hay_pasture_sd_b2|nan|nan|0.027647968|
|pct_nonag_mean_b0|nan|0.029563788|0.024789663|
|surplus_kgha_norm_mean_b2|nan|0.029501736|0.023327932|
|pct_small_grains_mean_b2|nan|0.026366511|0.021319106|
|max_dist_to_sensor|0.032434452|0.022073988|0.016055059|
|pct_nonag_b0|0.040032648|0.017394217|0.012291204|
|mean_dist_to_sensor|0.03404101|0.017538065|0.0142658185|
|pct_corn_b1|0.042735867|0.011774229|0.0069896327|
|log_basin_area|0.027482724|0.016038429|0.01334652|
|surplus_kgha_norm_mean_b1|nan|0.019798566|0.01588223|
|pct_hay_pasture_b2|0.03231956|0.0099690305|0.0071402295|
|pct_nonag_mean_b1|nan|0.01906895|0.013752058|
|pct_soybeans_sd_b2|nan|nan|0.016139302|
|pct_nonag_b1|0.027128046|0.00856779|0.0061657527|
|pct_nonag_b2|0.024359008|0.008960938|0.0067212493|
|pct_corn_b0|0.025776137|0.007391625|0.004682431|

# For REG

An ablation experiment for the long_run features, means of surplus and corn + standard deviations to try and keep some info. Seems like means do best. Notice the huge boost to between_r2.

#### Results
|recipe|n_sites|n_families|n_rows|n_feat|loso_r2|lofo_r2|lofo_rmse|lofo_between_r2|lofo_within_r2|lofo_macro_r2|
|-|-|-|-|-|-|-|-|-|-|-|
|base|79|20|157472|50|0.4481846247311337|0.3781125317721828|4.382089952039202|0.2330422859880373|0.4085860558696458|0.3113026054754745|
|mean|79|20|157472|77|0.4743985648140491|0.432937889668589|4.184472329715198|0.4498580114932978|0.4210109675748327|0.3029537779223386|
|mean_sd|79|20|157472|104|0.4713715124717096|0.4352820007167804|4.175814525238272|0.4395942004910035|0.4252942608985148|0.2956652700615723|

#### Exp experiments/test_results/_experiment20 Feature Importance XGBoost
(Top 12 kept for each column)
|features|base|mean|mean_sd|
|-|-|-|-|
|pct_corn_mean_b0|nan|0.11009759|0.089809336|
|pct_corn_mean_b1|nan|0.08695097|0.0688629|
|rest_of_state_nitrate_lag1|0.11949761|0.058023863|0.050932907|
|roll_n_avg_except_this7d|0.07208302|0.046633787|0.030180966|
|pct_hay_pasture_sd_b2|nan|nan|0.045302525|
|pct_corn_b1|0.07760405|0.023378639|0.018220384|
|rest_of_state_nitrate_lag3|0.05439335|0.018947804|0.013212016|
|pct_hay_pasture_sd_b1|nan|nan|0.027990121|
|pct_nonag_sd_b0|nan|nan|0.027601937|
|pct_hay_pasture_b2|0.047907908|0.017071497|0.011349059|
|surplus_kgha_norm_mean_b1|nan|0.0300143|0.020475052|
|pct_hay_pasture_mean_b2|nan|0.022960447|0.025788972|
|surplus_kgha_norm_mean_b0|nan|0.026371304|0.021085767|
|pct_soybeans_mean_b2|nan|0.022918874|0.022801746|
|pct_hay_pasture_mean_b1|nan|0.028002268|0.017652929|
|pct_nonag_mean_b0|nan|0.024662692|0.019447457|
|pct_small_grains_mean_b2|nan|0.025558341|0.016800942|
|pct_corn_b0|0.043385424|0.007749227|0.0055933977|
|pct_corn_b2|0.03888582|0.008816737|0.0073339837|
|log_basin_area|0.025983766|0.012381864|0.010423905|
|lat|0.0303093|0.010057504|0.0075962273|
|max_dist_to_sensor|0.02713331|0.009347035|0.0078441175|
|mean_dist_to_sensor|0.024538716|0.012509371|0.0072726384|
|pct_nonag_b0|0.02953215|0.008617173|0.004609096|