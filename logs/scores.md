# CLF Model 22: light5_0728_CLF

**Recipe:** light_CLF  
**True Lofo:** True (max_holdout_pct = 0.1) 
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b2, mean_dist_to_sensor, max_dist_to_sensor, pct_corn_b1, rest_of_state_nitrate_lag3, pct_hay_pasture_b2, pct_nonag_b0, lat, pct_nonag_b2, lon, log_basin_area, pct_nonag_b1, pct_corn_b0, surplus_kgha_norm_b2, pct_alfalfa_b0, Soybeans_expT2000, Corn_expT2000, pct_alfalfa_b2, Nonag_expT2000, pct_small_grains_b0, pct_other_b2, pct_soybeans_b0, pct_other_b0, pct_fallow_b1, doy_sin, Hay_Pasture_expT2000, surplus_kgha_norm_b0, pct_soybeans_b2, pct_fallow_b0, surplus_kgha_expT2000, pct_small_grains_b2, pct_fallow_b2, Fallow_expT2000, pct_hay_pasture_b0, Other_expT2000, pct_hay_pasture_b1, pct_soybeans_b1, doy_cos, pct_alfalfa_b1, Small_Grains_expT2000, pct_small_grains_b1, fuel_moisture_1000h_b2, surplus_kgha_norm_b1, pct_other_b1, doy_sin2, fuel_moisture_1000h_b1, doy_cos2, fuel_moisture_1000h_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 49 | 0.8750 | 0.8449 | 0.7413 | 0.6264 | 0.6792 | 0.5960 | 2.8739 | 0.7525 | 0.5662 | 0.6549 | 2.8779 | 0.7041 | 0.4748 | 0.5508 | 0.1181 | -3.2497 | 0.2580 | 0.4036 | 0.8996 |

**Beta Table:**


---

# REG Model 21: light5_0728_REG

**Recipe:** light_REG  
**True Lofo:** True (max_holdout_pct = 0.1) 
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b1, pct_hay_pasture_b2, roll_n_avg_except_this7d, mean_dist_to_sensor, pct_corn_b0, max_dist_to_sensor, lat, pct_corn_b2, log_basin_area, pct_nonag_b0, pct_nonag_b1, lon, fuel_moisture_1000h_b2, surplus_kgha_norm_b2, rest_of_state_nitrate_lag3, Corn_expT2000, pct_alfalfa_b2, pct_hay_pasture_b1, Nonag_expT2000, Soybeans_expT2000, Hay_Pasture_expT2000, Other_expT2000, pct_soybeans_b1, surplus_kgha_expT2000, pct_alfalfa_b0, pct_other_b1, surplus_kgha_norm_b1, pct_other_b0, pct_hay_pasture_b0, doy_sin, Fallow_expT2000, pct_fallow_b0, pct_soybeans_b0, fuel_moisture_1000h_b1, pct_soybeans_b2, pct_alfalfa_b1, pct_small_grains_b2, pct_small_grains_b0, pct_fallow_b2, pct_nonag_b2, Small_Grains_expT2000, pct_fallow_b1, pct_other_b2, fuel_moisture_1000h_b0, pct_small_grains_b1, doy_sin2, doy_cos, surplus_kgha_norm_b0, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 50 | 0.4020 | 0.2014 | 4.3068 | -7.1178 | 0.6771 | 0.2278 | 0.3968 | 0.2727 |

**Beta Table:**


---

# CLF Model 20: light4_0728_CLF

**Recipe:** light_CLF  
**True Lofo:** True (max_holdout_pct = 0.5) 
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b2, mean_dist_to_sensor, max_dist_to_sensor, pct_corn_b1, rest_of_state_nitrate_lag3, pct_hay_pasture_b2, pct_nonag_b0, lat, pct_nonag_b2, lon, log_basin_area, pct_nonag_b1, pct_corn_b0, surplus_kgha_norm_b2, pct_alfalfa_b0, Soybeans_expT2000, Corn_expT2000, pct_alfalfa_b2, Nonag_expT2000, pct_small_grains_b0, pct_other_b2, pct_soybeans_b0, pct_other_b0, pct_fallow_b1, doy_sin, Hay_Pasture_expT2000, surplus_kgha_norm_b0, pct_soybeans_b2, pct_fallow_b0, surplus_kgha_expT2000, pct_small_grains_b2, pct_fallow_b2, Fallow_expT2000, pct_hay_pasture_b0, Other_expT2000, pct_hay_pasture_b1, pct_soybeans_b1, doy_cos, pct_alfalfa_b1, Small_Grains_expT2000, pct_small_grains_b1, fuel_moisture_1000h_b2, surplus_kgha_norm_b1, pct_other_b1, doy_sin2, fuel_moisture_1000h_b1, doy_cos2, fuel_moisture_1000h_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 49 | 0.8750 | 0.8538 | 0.7413 | 0.6794 | 0.6792 | 0.6470 | 2.8739 | 0.7525 | 0.5662 | 0.6549 | 2.6338 | 0.7457 | 0.5167 | 0.5800 | 0.1181 | -3.2497 | 0.2580 | 0.4036 | 0.8996 |

**Beta Table:**


---

# REG Model 19: light4_0728_REG

**Recipe:** light_REG  
**True Lofo:** True (max_holdout_pct = 0.5) 
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b1, pct_hay_pasture_b2, roll_n_avg_except_this7d, mean_dist_to_sensor, pct_corn_b0, max_dist_to_sensor, lat, pct_corn_b2, log_basin_area, pct_nonag_b0, pct_nonag_b1, lon, fuel_moisture_1000h_b2, surplus_kgha_norm_b2, rest_of_state_nitrate_lag3, Corn_expT2000, pct_alfalfa_b2, pct_hay_pasture_b1, Nonag_expT2000, Soybeans_expT2000, Hay_Pasture_expT2000, Other_expT2000, pct_soybeans_b1, surplus_kgha_expT2000, pct_alfalfa_b0, pct_other_b1, surplus_kgha_norm_b1, pct_other_b0, pct_hay_pasture_b0, doy_sin, Fallow_expT2000, pct_fallow_b0, pct_soybeans_b0, fuel_moisture_1000h_b1, pct_soybeans_b2, pct_alfalfa_b1, pct_small_grains_b2, pct_small_grains_b0, pct_fallow_b2, pct_nonag_b2, Small_Grains_expT2000, pct_fallow_b1, pct_other_b2, fuel_moisture_1000h_b0, pct_small_grains_b1, doy_sin2, doy_cos, surplus_kgha_norm_b0, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 50 | 0.4020 | 0.3231 | 4.3068 | -7.1178 | 0.6771 | 0.2278 | 0.3968 | 0.2727 |

**Beta Table:**


---

# CLF Model 18: light3_0727_CLF

**Recipe:** light_CLF  
**True Lofo:** ?  
**Notes:** This performed worse than light2, moved back to light2 after this (bucketed weather, fewer lagged rest_of_site_days, kept log_basin_area) but we didn't add back Alfalfa_expT, kept that removed. Light3 did add substantial between_rate_r2 gain, however. Note that the REG comparison favors light2 even more. 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, pct_corn_b2, mean_dist_to_sensor, rest_of_state_nitrate_lag3, max_dist_to_sensor, pct_corn_b1, pct_hay_pasture_b2, lat, pct_nonag_b2, pct_nonag_b0, lon, pct_nonag_b1, pct_corn_b0, surplus_kgha_norm_b2, pct_alfalfa_b0, Corn_expT2000, Soybeans_expT2000, pct_alfalfa_b2, Nonag_expT2000, pct_small_grains_b0, pct_other_b2, pct_soybeans_b0, pct_other_b0, pct_fallow_b1, surplus_kgha_expT2000, Hay_Pasture_expT2000, surplus_kgha_norm_b0, pct_soybeans_b2, pct_fallow_b2, Fallow_expT2000, pct_hay_pasture_b0, pct_fallow_b0, Other_expT2000, pct_soybeans_b1, pct_hay_pasture_b1, doy_sin, pct_alfalfa_b1, pct_small_grains_b2, Small_Grains_expT2000, doy_cos, rest_of_state_nitrate_lag5, pct_small_grains_b1, fuel_moisture_1000h, pct_other_b1, surplus_kgha_norm_b1, doy_sin2, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 48 | 0.8748 | 0.8510 | 0.7420 | 0.6701 | 0.6799 | 0.6445 | 2.8765 | 0.7514 | 0.5691 | 0.6580 | 2.5977 | 0.7418 | 0.5104 | 0.5690 | 0.1178 | -3.2377 | 0.2580 | 0.4060 | 0.9019 |

**Beta Table:**


---

# REG Model 17: light3_0727_REG

**Recipe:** light_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b1, pct_hay_pasture_b2, roll_n_avg_except_this7d, mean_dist_to_sensor, pct_corn_b0, pct_corn_b2, lat, pct_nonag_b0, pct_hay_pasture_b1, pct_nonag_b1, pct_alfalfa_b2, lon, surplus_kgha_norm_b2, Nonag_expT2000, Corn_expT2000, Soybeans_expT2000, Other_expT2000, Hay_Pasture_expT2000, pct_other_b1, pct_soybeans_b1, doy_sin, fuel_moisture_1000h, pct_alfalfa_b0, surplus_kgha_norm_b1, pct_other_b0, Fallow_expT2000, pct_hay_pasture_b0, surplus_kgha_expT2000, pct_fallow_b0, pct_soybeans_b2, pct_soybeans_b0, pct_small_grains_b2, pct_fallow_b2, pct_alfalfa_b1, pct_small_grains_b0, pct_nonag_b2, Small_Grains_expT2000, pct_fallow_b1, pct_other_b2, pct_small_grains_b1, doy_sin2, doy_cos, surplus_kgha_norm_b0, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 45 | 0.4196 | 0.3251 | 4.2429 | -6.8686 | 0.6923 | 0.2307 | 0.4019 | 0.2533 |

**Beta Table:**


---

# CLF Model 16: light2_0727_CLF

**Recipe:** light_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b2, pct_hay_pasture_b2, rest_of_state_nitrate_lag3, pct_nonag_b1, pct_alfalfa_b2, mean_dist_to_sensor, max_dist_to_sensor, pct_corn_b1, lat, pct_nonag_b2, lon, pct_alfalfa_b0, pct_nonag_b0, surplus_kgha_norm_b2, log_basin_area, Corn_expT, pct_corn_b0, Soybeans_expT, pct_small_grains_b0, pct_hay_pasture_b1, pct_other_b2, pct_alfalfa_b1, doy_sin, pct_fallow_b1, pct_fallow_b0, pct_small_grains_b2, Hay_Pasture_expT, surplus_kgha_norm_b0, pct_other_b1, pct_hay_pasture_b0, surplus_kgha_expT, pct_soybeans_b1, pct_soybeans_b0, Other_expT, Fallow_expT, pct_soybeans_b2, Nonag_expT, Alfalfa_expT, pct_small_grains_b1, doy_cos, pct_other_b0, pct_fallow_b2, fuel_moisture_1000h_b2, Small_Grains_expT, surplus_kgha_norm_b1, doy_sin2, doy_cos2, fuel_moisture_1000h_b1, fuel_moisture_1000h_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 50 | 0.8666 | 0.8667 | 0.7047 | 0.7004 | 0.6678 | 0.6675 | 2.7320 | 0.7492 | 0.5519 | 0.6361 | 2.7151 | 0.7537 | 0.5489 | 0.6299 | 0.1243 | -3.4762 | 0.2580 | 0.2285 | 0.9038 |

**Beta Table:**


---

# CLF Model 15: later_0727_CLF

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, pct_corn_b2, mean_dist_to_sensor, pct_corn_b1, max_dist_to_sensor, rest_of_state_nitrate_lag3, pct_hay_pasture_b2, pct_nonag_b0, lat, pct_nonag_b2, lon, log_basin_area, surplus_kgha_norm_b2, pct_nonag_b1, pct_corn_b0, Soybeans_expT, pct_alfalfa_b0, Corn_expT, pct_alfalfa_b2, pct_small_grains_b0, Nonag_expT, pct_other_b2, Alfalfa_expT, rest_of_state_nitrate_lag5, pct_soybeans_b0, pct_other_b0, pct_fallow_b1, Hay_Pasture_expT, surplus_kgha_norm_b0, pct_fallow_b2, pct_soybeans_b2, surplus_kgha_expT, doy_sin, pct_fallow_b0, Fallow_expT, Other_expT, pct_soybeans_b1, pct_small_grains_b2, pct_hay_pasture_b0, pct_hay_pasture_b1, doy_cos, pct_alfalfa_b1, solar_rad_b1, Small_Grains_expT, fuel_moisture_1000h_b2, pct_other_b1, pct_small_grains_b1, surplus_kgha_norm_b1, doy_sin2, min_temp_b1, solar_rad_b2, fuel_moisture_1000h_b1, evapotranspiration_b1, min_temp_b0, fuel_moisture_1000h_b0, doy_cos2, solar_rad_b0, evapotranspiration_b0, min_temp_b2, max_temp_b0, max_temp_b2, evapotranspiration_b2, max_temp_b1, precip_in_1d_b0, vpd_b2, max_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, vpd_b1, min_rel_humidity_b2, precip_in_1d_b2, max_rel_humidity_b1, min_rel_humidity_b1, max_rel_humidity_b0, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 76 | 0.8741 | 0.8525 | 0.7415 | 0.6772 | 0.6766 | 0.6465 | 2.8744 | 0.7521 | 0.5647 | 0.6532 | 2.6254 | 0.7442 | 0.5133 | 0.5663 | 0.1180 | -3.2483 | 0.2580 | 0.4070 | 0.8991 |

**Beta Table:**


---

# REG Model 14: light2_0727_REG

**Recipe:** light_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_hay_pasture_b2, pct_corn_b2, roll_n_avg_except_this7d, mean_dist_to_sensor, lat, max_dist_to_sensor, pct_nonag_b1, pct_corn_b1, rest_of_state_nitrate_lag3, log_basin_area, Corn_expT, pct_alfalfa_b2, pct_nonag_b2, pct_other_b0, Soybeans_expT, pct_small_grains_b2, pct_nonag_b0, lon, Other_expT, pct_alfalfa_b0, surplus_kgha_norm_b2, pct_soybeans_b1, Nonag_expT, pct_alfalfa_b1, fuel_moisture_1000h_b2, pct_small_grains_b0, pct_hay_pasture_b1, doy_sin, pct_soybeans_b2, Hay_Pasture_expT, surplus_kgha_expT, pct_fallow_b1, Alfalfa_expT, pct_hay_pasture_b0, Fallow_expT, pct_fallow_b0, pct_soybeans_b0, pct_other_b2, surplus_kgha_norm_b0, pct_corn_b0, pct_fallow_b2, pct_other_b1, fuel_moisture_1000h_b1, Small_Grains_expT, pct_small_grains_b1, surplus_kgha_norm_b1, doy_cos, doy_sin2, fuel_moisture_1000h_b0, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 51 | 0.4225 | 0.3706 | 4.2322 | -6.8280 | 0.6841 | 0.3037 | 0.4083 | 0.2817 |

**Beta Table:**


---

# REG Model 13: later_0727_REG

**Recipe:** recipe_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, roll_n_avg_except_this7d, pct_hay_pasture_b1, pct_corn_b0, mean_dist_to_sensor, rest_of_state_nitrate_lag2, max_dist_to_sensor, pct_nonag_b0, lat, pct_corn_b1, fuel_moisture_1000h_b1, log_basin_area, lon, pct_alfalfa_b1, surplus_kgha_norm_b1, Corn_expT, Nonag_expT, Other_expT, pct_hay_pasture_b0, Soybeans_expT, pct_other_b0, Hay_Pasture_expT, rest_of_state_nitrate_lag3, pct_soybeans_b0, doy_sin, Alfalfa_expT, pct_soybeans_b1, Fallow_expT, surplus_kgha_expT, surplus_kgha_norm_b0, fuel_moisture_1000h_b0, Small_Grains_expT, pct_small_grains_b1, pct_fallow_b0, pct_fallow_b1, pct_alfalfa_b0, pct_nonag_b1, rest_of_state_nitrate_lag5, pct_other_b1, pct_small_grains_b0, doy_sin2, doy_cos, evapotranspiration_b1, min_temp_b1, solar_rad_b1, min_temp_b0, doy_cos2, max_temp_b1, solar_rad_b0, max_temp_b0, evapotranspiration_b0, vpd_b1, min_rel_humidity_b1, min_rel_humidity_b0, precip_in_1d_b1, vpd_b0, max_rel_humidity_b1, precip_in_1d_b0, max_rel_humidity_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 59 | 0.4001 | 0.3479 | 4.3135 | -7.1449 | 0.6771 | 0.1659 | 0.4207 | 0.2585 |

**Beta Table:**


---

# CLF Model 12: light_0727_CLF

**Recipe:** light_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, mean_dist_to_sensor, pct_corn, rest_of_state_nitrate_lag3, max_dist_to_sensor, pct_nonag, pct_hay_pasture, lat, lon, log_basin_area, Soybeans_expT, Corn_expT, surplus_kgha_norm, pct_other, Nonag_expT, pct_alfalfa, pct_fallow, Alfalfa_expT, Hay_Pasture_expT, Fallow_expT, Other_expT, doy_sin, pct_soybeans, surplus_kgha_expT, doy_cos, pct_small_grains, Small_Grains_expT, fuel_moisture_1000h, doy_sin2, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 30 | 0.8718 | 0.8429 | 0.7341 | 0.6519 | 0.6678 | 0.6313 | 2.8459 | 0.7560 | 0.5505 | 0.6345 | 2.5274 | 0.7384 | 0.4905 | 0.5455 | 0.1210 | -3.3558 | 0.2580 | 0.3255 | 0.9009 |

**Beta Table:**


---

# REG Model 11: light_0727_REG

**Recipe:** light_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, max_dist_to_sensor, mean_dist_to_sensor, roll_n_avg_except_this7d, pct_corn, pct_hay_pasture, rest_of_state_nitrate_lag3, lat, pct_nonag, log_basin_area, Corn_expT, Nonag_expT, lon, Other_expT, Soybeans_expT, surplus_kgha_norm, fuel_moisture_1000h, pct_other, pct_soybeans, Alfalfa_expT, doy_sin, pct_fallow, Hay_Pasture_expT, pct_small_grains, pct_alfalfa, Fallow_expT, surplus_kgha_expT, Small_Grains_expT, doy_sin2, doy_cos, doy_cos2  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | 20 | 158215 | 31 | 0.3995 | 0.3354 | 4.3157 | -7.1489 | 0.6817 | 0.1363 | 0.4266 | 0.2145 |

**Beta Table:**


---

# CLF Model 10: date0727_CLF

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b2, rest_of_state_nitrate_lag2, mean_dist_to_sensor, pct_corn_b1, max_dist_to_sensor, rest_of_state_nitrate_lag3, pct_nonag_b0, pct_hay_pasture_b2, lat, pct_nonag_b2, lon, log_basin_area, pct_nonag_b1, pct_alfalfa_b2, surplus_kgha_norm_b2, pct_alfalfa_b0, pct_corn_b0, Corn_expT, Soybeans_expT, pct_small_grains_b0, Nonag_expT, rest_of_state_nitrate_lag5, Alfalfa_expT, pct_other_b2, pct_fallow_b1, Hay_Pasture_expT, pct_other_b0, pct_soybeans_b0, Other_expT, surplus_kgha_norm_b0, pct_soybeans_b2, Fallow_expT, pct_fallow_b2, pct_hay_pasture_b0, pct_fallow_b0, surplus_kgha_expT, doy_sin, pct_hay_pasture_b1, pct_small_grains_b2, doy_cos, solar_rad_b1, Small_Grains_expT, pct_soybeans_b1, fuel_moisture_1000h_b2, pct_alfalfa_b1, surplus_kgha_norm_b1, pct_other_b1, doy_sin2, pct_small_grains_b1, min_temp_b1, solar_rad_b2, fuel_moisture_1000h_b1, evapotranspiration_b1, doy_cos2, min_temp_b0, fuel_moisture_1000h_b0, evapotranspiration_b0, solar_rad_b0, min_temp_b2, max_temp_b0, evapotranspiration_b2, max_temp_b2, max_temp_b1, min_rel_humidity_b0, vpd_b0, vpd_b2, max_rel_humidity_b2, vpd_b1, max_rel_humidity_b1, min_rel_humidity_b2, precip_in_1d_b0, min_rel_humidity_b1, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 79 | 20 | 157472 | 76 | 0.8727 | 0.8504 | 0.7368 | 0.6737 | 0.6744 | 0.6408 | 2.8743 | 0.7507 | 0.5632 | 0.6511 | 2.6283 | 0.7393 | 0.5065 | 0.5681 | 0.1184 | -3.2608 | 0.2563 | 0.4069 | 0.9001 |

**Beta Table:**


---

# REG Model 9: date0727_REG

**Recipe:** recipe_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, pct_corn_b0, pct_hay_pasture_b1, roll_n_avg_except_this7d, mean_dist_to_sensor, rest_of_state_nitrate_lag2, max_dist_to_sensor, pct_nonag_b0, lat, pct_corn_b1, fuel_moisture_1000h_b1, Soybeans_expT, log_basin_area, lon, pct_alfalfa_b1, Corn_expT, surplus_kgha_norm_b1, Other_expT, pct_soybeans_b0, pct_hay_pasture_b0, pct_other_b0, rest_of_state_nitrate_lag3, doy_sin, Nonag_expT, surplus_kgha_norm_b0, Fallow_expT, Alfalfa_expT, Hay_Pasture_expT, pct_soybeans_b1, fuel_moisture_1000h_b0, pct_fallow_b1, pct_alfalfa_b0, rest_of_state_nitrate_lag5, pct_fallow_b0, Small_Grains_expT, pct_nonag_b1, pct_small_grains_b1, pct_other_b1, doy_sin2, pct_small_grains_b0, surplus_kgha_expT, doy_cos, evapotranspiration_b1, min_temp_b0, solar_rad_b1, min_temp_b1, doy_cos2, max_temp_b1, max_temp_b0, solar_rad_b0, vpd_b1, evapotranspiration_b0, min_rel_humidity_b0, max_rel_humidity_b1, vpd_b0, min_rel_humidity_b1, precip_in_1d_b1, max_rel_humidity_b0, precip_in_1d_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 79 | 20 | 157472 | 59 | 0.4073 | 0.3404 | 4.2779 | -6.9807 | 0.6800 | 0.2016 | 0.4112 | 0.3033 |

**Beta Table:**


---

# CLF Model 8: recipe_CLF_far_30

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 2.6043 | 0.7265 | 0.4990 | 0.8144 | 2.4140 | 0.7106 | 0.4663 | 0.7789 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# CLF Model 7: recipe_CLF_far_20

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 2.6043 | 0.7265 | 0.4990 | 0.7270 | 2.4140 | 0.7106 | 0.4663 | 0.6867 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# CLF Model 6: recipe_CLF2

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 2.6043 | 0.7265 | 0.4990 | 0.5656 | 2.4140 | 0.7106 | 0.4663 | 0.5243 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# CLF Model 5: recipe_CLF2

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 2.6043 | 0.7265 | 0.4990 | 0.5656 | 2.4140 | 0.7106 | 0.4663 | 0.5243 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# CLF Model 4: recipe_CLF2

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | loso_prauc_lift | loso_f2 | loso_mcc | loso_recall_at_far | lofo_prauc_lift | lofo_f2 | lofo_mcc | lofo_recall_at_far | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 2.6043 | 0.7265 | 0.4990 | 0.5656 | 2.4140 | 0.7106 | 0.4663 | 0.5243 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# CLF Model 3: recipe_CLF2

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, lon, rest_of_state_nitrate_lag3, Corn_b0, log_basin_area, Hay_Pasture_b2, Hay_Pasture_b0, Corn_b1, Corn_b2, rest_of_state_nitrate_lag5, Alfalfa_b0, Nonag_b1, Soybeans_b0, Nonag_b2, Nonag_b0, Small_Grains_b0, Alfalfa_b2, Soybeans_b2, surplus_kgha_b2, Soybeans_b1, surplus_kgha_b1, total_kg_N_b2, Other_b0, total_kg_N_b1, Hay_Pasture_b1, Alfalfa_b1, Small_Grains_b2, surplus_kgha_b0, doy_sin, doy_cos, total_kg_N_b0, Other_b2, Fallow_b1, Other_b1, fuel_moisture_1000h_b2, Fallow_b2, solar_rad_b1, Fallow_b0, Small_Grains_b1, min_temp_b1, evapotranspiration_b1, min_temp_b2, min_temp_b0, fuel_moisture_1000h_b1, solar_rad_b2, evapotranspiration_b0, solar_rad_b0, max_temp_b0, fuel_moisture_1000h_b0, max_temp_b1, evapotranspiration_b2, max_temp_b2, min_rel_humidity_b2, min_rel_humidity_b0, vpd_b0, max_rel_humidity_b1, vpd_b1, max_rel_humidity_b2, vpd_b2, max_rel_humidity_b0, precip_in_1d_b2, precip_in_1d_b0, min_rel_humidity_b1, precip_in_1d_b1  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 68 | 0.8430 | 0.8197 | 0.6841 | 0.6341 | 0.6375 | 0.6113 | 0.1368 | -3.9490 | 0.2627 | 0.2375 | 0.9004 |

**Beta Table:**


---

# REG Model 2: recipe_REG2

**Recipe:** recipe_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, roll_n_avg_except_this7d, lat, mean_dist_to_sensor, log_basin_area, max_dist_to_sensor, lon, fuel_moisture_1000h_b1, rest_of_state_nitrate_lag2, Nonag_b0, Corn_b0, rest_of_state_nitrate_lag3, Alfalfa_b0, Hay_Pasture_b1, total_kg_N_b1, surplus_kgha_b1, Alfalfa_b1, Other_b0, Hay_Pasture_b0, doy_sin, Corn_b1, Soybeans_b1, Soybeans_b0, rest_of_state_nitrate_lag5, surplus_kgha_b0, Other_b1, fuel_moisture_1000h_b0, Nonag_b1, total_kg_N_b0, Small_Grains_b0, Fallow_b1, Fallow_b0, Small_Grains_b1, doy_cos, max_temp_b1, min_temp_b1, evapotranspiration_b0, min_temp_b0, solar_rad_b1, evapotranspiration_b1, solar_rad_b0, vpd_b1, max_temp_b0, max_rel_humidity_b1, min_rel_humidity_b1, vpd_b0, min_rel_humidity_b0, precip_in_1d_b1, max_rel_humidity_b0, precip_in_1d_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 50 | 0.3800 | 0.3280 | 4.3575 | -7.3928 | 0.6464 | 0.2094 | 0.4187 | 0.2445 |

**Beta Table:**


---

# CLF Model 1: recipe_CLF2

**Recipe:** recipe_CLF  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, rest_of_state_nitrate_lag2, lat, mean_dist_to_sensor, max_dist_to_sensor, rest_of_state_nitrate_lag3, lon, Nonag_b0, Corn_b0, log_basin_area, Soybeans_b0, Hay_Pasture_b1, Corn_b1, Nonag_b1, Alfalfa_b1, rest_of_state_nitrate_lag5, Hay_Pasture_b0, surplus_kgha_b0, Soybeans_b1, Alfalfa_b0, Other_b0, total_kg_N_b1, Other_b1, surplus_kgha_b1, total_kg_N_b0, doy_sin, Fallow_b1, Fallow_b0, Small_Grains_b1, doy_cos, solar_rad_b0, evapotranspiration_b0, Small_Grains_b0, fuel_moisture_1000h_b1, min_temp_b0, fuel_moisture_1000h_b0, solar_rad_b1, min_temp_b1, max_temp_b0, max_temp_b1, evapotranspiration_b1, min_rel_humidity_b1, max_rel_humidity_b1, vpd_b1, vpd_b0, min_rel_humidity_b0, precip_in_1d_b1, max_rel_humidity_b0, precip_in_1d_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_auc | lofo_auc | loso_prauc | lofo_prauc | loso_f1 | lofo_f1 | brier | persist_skill | base | between_rate_r2 | macro_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 49 | 0.8358 | 0.8066 | 0.6759 | 0.6249 | 0.6279 | 0.5976 | 0.1385 | -4.0123 | 0.2627 | 0.2201 | 0.8938 |

**Beta Table:**


---

# REG Model 0: recipe_REG2

**Recipe:** recipe_REG  
**True Lofo:** ?  
**Notes:** None 

**Features:** rest_of_state_nitrate_lag1, roll_n_avg_except_this7d, lat, mean_dist_to_sensor, log_basin_area, max_dist_to_sensor, lon, fuel_moisture_1000h_b1, rest_of_state_nitrate_lag2, Nonag_b0, Corn_b0, rest_of_state_nitrate_lag3, Alfalfa_b0, Hay_Pasture_b1, total_kg_N_b1, surplus_kgha_b1, Alfalfa_b1, Other_b0, Hay_Pasture_b0, doy_sin, Corn_b1, Soybeans_b1, Soybeans_b0, rest_of_state_nitrate_lag5, surplus_kgha_b0, Other_b1, fuel_moisture_1000h_b0, Nonag_b1, total_kg_N_b0, Small_Grains_b0, Fallow_b1, Fallow_b0, Small_Grains_b1, doy_cos, max_temp_b1, min_temp_b1, evapotranspiration_b0, min_temp_b0, solar_rad_b1, evapotranspiration_b1, solar_rad_b0, vpd_b1, max_temp_b0, max_rel_humidity_b1, min_rel_humidity_b1, vpd_b0, min_rel_humidity_b0, precip_in_1d_b1, max_rel_humidity_b0, precip_in_1d_b0  
**Scores:**

| n_sites | n_families | n_rows | n_feat | loso_r2 | lofo_r2 | rmse | persist_skill | spearman | between_r2 | within_r2 | macro_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 19 | 160074 | 50 | 0.3800 | 0.3280 | 4.3575 | -7.3928 | 0.6464 | 0.2094 | 0.4187 | 0.2445 |

**Beta Table:**


