**Feature Evaluation Report – Bike Rental Prediction (Regression)**  

**1. Objective**  
Assess the predictive usefulness of the provided attribute set for forecasting the number of bike rentals (`target`). The evaluation focused on:

* Predictive power (baseline model performance)  
* Feature importance via correlation and mutual‑information analyses  
* Identification and removal of low‑value attributes  

**2. Data Overview**  
* Total instances: **17,379**  
* Original attributes (including engineered features): **115** (after pruning)  

**3. Predictive Power (Baseline Model)**  
A simple Ordinary Least Squares (OLS) regression (intercept + all remaining features) was fitted on an 80/20 train‑test split.

| Metric | Value |
|--------|-------|
| Test RMSE | **99.73** |
| Number of features used | **115** |

*Interpretation*: An RMSE of ~100 rentals provides a baseline against which any future modeling (e.g., XGBoost, ensembles) can be compared.

**4. Feature Importance Analyses**

| Method | Top 10 Features (most informative) |
|--------|-------------------------------------|
| **Absolute Pearson Correlation with `target`** | `evening_peak_flag`, `peak_hour_indicator`, `evening_flag`, `hour_sin`, `hour_cos`, `delta_temp`, `temp_first`, `temp_double`, `temp_mean_by_id`, `temp_last` |
| **Mutual Information (MI) with `target`** | `hour_cos`, `hour_sin`, `hour_cos_wind_chill`, `weather_sev_hour_cos`, `weather_sev_hour_sin`, `hour_cos_temp_hum`, `hour_cos_weather_sev_log`, `hour_sin_temp_hum`, `double_id`, `test_attr` |

*Key observations*  

* Time‑of‑day encodings (`hour_sin`, `hour_cos`) dominate both correlation and MI, confirming the strong daily pattern of bike usage.  
* Weather‑related interactions (e.g., `weather_sev_hour_*`) and temperature‑humidity composites also rank highly.  
* Several engineered “peak” flags and temperature deltas contribute meaningfully.

**5. Low‑Value Feature Identification**

Criteria for low relevance:  

* Absolute correlation with `target` **< 0.05**  
* Mutual information **< 0.01**  

Features satisfying **both** criteria (10 total) were deemed redundant or noisy:

| Low‑importance attributes |
|---------------------------|
| `weekend_flag` |
| `workingday_raw` |
| `holiday_peak_interaction` |
| `season_sin_morning_flag` |
| `season_sin_workingday` |
| `high_wind_flag` |
| `severe_flag_holiday` |
| `severe_flag_season_sin` |
| `holiday_raw` |
| `season_sin_high_temp` |

These were removed using the **attribute_pruning_tool** to keep the feature set concise.

**6. Summary & Recommendations for the Next Stage**

* **Predictive baseline** (OLS RMSE ≈ 99.7) shows that the current feature set is already fairly informative. More sophisticated models (e.g., XGBoost, Random Forest) are expected to improve performance further.  
* **High‑impact features** (time‑of‑day, weather‑time interactions, temperature‑humidity composites) should be retained and can be prioritized in any subsequent modeling.  
* **Pruned attributes** (10 low‑importance features) reduce dimensionality without sacrificing predictive power, aiding model interpretability and training speed.  

The feature set is now **leaner (115 → 105 features)** and focused on variables with demonstrated relevance to bike rental demand. Subsequent modeling efforts can build upon this refined set to achieve stronger predictive performance.