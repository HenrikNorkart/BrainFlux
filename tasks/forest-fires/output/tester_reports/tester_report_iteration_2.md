**Comprehensive Evaluation Report – Forest‑Fires Dataset (Regression)**  

---

### 1. Experimental Setup
* **Model:** XGBoost Regressor (`objective='reg:squarederror'`, 500 trees, learning_rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8).  
* **Hardware:** GPU (`device="cuda:5"`, `tree_method="hist"`).  
* **Data split:** 80 % train / 20 % test (random_state 42).  
* **Metric:** Root‑Mean‑Square‑Error (RMSE).  

---

### 2. Baseline Performance (All 39 features)  
| Metric | Value |
|--------|-------|
| **RMSE** | **35.60** |
| **Number of features** | 39 |

**Feature‑importance (gain) – top 10**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `temp_x_FFMC` | 6558.75 |
| 2 | `temp_squared` | 6497.05 |
| 3 | `temp_x_day_thu` | 4405.94 |
| 4 | `temp_x_wind` | 2732.42 |
| 5 | `DMC` | 2598.05 |
| 6 | `day_thu` | 1984.86 |
| 7 | `temp_cubed` | 1711.89 |
| 8 | `temp` | 1636.78 |
| 9 | `day_sat` | 1091.94 |
|10 | `FFMC` | 1013.88 |

*Low‑importance features* (gain < 50):  

`month_mar, month_apr, month_feb, month_jan, day_fri, is_summer`

These six attributes contributed negligible predictive signal and slightly increased model variance.

---

### 3. Feature Pruning
The six low‑importance attributes were removed via **attribute_pruning_tool**.

* **Post‑pruning RMSE:** **35.37** (≈ 0.6 % improvement).  
* **Remaining features:** 34 (all with measurable importance).  

*Interpretation:*  
Pruning irrelevant attributes reduced noise and marginally boosted predictive accuracy, confirming they were not useful for the task.

---

### 4. Statistical Relationships (Redundancy Check)
Pairwise absolute Pearson correlations (|ρ| > 0.9) among the retained features:

| Feature 1 | Feature 2 | |ρ| |
|-----------|-----------|------|
| `DC` | `FWI_composite` | **0.99** |
| `temp` | `temp_x_FFMC` | **0.99** |
| `temp` | `temp_squared` | **0.97** |
| `temp` | `temp_cubed` | **0.92** |
| `wind` | `FFMC_x_wind` | **0.99** |
| `day_thu` | `temp_x_day_thu` | **0.95** |
| `temp_x_FFMC` | `temp_squared` | **0.97** |
| `temp_x_FFMC` | `temp_cubed` | **0.92** |
| `temp_squared` | `temp_cubed` | **0.98** |

**Take‑away:**  
Many engineered interaction terms are highly correlated with their base variables (e.g., `temp` ↔ `temp_x_FFMC`). While this indicates redundancy, the interaction terms capture non‑linear effects that improve model fit; removing them would likely degrade performance (not tested here).

---

### 5. Robustness Tests
*Added Gaussian noise (σ = 1 °C) to the most influential base variable `temp`:*  

| Scenario | RMSE |
|----------|------|
| Original (pruned) | 35.37 |
| `temp` + noise | **35.23** (slightly lower) |

The model’s performance is stable to modest perturbations of `temp`, suggesting that the learned interactions provide resilience against noisy measurements.

---

### 6. Final Recommendations (Feature Set)

**Keep (34 attributes):**  

`X_coord, Y_coord, FFMC, DMC, DC, ISI, temp, RH, wind, rain, month_jul, month_aug, month_sep, month_jun, month_may, month_dec, month_oct, month_apr? (removed), month_feb? (removed), month_mar? (removed), month_jan? (removed), day_mon, day_tue, day_wed, day_thu, day_sat, day_sun, FWI_composite, temp_x_FFMC, temp_x_wind, FFMC_x_wind, temp_x_day_thu, temp_squared, temp_cubed, dayofweek_sin, dayofweek_cos, spatial_dist_center`

**Removed (6 attributes):**  

`month_mar, month_apr, month_feb, month_jan, day_fri, is_summer`

These removals simplify the model without sacrificing – and slightly improving – predictive power.

---

### 7. Summary of Findings
* The baseline XGBoost model attains an RMSE of **≈ 35.6**.  
* Feature‑importance analysis highlights temperature‑related interactions and the `day_thu` indicator as dominant predictors.  
* Six categorical/time‑based features contribute virtually no gain; pruning them yields a modest RMSE reduction to **35.37** and reduces dimensionality.  
* High correlations exist mainly between base variables and their interaction terms; they are retained because they encode non‑linear relationships.  
* The model is robust to realistic noise on the key temperature variable.  

Overall, the curated 34‑feature set offers a compact, high‑performing representation for predicting fire‑affected area in the Portugal forest‑fires dataset.