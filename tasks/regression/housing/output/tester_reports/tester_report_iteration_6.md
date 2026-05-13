**Feature‑Testing Report – Housing Median‑Value Prediction**

---

### 1. Experimental Setup  
* **Model:** XGBoostRegressor (400 trees, max_depth 6, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8).  
* **Hardware:** GPU `cuda:5` with `tree_method="hist"` (as required).  
* **Data split:** 80 % training / 20 % hold‑out (random_state 42).  
* **Metrics evaluated:** R², MAE, RMSE.  
* **Feature‑importance metric:** Gain from XGBoost’s booster.  

---

### 2. Baseline Results (All 88 attributes)  

| Metric | Value |
|--------|-------|
| **R²** | **0.867** |
| **MAE** | **27,258** |
| **RMSE** | **42,218** |

**Top‑15 gain‑based features (baseline)**  

1. `ocean_proximity_INLAND`  
2. `income_per_person`  
3. `inland_x_log_age`  
4. `income_bedrooms_per_person`  
5. `log_age_x_median_income`  
6. `dist_to_nearest_city`  
7. `income_longitude_interaction`  
8. `income_rooms_per_person`  
9. `housing_median_age_sq`  
10. `log_age_x_dist_to_city`  
11. `dist_to_sf`  
12. `latitude_squared`  
13. `dist_to_coast`  
14. `lat_sq_long`  
15. `near_ocean_rooms_per_person`  

*Observation:* Many of these features are highly correlated (|ρ| > 0.8), suggesting redundancy.

---

### 3. Redundancy & Correlation Analysis  

Pairs with absolute Pearson correlation > 0.8 (selected examples):

| Feature A | Feature B | ρ |
|-----------|-----------|---|
| `ocean_proximity_INLAND` | `inland_x_log_age` | 0.976 |
| `income_per_person` | `log_age_x_median_income` | 0.841 |
| `income_per_person` | `income_longitude_interaction` | 0.893 |
| `income_per_person` | `income_rooms_per_person` | 0.826 |
| `income_bedrooms_per_person` | `income_rooms_per_person` | 0.916 |
| `dist_to_nearest_city` | `log_age_x_dist_to_city` | 0.972 |
| `dist_to_sf` | `latitude_squared` | 0.891 |
| `dist_to_sf` | `dist_to_coast` | 0.965 |
| `dist_to_sf` | `lat_sq_long` | 0.905 |
| `latitude_squared` | `lat_sq_long` | 0.999 |

These clusters indicate that several engineered attributes convey essentially the same information.

---

### 4. Pruning Decision  

**Removed attributes (8 total):**  

- `ocean_proximity_INLAND`  
- `income_bedrooms_per_person`  
- `log_age_x_median_income`  
- `income_longitude_interaction`  
- `latitude_squared`  
- `lat_sq_long`  
- `dist_to_coast`  
- `near_ocean_rooms_per_person`

*Rationale:* Each belongs to a high‑correlation cluster and contributed little unique predictive power relative to retained representatives (e.g., `inland_x_log_age`, `income_per_person`, `dist_to_sf`).

---

### 5. Post‑Pruning Results  

| Metric | Value |
|--------|-------|
| **R²** | **0.8655** |
| **MAE** | **27,449** |
| **RMSE** | **42,444** |

**Top‑10 gain‑based features (after pruning)**  

1. `income_per_person`  
2. `inland_x_log_age`  
3. `income_rooms_per_person`  
4. `ocean_proximity_LT1H_OCEAN`  
5. `dist_to_nearest_city`  
6. `pop_per_bedroom_sq`  
7. `log_age_x_dist_to_city`  
8. `dist_to_sf`  
9. `median_income_squared`  
10. `ocean_proximity_ISLAND`  

*Observation:* Predictive performance dropped marginally (ΔR² ≈ ‑0.0014, ΔMAE ≈ +191, ΔRMSE ≈ +226). The loss is negligible given the 9 % reduction in feature count, confirming that the pruned attributes were largely redundant.

---

### 6. Conclusions  

1. **Predictive Power:** The engineered feature set yields strong performance (R² ≈ 0.87).  
2. **Key Drivers:** Income‑related metrics (`income_per_person`, `income_rooms_per_person`, `inland_x_log_age`) dominate importance, followed by geographic distance variables and specific ocean‑proximity categories.  
3. **Redundancy:** Numerous features are highly collinear; pruning them does **not** materially degrade model quality.  
4. **Feature Set Size:** Reducing from 88 to **80** attributes (≈ 9 % fewer) improves model interpretability and reduces computational load while preserving accuracy.  

**Recommendation:** Retain the pruned attribute list as the final feature set for downstream modeling. Further improvements should focus on model hyper‑parameter tuning or alternative algorithms rather than additional feature engineering, as the current set already captures the dominant predictive signals.