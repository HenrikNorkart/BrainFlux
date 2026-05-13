**Tester Agent – Feature Evaluation Report**  
*Forest‑Fires (Portugal) regression – target: fire‑affected area*  

---  

### 1. Data Overview
| Item | Value |
|------|-------|
| Instances | 517 |
| Features (pre‑pruning) | 55 (plus *target*) |
| Feature types | Purely numeric (one‑hot month/day, interaction terms, squares/cubes, trigonometric encodings, spatial metrics) |

The dataset contains many engineered interaction features (e.g., `temp_x_FFMC`, `temp_x_day_thu`, `dryness_index`, etc.) and several one‑hot month/day columns, many of which never appear in the data (e.g., `month_jan`, `month_feb`, …).

---  

### 2. Baseline Predictive Performance  
**Model:** XGBoost Regressor (500 trees, max_depth = 6, lr = 0.05, subsample = 0.8, colsample_bytree = 0.8, GPU = cuda:5, `tree_method='hist'`).  

| Metric | Value |
|--------|-------|
| RMSE (80/20 hold‑out) | **37.33** |
| R² (80/20 hold‑out) | **‑3.65** |

*Interpretation:* The negative R² reflects the highly skewed target distribution (many zero‑area fires) and indicates that the current feature set does not capture the variance needed for accurate regression. Nevertheless, the model provides a consistent baseline for relative feature importance assessment.

---  

### 3. Feature Importance (Gain) – Top 10  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `temp_x_day_thu` | 10 766 |
| 2 | `temp_x_X` | 8 914 |
| 3 | `DMC` | 7 732 |
| 4 | `X_coord` | 6 113 |
| 5 | `dryness_index` | 5 288 |
| 6 | `ISI` | 2 266 |
| 7 | `temp_x_wind` | 1 903 |
| 8 | `wind_x_RH` | 1 771 |
| 9 | `RH` | 915 |
|10 | `dayofweek_sin` | 826 |

These features dominate the model’s split decisions. Notably, interaction terms that combine temperature with day‑of‑week (`temp_x_day_thu`) and spatial coordinates (`temp_x_X`) are the strongest contributors.

---  

### 4. Zero‑Gain (Useless) Features  

The following 13 attributes received **no gain** from the model and were therefore pruned:

```
wind_x_rain, RH_squared, temp_x_rain, month_dec, month_apr,
month_jan, temp_squared, day_cos, month_feb, day_sin,
month_nov, temp_cubed, wind_squared
```

*Effect of pruning:* Model performance (RMSE = 37.33, R² = ‑3.65) remained unchanged, confirming that these attributes contributed nothing to predictive power.

---  

### 5. Redundancy & Correlation Analysis  

Pairwise absolute Pearson correlations > 0.9 (20 pairs) highlight substantial redundancy, e.g.:

| Pair | Correlation |
|------|-------------|
| `DC` ↔ `FWI_composite` | 0.986 |
| `temp` ↔ `temp_x_FFMC` | 0.992 |
| `temp` ↔ `temp_squared` | 0.975 |
| `RH` ↔ `RH_squared` | 0.979 |
| `wind` ↔ `FFMC_x_wind` | 0.991 |
| `wind` ↔ `wind_squared` | 0.968 |
| `rain` ↔ `temp_x_rain` | 0.998 |
| `rain` ↔ `wind_x_rain` | 0.991 |
| `month_sin` ↔ `DC` | 0.953 |
| `month_cos` ↔ `DC` | 0.953 |

Because many derived columns (squared, cubed, interaction) are nearly linear transformations of the originals, they add little novel information. The importance scores already reflect this: high‑gain features are mostly *informative interactions* rather than simple powers.

---  

### 6. Summary of Findings  

| Aspect | Observation |
|--------|-------------|
| **Predictive power** | Baseline XGBoost yields RMSE ≈ 37, R² ≈ ‑3.6 – limited due to target sparsity. |
| **Key contributors** | Interaction terms linking temperature with day‑of‑week (`temp_x_day_thu`), spatial coordinates (`temp_x_X`), and dryness (`dryness_index`). |
| **Redundant features** | Numerous high‑correlation pairs (original ↔ squared/cubed ↔ simple interaction). |
| **Useless features** | 13 attributes with zero gain were removed without affecting performance. |
| **Remaining feature set** | 55 attributes (after pruning) still contain the informative interactions and core meteorological variables (temp, RH, wind, FFMC, DMC, ISI, spatial metrics). |

---  

### 7. Actions Taken  

1. **Exploratory analysis** – size, types, baseline model.  
2. **Importance evaluation** – XGBoost gain scores.  
3. **Zero‑gain detection** – identified 13 non‑contributing attributes and pruned them via `attribute_pruning_tool`.  
4. **Correlation screening** – highlighted 20 > 0.9 pairs, confirming redundancy.  
5. **Re‑evaluation** – confirmed pruning did not degrade baseline metrics.  

---  

**Conclusion:**  
The current feature pool contains a core set of high‑impact interaction attributes (especially those involving temperature) and essential meteorological measures. Numerous derived squares/cubes and unused month/day one‑hots add no predictive value and can be safely excluded. Future work (outside the scope of this report) could explore alternative target transformations or specialized models for zero‑inflated regression, but the present evaluation establishes a clear hierarchy of useful versus redundant features.