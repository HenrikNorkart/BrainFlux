**Feature Evaluation Report – Forest‑Fires Regression (Target: Burned Area)**  

---

### 1. Experimental Setup
| Step | Methodology |
|------|--------------|
| **Model** | XGBoost Regressor (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `tree_method='hist'`, `device='cuda:5'`) |
| **Data split** | 80 % train / 20 % test (random_state = 42) |
| **Metric** | Root‑Mean‑Square‑Error (RMSE) on the held‑out test set |
| **Feature pool** | 84 engineered attributes supplied in `df_attributes` (including the original meteorological variables and numerous interaction / polynomial terms). |

---

### 2. Baseline Performance
* **All 84 features** → **RMSE = 121.80**  

The model already benefited from the extensive engineered set, but many attributes contributed little or were redundant.

---

### 3. Feature‑Importance‑Driven Reduction
* Gain‑based importance from the full‑feature model was extracted.
* **Top 30 features by gain** were selected and a reduced model was trained.

| Result | RMSE |
|--------|------|
| **Top‑30 features only** | **104.18** |

The reduction already improved predictive power while cutting the feature count by ~ 64 %.

---

### 4. Redundancy / Correlation Analysis
* Pairwise absolute Pearson correlation (> 0.90) among the top‑30 revealed **15 highly correlated pairs** (e.g., `RH_sq` ↔ `RH_cu`, `Y_temp_sq` ↔ `Y_temp`, `log_FFMC_day_sin` ↔ `day_sin`, etc.).
* For each highly correlated pair, the attribute with the **higher gain** was retained.

**Attributes pruned (9 total):**  
`RH_cu`, `Y_temp`, `sqrt_ISI_day_sin`, `day_sin`, `fire_weather_composite_month_cos`, `fire_weather_composite_Y`, `X_temp`, `X_temp_sq`, `XY_temp`.

---

### 5. Final Compact Feature Set
After removing the redundant attributes, the **21‑feature** subset retained is:

| Feature | Approx. Gain* |
|---------|---------------|
| `RH_sq` | 14 192 |
| `Y_temp_sq` | 12 547 |
| `log_FFMC_day_sin` | 10 378 |
| `fire_weather_composite_X` | 9 513 |
| `dist_temp` | 7 296 |
| `ISI_RH` | 4 124 |
| `temp_sq` | 3 208 |
| `wind_DMC` | 2 965 |
| `temp_RH` | 2 393 |
| `fire_weather_composite_day_cos` | 2 264 |
| `log_ISI` | 1 734 |
| `ISI_wind` | 1 716 |
| `sqrt_ISI` | 1 519 |
| `temp_cu` | 1 422 |
| `dist_wind` | 1 251 |
| `weighted_fire_weather_composite_Y` | 1 199 |
| `sqrt_ISI_day_cos` | 1 120 |
| `RH_wind` | 1 045 |
| `sqrt_ISI_X` | 1 014 |
| `fire_weather_composite_day_sin` | 903 |
| `dist_RH` | 624 |

\*Gain values are taken from the full‑model importance ranking (higher → more predictive contribution).

---

### 6. Final Model Performance
* **21‑feature model** → **RMSE = 100.33**  

This is a **~18 % improvement** over the baseline and uses **only 25 % of the original attributes**, demonstrating that a compact, non‑redundant set can achieve superior predictive accuracy.

---

### 7. Key Take‑aways
1. **Gain‑based importance** is an effective first filter for high‑impact features in XGBoost.  
2. **Correlation screening** is essential; many engineered variables are mathematically linked (e.g., squares vs. cubes, sin/cos of the same periodic term). Retaining the higher‑gain member of each correlated pair prevents over‑parameterisation and improves generalisation.  
3. A **small, well‑chosen subset (21 features)** delivers the best trade‑off between model simplicity and predictive power for this forest‑fire regression task.  

---

**Prepared by:** Tester Agent – Feature Evaluation Loop.   (All steps executed via the provided Python, note‑taking, and pruning tools.)