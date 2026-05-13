**Comprehensive Feature Evaluation Report – Crab Age Regression**

---

### 1. Experimental Setup
- **Model:** XGBoostRegressor  
  - `objective='reg:squarederror'`, `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`  
  - GPU‑accelerated (`device='cuda:5'`, `tree_method='hist'`)  
- **Data Split:** 80 % train / 20 % test (random_state = 42).  
- **Metrics:** RMSE (root‑mean‑square error) and R² (coefficient of determination).  

---

### 2. Baseline Performance (All 44 engineered features)

| Metric | Value |
|--------|-------|
| **RMSE** | **2.172** |
| **R²**   | **0.544** |
| **Number of features** | 44 |

*The baseline model already shows moderate predictive power for crab age.*

---

### 3. Feature Importance (Gain)

Top contributors (gain > 5) – sorted descending:

| Rank | Feature | Gain |
|------|------------------------------|------|
| 1 | Height_Weight_Interaction_Squared | 305.64 |
| 2 | Height_Weight_Interaction | 100.76 |
| 3 | Estimated_Volume_Squared | 76.54 |
| 4 | Log_Shucked_Weight_Ratio | 36.34 |
| 5 | Height_Weight_Diameter_Interaction | 26.14 |
| 6 | Height_Length_Diameter_Interaction | 24.58 |
| 7 | Shucked_Weight_Ratio | 23.69 |
| 8 | Sex_Shucked_Weight_Ratio_Squared | 22.14 |
| 9 | Shucked_Weight_Ratio_Squared | 21.12 |
| 10 | Diameter_Squared | 15.63 |
| … | … | … |

Features with **gain < 5** (effectively negligible) were:

- `Weight_to_Volume_Ratio`
- `Sex_Encoded`
- `Log_Weight_to_Length_Ratio`

These three contributed little to model decisions.

---

### 4. Pruning Low‑Impact Features

After removing the three low‑importance attributes:

| Metric | Value |
|--------|-------|
| **RMSE** | **2.157** (slight improvement) |
| **R²**   | **0.550** (increase) |
| **Number of features** | **41** |

*Pruning did **not** degrade performance; it actually yielded a modest gain.*

---

### 5. Robustness Tests

| Noise Level (σ × std) | RMSE |
|----------------------|------|
| 0 % (baseline) | 2.172 |
| 10 % | 2.095 (unexpectedly better – model regularization) |
| 30 % | 2.238 (performance drop but still acceptable) |

The feature set demonstrates **reasonable robustness** to moderate perturbations.

---

### 6. Correlation & Redundancy Analysis

- **131** feature pairs exhibit **|ρ| > 0.9**, many stemming from the `Estimated_Volume` family.
- Despite high collinearity, most of these features retain non‑trivial importance; removing them en masse harms predictive power.
- A “top‑15 only” experiment (using the highest‑gain features) reduced performance sharply (RMSE = 2.33, R² = 0.48), confirming that a broader feature set is beneficial.

**Conclusion:** Redundancy is present, but the current importance distribution suggests that each correlated feature adds incremental value; aggressive reduction is not advisable.

---

### 7. Final Feature Set

- **Retained:** 41 engineered attributes after pruning `Weight_to_Volume_Ratio`, `Sex_Encoded`, `Log_Weight_to_Length_Ratio`.
- **Removed:** The three low‑importance attributes (pruned via `attribute_pruning_tool`).

This set balances **predictive performance**, **model stability**, and **manageability**.

---

### 8. Key Takeaways

1. **Predictive Power:** RMSE ≈ 2.16, R² ≈ 0.55 with 41 features – a solid baseline for crab‑age prediction.
2. **Feature Importance:** A handful of interaction‑heavy features dominate, but many moderate‑importance attributes collectively improve accuracy.
3. **Pruning:** Eliminating only the clearly non‑contributive features improves both metrics and reduces dimensionality.
4. **Robustness:** Model tolerates noise up to ~30 % of feature standard deviation with limited performance loss.
5. **Redundancy Management:** While many features are highly correlated, they each provide marginal gains; retain them unless computational constraints demand further reduction.

*No further feature engineering or preprocessing recommendations are provided, per the task scope.*