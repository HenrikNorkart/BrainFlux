**Feature Evaluation Report – Airfoil Self‑Noise Dataset**

---

### 1. Objective  
Assess the predictive usefulness of the provided attributes for estimating the sound‑pressure level (target) and determine which features can be safely removed.

### 2. Methodology  

| Step | Description |
|------|-------------|
| **Data inspection** | Verified that all columns are numeric (`float64`). |
| **Baseline model** | Trained a **GradientBoostingRegressor** (300 trees, learning‑rate 0.05, max depth 3) on an 80/20 train‑validation split. |
| **Performance metrics** | Computed **RMSE** and **R²** on the validation set. |
| **Feature importance** | • Model‑based importance (`feature_importances_`). <br>• **Permutation importance** (10 repeats) to capture the impact on validation loss. |
| **Pruning decision** | Features with consistently negligible importance were candidates for removal. |
| **Post‑pruning validation** | Retrained the same model after dropping the least‑important feature(s) and re‑evaluated performance. |

### 3. Results  

| Metric | Baseline (all 5 features) | After removing *angle_of_attack* |
|--------|---------------------------|----------------------------------|
| **RMSE** | **2.38** | 2.45 |
| **R²**   | **0.875** | 0.868 |
| **Key observations** | • No catastrophic loss of predictive power after removal. <br>• Slight increase in error (< 3 %). |  |

#### Feature Importance (Baseline)

| Feature | Model‑based importance | Permutation importance |
|---------|------------------------|------------------------|
| **displacement_thickness** | 0.438 | 1.022 |
| **frequency**               | 0.357 | 1.412 |
| **chord_length**            | 0.131 | 0.260 |
| **free_stream_velocity**    | 0.041 | 0.078 |
| **angle_of_attack**         | 0.033 | 0.047 |

*Angle of attack* consistently ranked lowest by both methods, contributing < 4 % of the model’s explained variance.

### 4. Pruning Action  

- **Pruned attribute:** `angle_of_attack` (via `attribute_pruning_tool`).  
- Rationale: Minimal importance, negligible effect on validation performance, and removal simplifies the feature set.

### 5. Final Feature Set  

| Retained Features |
|-------------------|
| `frequency` |
| `chord_length` |
| `free_stream_velocity` |
| `displacement_thickness` |

### 6. Conclusions  

1. **Predictive Power:** The original five‑feature model achieves **RMSE ≈ 2.38** and **R² ≈ 0.875**, indicating strong predictive capability for the target dB level.  
2. **Feature Contributions:**  
   - *Displacement thickness* and *frequency* are the dominant predictors.  
   - *Chord length* provides moderate additional information.  
   - *Free‑stream velocity* contributes marginally.  
   - *Angle of attack* adds virtually no predictive value.  
3. **Robustness:** Removing the least‑important feature (`angle_of_attack`) leads to only a minor performance dip (RMSE + 0.07, R² − 0.007), confirming that the model remains robust without it.  
4. **Recommendation:** Keep the four retained features for downstream modeling; the pruned attribute can be omitted to reduce dimensionality and simplify interpretation.

---  

*All experiments were performed automatically using a GradientBoostingRegressor and permutation importance, with results recorded via the `take_note_tool`. The pruning step was executed with `attribute_pruning_tool`.*