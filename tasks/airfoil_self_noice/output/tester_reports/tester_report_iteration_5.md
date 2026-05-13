**Tester Agent Report – Feature Evaluation for Airfoil Self‑Noise Regression**

**1. Experimental Setup**
- **Model:** XGBoost regressor (objective = reg:squarederror, 300 trees, depth = 6, learning_rate = 0.05, subsample = 0.9, colsample_bytree = 0.9, device = cpu, tree_method = hist).  
- **Data split:** 80 % train / 20 % test (random_state = 42).  
- **Metrics:** Root Mean Squared Error (RMSE).  
- **Feature‑importance methods:**  
  * *Gain* from XGBoost (normalized).  
  * *Permutation importance* (negative MSE, 5 repeats).  
- **Correlation analysis:** Absolute Pearson correlation among all features.  

**2. Baseline Results (All 57 features)**
- **RMSE:** **1.49** – strong predictive performance.  
- **Gain importance (top contributors, > 0.5 % of total gain):**  

| Feature | Normalized Gain |
|---|---|
| angle_log_frequency_strouhal | **0.3429** |
| log_frequency_angle_strouhal | **0.2260** |
| angle_strouhal_interaction | 0.0791 |
| freq_chord_product | 0.0349 |
| sin_angle_strouhal_interaction | 0.0260 |
| angle_rad_cubic_disp_vel_ratio | 0.0268 |
| freq_disp_interaction | 0.0142 |
| angle_in_radians | 0.0160 |
| … (total 24 features) |  |

- **Permutation importance (high impact):**  
  - `strouhal_number` (0.150),  
  - `angle_strouhal_interaction` (0.147),  
  - `log_frequency_angle_strouhal` (0.142),  
  - `freq_disp_interaction` (0.226),  
  - `freq_chord_product` (0.108).  

- **Correlation insights:**  
  - Many top features are highly correlated (e.g., `angle_log_frequency_strouhal` ↔ `log_frequency_angle_strouhal` r ≈ 0.99).  
  - Interaction terms often share strong linear relationships with their base variables, suggesting potential redundancy.

**3. Feature Pruning**
- **Criterion:** Keep features with normalized gain > 0.005 (≈ 0.5 % contribution).  
- **Retained (24) features:**  

```
chord_length, displacement_thickness, strouhal_number,
log_frequency, log_displacement_thickness, freq_disp_interaction,
freq_chord_product, reduced_freq_variant, angle_chord_interaction,
angle_disp_interaction, angle_strouhal_interaction,
freq_chord_sq_interaction, Re_strouhal, angle_in_radians,
sin_angle_strouhal_interaction, sin_angle_reynolds_interaction,
cos_angle_log_chord_length, angle_rad_cubic_reynolds_interaction,
sin_angle_freq_chord_interaction, angle_rad_cubic_disp_vel_ratio,
sin_angle_strouhal_mach, angle_squared_reynolds,
log_frequency_angle_strouhal, angle_cubic_reynolds,
angle_log_frequency_strouhal
```

- **Pruned (32) attributes:** all others (e.g., raw `frequency`, `angle_of_attack`, `free_stream_velocity`, basic squared terms, many raw trigonometric features, etc.).

**4. Post‑Pruning Evaluation**
- **RMSE with reduced set:** **1.54** (≈ 4 % increase vs. full model).  
- **Interpretation:** The modest loss confirms that the removed attributes contributed little predictive value, while the model remains highly accurate.

**5. Conclusions**
- The engineered interaction and logarithmic features dominate predictive power; raw physical measurements (`frequency`, `angle_of_attack`, `free_stream_velocity`) are largely redundant after transformation.
- Retaining the 24 high‑gain features yields a compact, interpretable model with minimal performance degradation.
- Further work (outside current scope) could explore dimensionality reduction (e.g., PCA) on the correlated interaction terms, but the current feature set is already efficient.

**6. Action Items**
- **Pruned attributes** have been removed via the `attribute_pruning_tool`.  
- The **selected 24 features** should be used for downstream modeling and documentation.  

*All observations have been recorded via `take_note_tool` for the final report.*