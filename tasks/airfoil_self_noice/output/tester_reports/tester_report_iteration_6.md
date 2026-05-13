**Comprehensive Feature Evaluation Report**

**1. Baseline Model (All 65 features)**
- **Algorithm:** XGBoostRegressor (n_estimators=500, max_depth=6, learning_rate=0.05, device="cuda:5", tree_method="hist")
- **Performance:**  
  - **RMSE:** 1.367  
  - **R²:** 0.959  
- **Top 10 Feature Gains (importance):**  
  1. `pca_component_1` – 282.64  
  2. `pca1_thickness_to_chord_interaction` – 167.89  
  3. `log_chord_length` – 66.65  
  4. `log_frequency_angle_strouhal` – 66.10  
  5. `pca1_Mach_interaction` – 64.46  
  6. `angle_rad_cubic_disp_vel_ratio` – 63.26  
  7. `angle_log_frequency_strouhal` – 56.22  
  8. `freq_chord_product` – 46.63  
  9. `angle_disp_interaction` – 42.28  
  10. `cos_angle_thickness_to_chord_interaction` – 35.13  

**2. Correlation & Redundancy Analysis**
- **Highly correlated (>0.9) pairs** (examples):  
  - `free_stream_velocity` ↔ `log_free_stream_velocity` (0.995)  
  - `free_stream_velocity` ↔ `free_stream_velocity_squared` (0.996)  
  - `chord_length` ↔ `log_chord_length` (0.945)  
  - `displacement_thickness` ↔ `displacement_thickness_squared` (0.947)  
  - `frequency` ↔ `frequency_squared` (0.913)  
- **Target‑feature correlations:**  
  - Strongest: `pca_component_1` (0.694), `pca1_Mach_interaction` (0.688), several angle‑Strouhal interaction terms (~0.66).

**3. Feature Pruning**
- **Low‑gain (<1) features:** `angle_in_radians`, `log_free_stream_velocity`, `free_stream_velocity`, `free_stream_velocity_squared`, `chord_length_squared`.
- **Redundant features (based on >0.95 correlation & lower importance):** 23 attributes including `Reynolds_number`, `Ma_thickness_to_chord`, `angle_rad_cubic`, `pca2_Mach_interaction`, `cos_angle`, `sin_angle`, `displacement_thickness`, `frequency_cubic`, `pca1_Mach_interaction`, etc.
- **Pruned attribute list:**  
  ```
  angle_in_radians, log_free_stream_velocity, free_stream_velocity,
  free_stream_velocity_squared, chord_length_squared, Reynolds_number,
  Ma_thickness_to_chord, angle_rad_cubic, pca2_Mach_interaction,
  cos_angle, cos_angle_log_chord_length, sin_angle, displacement_thickness,
  angle_rad_cubic_reynolds_interaction, frequency_cubic,
  freq_chord_sq_interaction, pca1_Mach_interaction, thickness_to_chord,
  angle_log_frequency_strouhal, Re_strouhal, reduced_freq_variant,
  cos_angle_mach_interaction, angle_rad_squared
  ```

**4. Post‑Pruning Model (42 features)**
- **Performance:**  
  - **RMSE:** 1.363 (virtually unchanged)  
  - **R²:** 0.959 (unchanged)  
- **Interpretation:** Removing redundant/low‑impact attributes did **not** degrade predictive performance, confirming that the remaining 42 features capture the essential information.

**5. Conclusions**
- The engineered feature set provides **very strong predictive power** for the airfoil self‑noise regression task (R² ≈ 0.96, RMSE ≈ 1.36 dB).
- **Key drivers** are PCA‑derived components and interaction terms involving angle of attack, Strouhal number, and Mach number.
- **Redundancies** mainly stem from raw vs. transformed versions of the same physical measurements (e.g., velocity and its log/squared forms). Pruning these yields a more parsimonious model without sacrificing accuracy.
- **Recommended final feature set:** the 42 retained attributes after pruning, which balance model simplicity and performance.