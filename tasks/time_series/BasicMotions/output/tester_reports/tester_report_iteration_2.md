**Comprehensive Evaluation Report – Human‑Motion Sensor Features**

**1. Experimental Setup**  
- **Data:** 40 instances, 74 original attributes (accelerometer & gyroscope time‑domain & frequency‑domain statistics) + target (walking, resting, running, badminton).  
- **Model:** RandomForestClassifier (300 trees, `random_state=42`, `n_jobs=-1`).  
- **Validation:** Stratified 80/20 train‑test split (8 samples test).  
- **Metrics:** Accuracy, per‑class precision/recall/F1, feature‑importance (RF Gini), Pearson correlation for redundancy.  

**2. Initial Findings (All 74 features)**  
- **Predictive Power:** 100 % accuracy on test set (perfect classification of all four activities).  
- **Top‑10 Importance (Gini gain):**  
  1. `acc_x_min`  
  2. `gyro_z_std`  
  3. `acc_x_std`  
  4. `gyro_y_min`  
  5. `acc_y_std`  
  6. `gyro_z_spec_energy`  
  7. `gyro_z_min`  
  8. `acc_x_median`  
  9. `acc_x_iqr`  
  10. `acc_y_max`  
- **Redundancy:** 195 feature pairs showed Pearson |r| > 0.9. Notably, `acc_x_mean` correlated >0.95 with many gyro‑derived statistics (e.g., `gyro_z_std`, `gyro_sma`, `gyro_mag_mean`).  

**3. Feature Pruning Strategy**  
- For each highly correlated pair (|r| > 0.9), the feature with lower RandomForest importance was removed.  
- Goal: retain a compact, non‑redundant set while preserving predictive performance.  
- **Result:** 32 attributes pruned, leaving **42** features.  

**4. Post‑Pruning Evaluation**  
- **Predictive Power:** Accuracy remained **100 %** on the same test split.  
- **Top‑5 Importance (post‑pruning):**  
  1. `acc_x_iqr` (0.086)  
  2. `acc_mag_std` (0.075)  
  3. `gyro_x_spec_energy` (0.073)  
  4. `gyro_y_max` (0.068)  
  5. `gyro_mag_spec_energy` (0.055)  
- Redundant high‑correlation pairs were effectively eliminated, reducing multicollinearity and simplifying the feature space.  

**5. Robustness Observations**  
- The model’s perfect test performance persists despite a 43 % reduction in dimensionality, indicating that the retained features capture the essential discriminative signal of each activity.  
- High‑correlation pruning did not degrade performance, suggesting that many original features were redundant surrogates.  

**6. Conclusions**  
- The current feature set is **highly predictive** for classifying the four basic motions.  
- **Key predictive attributes** are mainly inter‑quartile ranges, magnitude standard deviations, and spectral energy measures from both accelerometer and gyroscope signals.  
- **Redundancy removal** (32 attributes) yields a **leaner model** (42 features) without sacrificing accuracy, improving interpretability and computational efficiency for downstream deployment.  

**Next Steps for the Team**  
- Communicate the pruned attribute list to the Extractor Agent for future data collection focus.  
- Consider cross‑validation with more folds or external validation (if additional data become available) to confirm generalisation beyond the current small sample.  

*All notes and pruning actions have been recorded via the system tools.*