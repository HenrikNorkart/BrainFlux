**Comprehensive Feature Evaluation Report**

**1. Objective**  
Assess the predictive power of the provided sensor‑derived attributes for classifying human motions (walking, resting, running, badminton) and identify a compact, high‑performing feature set.

**2. Methodology**  
- **Model:** XGBoost (multi‑class, `device="cuda:5"`, `tree_method="hist"`).  
- **Evaluation:** Stratified 5‑fold cross‑validation, accuracy and classification report.  
- **Feature Importance:** XGBoost gain importance.  
- **Redundancy Check:** Pearson correlation (>0.9) among top features.  
- **Feature Sub‑set Experiments:** Incrementally reduced the attribute set while monitoring accuracy.  

**3. Baseline Results (All 263 features)**  
| Metric | Value |
|--------|-------|
| Mean CV Accuracy | **0.975** |
| Std. Dev | **0.05** |
| Best Fold Accuracy | 1.0 |
| Worst Fold Accuracy | 0.875 |
| Classification Report (average) | Precision ≈ 0.98, Recall ≈ 0.98, F1 ≈ 0.975 (across 4 classes) |

**4. Feature Importance (Gain) – Top 20**  
1. `acc_y_num_peaks` – 12.05  
2. `acc_y_autocorr_lag2` – 9.40  
3. `acc_x_max` – 3.20  
4. `acc_x_mean` – 3.19  
5. `acc_x_std` – 2.93  
6. `acc_y_autocorr_lag1` – 2.90  
7. `gyro_mag_peak_to_peak_freq` – 2.18  
8. `acc_y_zero_crossing_rate` – 2.03  
9. `acc_x_min` – 1.96  
10. `acc_y_mean_interpeak_interval` – 1.84  
*(remaining features contributed < 1.2 gain each)*  

**5. Redundancy Analysis**  
High correlations (> 0.9) were observed among several acc‑x statistics (`acc_x_mean`, `acc_x_std`, `acc_x_max`, `acc_x_min`) and between `acc_mag_std` & `gyro_x_std`, `gyro_x_max` & `gyro_x_std`. This indicated substantial redundancy.

**6. Feature Sub‑set Experiments**

| Feature Set | Size | Mean CV Accuracy | Remarks |
|-------------|------|------------------|---------|
| Top 10 (including gyro & zero‑crossing) | 10 | **1.0** | Perfect classification, no over‑fitting observed. |
| Compact 6‑feature set | 6 | **0.975** | Matches baseline performance; most parsimonious with no loss. |
| 5‑feature set (removed `acc_x_max`) | 5 | 0.95 | Slight drop, indicating `acc_x_max` adds useful nuance. |

**7. Final Selected Feature Set (6 attributes)**  
- `acc_y_num_peaks`  
- `acc_y_autocorr_lag2`  
- `acc_x_mean`  
- `acc_x_std`  
- `acc_y_autocorr_lag1`  
- `acc_x_max`  

These features capture peak activity count, temporal autocorrelation of the y‑axis acceleration, and core statistical moments of the x‑axis acceleration, providing a balanced view of magnitude and dynamics across activities.

**8. Pruning Action**  
All 256 attributes not in the selected set (plus `target`) were removed using **attribute_pruning_tool**. This leaves a manageable, high‑impact feature space while preserving predictive performance.

**9. Conclusions**  
- The full attribute set yields high accuracy but contains many redundant, low‑importance features.  
- A **6‑feature** subset achieves **97.5 %** accuracy, identical to the full model, and a **10‑feature** subset reaches **100 %** accuracy.  
- Pruning to this compact set simplifies downstream modeling, reduces computation, and mitigates over‑fitting risk without sacrificing performance.  

**Next Steps for the Team**  
- Communicate the selected 6 (or 10) features to the Scientist and Extractor agents for focused hypothesis generation and extraction.  
- Optionally explore model interpretability (e.g., SHAP) on the compact set for deeper insight into class‑specific behavior.  