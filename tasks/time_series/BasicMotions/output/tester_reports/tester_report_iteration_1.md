**Comprehensive Feature Evaluation Report**

**1. Dataset Overview**  
- 16 raw sensor‑derived attributes (accelerometer & gyroscope statistics) plus the target class.  
- Classes: *walking, resting (standing), running, badminton*.  
- Small sample size (≈ 10 seconds per recording, 5 repetitions per activity).

**2. Baseline Model (All Features)**  
- **Model:** XGBoost (multiclass, `device="cuda:5"`, `tree_method="hist"`).  
- **Train‑test split:** 80/20 stratified.  
- **Accuracy:** **100 %** on the held‑out set.  
- **Top‑20 Feature Gains:**  
  1. `acc_mag_std`  
  2. `acc_x_std`  
  3. `acc_x_max`  
  4. `acc_x_min`  
  5. `acc_x_mean` …  

**3. Correlation & Redundancy Analysis**  
- Pairwise absolute Pearson correlations > 0.9 were abundant (e.g., `acc_x_std` ↔ `acc_x_mean`, `acc_x_std` ↔ `acc_x_iqr`, `acc_mag_mean` ↔ `acc_x_std`, etc.).  
- Many *acc_x* and *acc_y* statistics are almost interchangeable.  
- `acc_x_skew` and `acc_x_kurtosis` are only moderately correlated with the others (≈ 0.86), offering complementary information.

**4. Feature Pruning**  
- **Retained features:** `acc_x_std`, `acc_mag_std`, `acc_x_skew`, `acc_x_kurtosis`.  
- **Pruned (12 attributes):** `acc_x_mean`, `acc_x_min`, `acc_x_median`, `acc_x_iqr`, `acc_x_max`, `acc_y_mean`, `acc_y_std`, `acc_y_min`, `acc_y_max`, `acc_y_median`, `acc_y_iqr`, `acc_mag_mean`.  
- Rationale: high redundancy with retained features and low incremental gain.

**5. Model Performance with Pruned Feature Set**  
- **Accuracy (80/20 split):** **100 %** – identical to the full‑feature model.  
- **Cross‑validation (4‑fold, stratified):**  
  - Mean accuracy **92.5 %** (std 0.083).  
  - Fold accuracies: [0.80, 1.00, 0.90, 1.00].  
- **Robustness Test:** Added Gaussian noise (σ = 0.1) to the test set; accuracy remained **100 %**.

**6. Key Findings**  
- **Predictive Power:** A minimal set of four features captures virtually all discriminative information for the four activity classes.  
- **Feature Importance:** `acc_mag_std` and `acc_x_std` are the strongest predictors; skewness and kurtosis add complementary shape information.  
- **Statistical Redundancy:** Over 70 % of the original attributes are highly collinear and do not improve model performance.  
- **Robustness:** The reduced feature set is resilient to modest measurement noise.  
- **Model Stability:** Cross‑validation indicates good but not perfect stability, reflecting the limited data size; nevertheless, the reduced set consistently yields high accuracy.

**7. Recommendations (Testing‑only)**  
- **Adopt the pruned feature set** for downstream classification pipelines to reduce computational load and simplify model interpretation.  
- **Monitor** performance on larger, more diverse datasets to confirm that the four‑feature representation scales.  

*All evaluations were performed autonomously using the provided tools, adhering strictly to the task scope (no additional feature engineering or preprocessing beyond the given attributes).*