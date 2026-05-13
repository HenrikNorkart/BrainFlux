**Tester Agent – Feature‑Evaluation Report**  

---

### 1.  Experimental Design  

| Step | Methodology | Rationale |
|------|--------------|-----------|
| **Baseline model** | Stratified 5‑fold cross‑validation with XGBoost (multi‑class, `device="cuda:5"`, `tree_method="hist"`). | Provides an unbiased estimate of predictive power using the full 210‑feature set. |
| **Feature‑importance (model‑based)** | XGBoost **gain** importance (built‑in). | Directly measures each feature’s contribution to the reduction of loss during tree building. |
| **Statistical screening** | One‑way ANOVA **F‑test** (`sklearn.feature_selection.f_classif`). | Quantifies how strongly each feature separates the four activity classes, independent of any model. |
| **Feature selection** | Union of:  (i) all features with non‑zero gain (4 features) and (ii) the top‑30 features by F‑value. → **32 retained attributes**. | Combines model‑driven and statistical evidence to keep the most predictive and discriminative attributes while discarding the rest. |
| **Pruning** | `attribute_pruning_tool` removed the 178 attributes not in the union. | Reduces dimensionality and simplifies interpretation without sacrificing performance. |
| **Post‑pruning validation** | Same XGBoost 5‑fold CV on the reduced 32‑feature set. | Checks that predictive power is retained after pruning. |
| **Redundancy check** | Pair‑wise Pearson correlation (|r| > 0.9) among retained features. | Highlights highly correlated groups that could be merged in future work (not acted on here, per guidelines). |

---

### 2.  Results  

| Metric | Full‑Feature Model | Pruned‑Feature Model |
|--------|-------------------|----------------------|
| Mean accuracy (5‑fold) | **0.95** (folds: 0.75, 1.0, 1.0, 1.0, 1.0) | **0.95** (identical distribution of folds) |
| Number of features used | 210 (incl. target) | **32** (incl. target) |
| Top‑gain features (gain ≈ 3.0) | `acc_x_mean`, `acc_x_std`, `acc_x_max`, `acc_y_autocorr_lag1` | Same four appear in the retained set. |
| Top‑10 ANOVA F‑features | `acc_x_max`, `pc1_acc_stats`, `acc_x_std`, `acc_x_jerk_rms`, `acc_x_jerk_std`, `acc_x_jerk_mad`, `acc_x_jerk_sma`, `acc_mag_max`, `median_std_axes`, `acc_x_mad` | All are included among the 32 kept attributes. |
| High‑correlation pairs (|r| > 0.9) | 2 091 pairs (e.g., `acc_x_mean` vs. `gyro_z_std` = 0.96) | Same redundancy pattern persists in the reduced set. |

**Interpretation**

* The classifier already attains near‑perfect accuracy; therefore many raw sensor‑derived attributes are **redundant** or **uninformative**.
* Model‑based gain identifies only four features that actually drive the tree splits, confirming the dominance of basic acceleration statistics.
* The ANOVA screen reveals additional discriminative attributes (jerk statistics, magnitude‑based metrics, PCA components) that are statistically significant but not exploited by the default XGBoost configuration.
* Keeping the union of gain and top‑30 F‑features yields a **compact 32‑feature subset** that **preserves** the 0.95 accuracy while eliminating ~85 % of the original attributes.

---

### 3.  Actions Taken  

1. **Feature retention list (32 attributes)**  
   ```
   ['gyro_z_iqr','median_std_axes','acc_x_jerk_mad','acc_x_rms','overall_std_mean',
    'acc_x_iqr','acc_y_median','acc_mag_sma','std_ratio_x','acc_y_sma','acc_x_max',
    'gyro_z_mean_peak_amp','acc_x_sma','gyro_z_sma','acc_x_std','acc_mag_std',
    'acc_mag_mean','acc_mag_median','acc_x_jerk_sma','std_of_std_axes',
    'acc_x_jerk_std','acc_mag_rms','acc_iqr_mean','pc2_acc_stats',
    'acc_y_autocorr_lag1','acc_x_mean','pc1_acc_stats','acc_x_min',
    'acc_x_mad','acc_y_rms','acc_x_jerk_rms','acc_mag_max']
   ```
2. **Pruned** all other attributes via `attribute_pruning_tool`.  
3. **Validated** that the pruned dataset achieves the same mean accuracy (0.95) as the full set.  

---

### 4.  Key Take‑aways for the Team  

| Observation | Implication |
|-------------|-------------|
| **Predictive power is concentrated** in a handful of acceleration‑mean/std/max and a few jerk/magnitude statistics. | Future models can be built on this small subset, reducing training time and memory. |
| **Statistical F‑test highlights additional useful features** (jerk, PCA) that are not automatically selected by XGBoost. | If a different classifier (e.g., SVM, neural net) is later explored, these features may become more valuable. |
| **High inter‑feature correlations (>0.9)** are abundant among the retained set. | Though not required now, further dimensionality reduction (e.g., hierarchical clustering of features) could yield an even leaner representation. |
| **No loss in accuracy after pruning** demonstrates that the discarded 178 attributes are essentially noise for this task. | The dataset is now manageable for downstream analysis and for the Scientist/Extractor agents to focus on meaningful attributes. |

---

**End of Report**  

*All notes have been recorded via `take_note_tool` for reference.*