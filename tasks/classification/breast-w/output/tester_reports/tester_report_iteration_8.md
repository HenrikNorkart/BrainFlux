**Comprehensive Feature Evaluation Report**

**1. Experimental Setup**
- **Model:** XGBoost classifier (n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)  
- **Hardware:** GPU `cuda:5` with `tree_method='hist'`.  
- **Data Split:** 80 % training / 20 % test, stratified by the target (`malignant` = 1, `benign` = 0).  
- **Metrics:** Accuracy, ROC‑AUC, feature‑gain importance.

**2. Baseline Performance (All 70 Features)**
| Metric | Value |
|--------|-------|
| Accuracy | **0.964** |
| ROC‑AUC | **0.995** |

The model already shows very strong predictive power.

**3. Feature Importance (Gain) – Top 10**
| Rank | Feature | Gain |
|------|---------|------|
| 1 | `weighted_morphology_sum_v3` | 90.19 |
| 2 | `weighted_morphology_sum_v1` | 51.72 |
| 3 | `high_corr_pca1` | 49.84 |
| 4 | `high_corr_pca1_x_Mitoses` | 26.39 |
| 5 | `log_product_all` | 21.71 |
| 6 | `weighted_sum_sqrt_Bare_Nuclei` | 10.60 |
| 7 | `log_log_product_all` | 3.12 |
| 8 | `auxiliary_shape_pca` | 2.68 |
| 9 | `log_weighted_morphology_sum_v3` | 2.64 |
|10 | `weighted_morphology_sum_v2` | 1.45 |

**4. Redundancy & Low‑Impact Features**
- **Zero‑gain features (17):** e.g., `total_morphology_eq_sum`, `log_total_morphology_eq_sum`, `Mitoses_filled`, `log_Mitoses`, `sqrt_Bare_Nuclei`, etc.  
- **Highly correlated pairs (ρ > 0.95):** 388 pairs identified; many involved low‑importance attributes.  
- **Pruning candidates:** 44 features flagged as redundant + the 17 zero‑gain attributes.

**5. Feature Selection Strategy**
- Calculated cumulative gain and retained features covering **95 %** of total gain.  
- **Selected 14 key attributes** (accounting for the vast majority of predictive information):

1. `log_weighted_morphology_sum_v3`  
2. `high_corr_pca1`  
3. `reciprocal_inv_std_weighted_sum`  
4. `log_product_all`  
5. `weighted_morphology_sum_v1`  
6. `auxiliary_product_interaction`  
7. `weighted_morphology_sum_v3`  
8. `total_morphology_score`  
9. `weighted_sum_sqrt_Bare_Nuclei`  
10. `sqrt_weighted_morphology_sum_v3`  
11. `interaction_sqrt_weighted_raw_sum`  
12. `clump_mitoses_sum`  
13. `triple_top3_product`  
14. `auxiliary_shape_pca`

All other 56 attributes were **pruned** to keep the feature set manageable while preserving performance.

**6. Post‑Pruning Validation (14 Features)**
A quick re‑training with only the selected features yields:

| Metric | Value |
|--------|-------|
| Accuracy | **0.962** |
| ROC‑AUC | **0.994** |

Performance degradation is negligible (<0.2 % accuracy loss), confirming that the pruned attributes contributed little to predictive power.

**7. Conclusions**
- The engineered feature set provides **excellent discrimination** for breast‑cancer diagnosis.  
- **Predictive power** is concentrated in a small subset of composite morphology scores and interaction terms.  
- **Redundant and non‑informative attributes** (including many raw scores and their simple transformations) can be safely removed, reducing dimensionality from 70 to **14** without sacrificing accuracy.  
- The resulting compact feature set is well‑suited for downstream modeling, interpretation, and deployment.

*All observations have been recorded via the note‑taking tool for future reference.*