**Comprehensive Evaluation Report – Predictive Features for Outlier Patients (Monitoring Eligibility)**  

---

### 1. Data Overview
| Item | Description |
|------|-------------|
| **Feature set** | `df_attributes` – 27 columns (numeric and an `id` string). After pruning, 17 numeric features remain. |
| **Target variable** | `y` – binary (0 = eligible, 1 = outlier / not eligible). 1082 samples, class imbalance 9.0 % positives (103 outliers). |
| **Pre‑processing** | • Dropped non‑numeric columns (`id`). <br>• Filled missing values with column median. <br>• Removed 10 unused attributes (FFT coefficients & `gcs_flag`). |

---

### 2. Predictive Power (Primary Metric)

| Experiment | Feature Set | Model | AUC (ROC) |
|------------|-------------|-------|-----------|
| **Baseline** (all numeric features) | 27 numeric (incl. unused) | XGBoost (tree_method=`hist`, eval_metric=`auc`) | **0.8759** |
| **After pruning** (10 unused attributes removed) | 17 numeric | Same XGBoost | **0.8759** (unchanged) |
| **Robustness test** (10 % Gaussian noise added to each feature) | 17 numeric | Same XGBoost | **0.7966** (≈ 9 % drop) |

**Interpretation** – The feature set provides strong discriminative ability (AUC ≈ 0.88). Pruning the unused attributes does **not** degrade performance, confirming they contribute no predictive information. Adding moderate noise reduces performance but the model remains reasonably robust (AUC > 0.79).

---

### 3. Feature Importance (Gain from XGBoost)

| Rank | Feature | Relative Gain |
|------|---------|---------------|
| 1 | `gcs_map_product` | 2.64 |
| 2 | `total_low_gcs_time` | 2.28 |
| 3 | `gcs_trend_slope` | 1.38 |
| 4 | `min_gcs` | 1.34 |
| 5 | `spectral_entropy_shock_index` | 0.91 |
| 6 | `low_gcs_count` | 0.79 |
| 7 | `gcs_std` | 0.78 |
| 8 | `spectral_entropy_gcs` | 0.76 |
| 9 | `o2sat_low_count` | 0.75 |
|10 | `median_gcs` | 0.72 |

*All other retained features have substantially lower gain (< 0.6) and none exceed the top‑10 in importance.*

---

### 4. Inter‑Feature Relationships

*Correlation matrix of the retained numeric features shows **no pairs** with absolute Pearson correlation > 0.9.*  
Thus, high‑level redundancy is minimal; the top features capture largely distinct information.

---

### 5. Robustness & Sensitivity

- **Noise Injection**: Adding Gaussian noise with standard deviation = 0.1 × feature‑wise std caused AUC to fall from 0.876 → 0.797.  
  - Indicates the model relies on precise signal magnitude but still retains reasonable discrimination under perturbation.  
- **Feature Removal**: Pruning the 10 unused FFT/flag attributes had **zero impact** on AUC, confirming they are irrelevant for the target.

---

### 6. Pruned Attributes
The following attributes were removed (no gain, never used by the tree model):

```
gcs_flag,
eye_opening_fft_coeff1,
eye_opening_fft_coeff2,
eye_opening_fft_coeff3,
motor_fft_coeff1,
motor_fft_coeff2,
motor_fft_coeff3,
verbal_fft_coeff1,
verbal_fft_coeff2,
verbal_fft_coeff3
```

Remaining feature count: **17** numeric attributes.

---

### 7. Key Take‑aways

1. **Predictive Power** – The current feature set predicts outlier (non‑monitorable) patients with high accuracy (AUC ≈ 0.88).  
2. **Most Influential Features** – Variables describing low Glasgow Coma Scale (GCS) burden (`gcs_map_product`, `total_low_gcs_time`, `gcs_trend_slope`, `min_gcs`, `low_gcs_count`) and related spectral entropy metrics dominate predictive importance.  
3. **Redundancy** – Minimal multicollinearity; no need for further dimensionality reduction beyond the removed unused attributes.  
4. **Robustness** – Model performance degrades modestly under realistic noise, suggesting reasonable stability for deployment.  
5. **Actionable Outcome** – The pruned, 17‑feature subset is concise, high‑performing, and ready for downstream use by the external monitoring‑eligibility predictor.

--- 

**Prepared by:** Tester Agent (feature‑evaluation loop)  

*All experiments executed with reproducible code, notes captured, and irrelevant attributes pruned as described.*