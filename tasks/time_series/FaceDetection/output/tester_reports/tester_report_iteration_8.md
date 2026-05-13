**Tester‑Agent Report – Feature Evaluation for the *FaceDetection* MEG dataset**

---

### 1.  Experimental Setup
| Item | Detail |
|------|--------|
| **Target** | `target` (binary: Face = 1, Scramble = 0) |
| **Model** | `xgboost.XGBClassifier` – `objective='binary:logistic'`, `eval_metric='auc'`, `tree_method='hist'`, `device='cuda:5'`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |
| **Validation** | 5‑fold stratified‑K‑fold (shuffle = True, seed = 42) |
| **Metrics** | ROC‑AUC (primary), feature‑gain importance, Pearson‑correlation, robustness to Gaussian noise (σ = 0.01) |
| **Pruning** | Features with *zero* gain (115 features) were removed via `attribute_pruning_tool`. |

---

### 2.  Predictive Performance  

| Model version | Mean 5‑fold AUC |
|---------------|-----------------|
| **All features (≈ 1 862)** | **0.699 ± 0.018** |
| **After zero‑gain pruning** | 0.699 (identical – no loss) |
| **Noise‑perturbed features** | **0.704 ± 0.017** (slightly higher → model is robust) |

*Interpretation*: The current feature set yields moderate discriminative power (AUC ≈ 0.70). Adding tiny random noise does **not** degrade performance, indicating resilience to minor measurement errors.

---

### 3.  Feature‑Importance Highlights  

The top‑20 features by **gain** (sum of split improvements) are:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `ch1_min` | 16.59 |
| 2 | `ch14_late_mean` | 15.24 |
| 3 | `ch92_skew` | 15.07 |
| 4 | `corr_ch124_ch125` | 14.45 |
| 5 | `ch124_m170_peak_amp` | 12.60 |
| 6 | `ch60_ptp` | 11.74 |
| 7 | `ch4_late_mean` | 11.73 |
| 8 | `ch117_max` | 11.46 |
| 9 | `ch50_late_mean` | 11.36 |
|10 | `ch104_mean` | 11.08 |
|11 | `ch79_early_mean` | 10.69 |
|12 | `ch53_max` | 10.49 |
|13 | `ch82_late_mean` | 10.39 |
|14 | `ch122_latency` | 10.30 |
|15 | `ch38_mean` | 10.30 |
|16 | `ch64_var` | 10.11 |
|17 | `ch22_m170_latency` | 9.80 |
|18 | `ch56_max` | 9.70 |
|19 | `ch9_early_mean` | 9.69 |
|20 | `ch1_var` | 9.69 |

*Key observations*  

* **Channel‑level statistics** (min, max, late‑mean, ptp) dominate importance – they capture amplitude and variability differences between face and scrambled trials.  
* **Spectral/skewness measures** (`ch92_skew`, `ch92_skew`) are also highly informative, suggesting shape of the power distribution matters.  
* **Inter‑channel correlation** (`corr_ch124_ch125`) appears as a top predictor, indicating that relationships between neighboring sensors add discriminative information.  
* **Peak‑amplitude of the M170 component** (`ch124_m170_peak_amp`) – a well‑known face‑sensitive ERP – ranks among the best features, confirming neuro‑physiological relevance.

---

### 4.  Redundancy & Correlation  

* Among the top‑20 features **no pair shows Pearson |r| > 0.9** (the highest observed correlation is ≈ 0.71 between `ch124_m170_peak_amp` and `ch124_mean`).  
* Therefore, the most important features are not strongly collinear and can be safely retained together.

---

### 5.  Feature Pruning Results  

* **Zero‑gain features**: 115 attributes never contributed to any split (e.g., `ch35_var`, `ch84_kurt`, `connectivity_dummy`, `avg_node_degree_est`, etc.).  
* After removing all zero‑gain attributes the AUC **did not change** (0.699 → 0.699).  
* This reduces the feature space from **≈ 1 862** to **≈ 1 747** attributes, easing memory and computation without hurting accuracy.

*Further pruning*: Keeping only the **top‑100** gain features (≈ 5 % of the original set) yields an AUC of **≈ 0.682**, a modest drop that may be acceptable for very lightweight models.

---

### 6.  Robustness Checks  

| Perturbation | AUC change |
|--------------|------------|
| Add Gaussian noise σ = 0.01 to all features | +0.005 (0.704) – model remains stable |
| Remove all zero‑gain features | 0 (no change) |
| Sub‑sample 70 % of features randomly (repeat 5×) | AUC variation ± 0.012 – indicates moderate sensitivity to feature set size but no catastrophic failure |

---

### 7.  Recommendations for the Scientist Agent  

1. **Focus on the top‑20–30 gain features** (listed above) for downstream hypothesis generation – they capture the strongest discriminative patterns.  
2. **Investigate the M170 peak amplitude and latency** (`*_m170_peak_amp`, `*_m170_latency`) across channels; they already rank high and are neuro‑physiologically meaningful.  
3. **Explore inter‑channel correlations** (e.g., `corr_ch124_ch125`) as potential network‑level biomarkers.  
4. **Consider dimensionality reduction** (e.g., keep the top 200 gain features) to obtain a manageable attribute set for the Extractor Agent while preserving most predictive power.  
5. **Validate on a leave‑one‑subject‑out scheme** (the competition’s intended protocol) to confirm that the observed AUC generalises across subjects.

---

### 8.  Action Taken  

* Identified and pruned 115 zero‑importance attributes via `attribute_pruning_tool`.  
* Documented baseline performance, importance ranking, redundancy analysis, and robustness outcomes.  

*The feature set now consists of ~1 747 high‑potential attributes, with a clear shortlist of the most predictive ones ready for further scientific exploration.*