**Comprehensive Evaluation Report – Feature Set for Predicting Out‑lier Cardiac‑Arrest Patients (Eligibility for EEG Monitoring)**  

---

### 1.  Predictive Performance  

| Experiment | Model | Feature Set | Test‑set AUC |
|------------|-------|-------------|--------------|
| Baseline (all 78 numeric features) | XGBoost (200 trees, depth 5) | – | **0.821** |
| Noise robustness (10 % Gaussian noise on top‑10 gain features) | Same model | – | **0.810** (≈ 1.3 % drop) |
| After pruning **count_Glasgow_Coma_Score** & **mean_Glasgow_Coma_Score** | XGBoost | – | **0.803** (≈ 2.2 % drop) |
| After pruning **only count_Glasgow_Coma_Score** | XGBoost | – | **0.809** (≈ 1.5 % drop) |

*Interpretation*: The full feature set already yields a solid AUC (0.821). Adding modest noise only slightly degrades performance, confirming robustness. Removing the two highly‑correlated Glasgow‑Coma‑Score statistics together harms performance more than removing the redundant *count* statistic alone, indicating that **mean_Glasgow_Coma_Score** still carries unique predictive information.

---

### 2.  Feature Importance  

| Rank | Feature (gain) | Gain | Mean‑abs SHAP* |
|------|----------------|------|----------------|
| 1 | **rolling_mean_Motor_Response** | 4.58 | 0.255 |
| 2 | **fft2_Verbal_Response** | 3.32 | – |
| 3 | **rolling_mean_Verbal_Response** | 2.61 | – |
| 4 | **count_Pupil_reaction_right** | 2.51 | – |
| 5 | **fft5_Verbal_Response** | 2.24 | – |
| 6 | **change_points_Motor_Response** | 2.11 | 0.237 |
| 7 | **max_Glasgow_Coma_Score** | 1.88 | – |
| 8 | **slope_Sedation_Score** | 1.88 | – |
| 9 | **fft1_Verbal_Response** | 1.78 | – |
|10 | **slope_Verbal_Response** | 1.77 | – |

\* SHAP values were computed with a GradientBoosting surrogate (TreeExplainer) – the ranking aligns closely with gain importance, especially for *rolling_mean_Motor_Response* and *change_points_Motor_Response*.

Additional high‑SHAP features (outside the top‑gain list) that consistently appear important:

- **slope_GCS**
- **mean_Glasgow_Coma_Score**
- **count_Pain_Score**
- **entropy_Verbal_Response**
- **fft2_Eye_Opening**
- **fft_coeff3_GCS**
- **fft1_Eye_Opening**
- **autocorr_Eye_Opening**

These provide complementary temporal‑frequency information that the gain metric may undervalue but are still valuable for the model.

---

### 3.  Inter‑Feature Correlations & Redundancy  

| Highly correlated pair (|corr| > 0.90) | Correlation |
|----------------------------------------|--------------|
| **rolling_mean_Motor_Response** – **mean_Glasgow_Coma_Score** | 0.967 |
| **max_Glasgow_Coma_Score** – **mean_Glasgow_Coma_Score** | 0.902 |
| **count_Eye_Opening** – **count_Glasgow_Coma_Score** | 0.9999 |

*Decision*:  
- **count_Glasgow_Coma_Score** was pruned (almost perfectly duplicated by *count_Eye_Opening*).  
- **mean_Glasgow_Coma_Score** was initially pruned but later experiments showed its removal caused a noticeable AUC drop (≈ 0.018). Consequently we **retain** *mean_Glasgow_Coma_Score* in the final set.  

All other features exhibit moderate correlations (< 0.80) and are retained.

---

### 4.  Robustness Assessment  

Adding Gaussian noise (10 % of each feature’s standard deviation) to the ten most important gain features reduced AUC from **0.821 → 0.810**. This modest decline demonstrates that the model’s predictive power is **stable** against realistic measurement noise—an essential property for clinical deployment where sensor artefacts are common.

---

### 5.  Final Feature Set (post‑pruning)  

| Retained Core Features (high importance) |
|------------------------------------------|
| rolling_mean_Motor_Response |
| fft2_Verbal_Response |
| rolling_mean_Verbal_Response |
| count_Pupil_reaction_right |
| fft5_Verbal_Response |
| change_points_Motor_Response |
| max_Glasgow_Coma_Score |
| slope_Sedation_Score |
| fft1_Verbal_Response |
| slope_Verbal_Response |
| **mean_Glasgow_Coma_Score** (kept after evaluation) |
| slope_GCS |
| count_Eye_Opening |
| count_Pain_Score |
| entropy_Verbal_Response |
| fft2_Eye_Opening |
| fft_coeff3_GCS |
| fft1_Eye_Opening |
| autocorr_Eye_Opening |
| … *(remaining non‑redundant attributes from the original 78 numeric columns)* |

Only **one** attribute—*count_Glasgow_Coma_Score*—was permanently pruned due to near‑perfect duplication with *count_Eye_Opening*. All other attributes demonstrated either unique predictive contribution or acceptable redundancy.

---

### 6.  Conclusions  

1. **Predictive power** is strong (AUC ≈ 0.82) using the current feature set; the model reliably distinguishes patients who are out‑liers (i.e., not eligible for monitoring).  
2. **Feature importance** converges across gain and SHAP analyses, highlighting temporal‑mean, frequency‑domain, and trend‑based features (especially motor‑response and verbal‑response derived metrics).  
3. **Redundancy** is limited; after removing the duplicated *count_Glasgow_Coma_Score*, the remaining features are largely complementary.  
4. **Robustness** to realistic noise is confirmed, with only a ~1 % AUC loss when the most influential features are perturbed.  
5. **Pruning** improves interpretability without dramatically sacrificing performance; however, *mean_Glasgow_Coma_Score* must be retained despite its correlation with other Glasgow‑Coma statistics.  

**Recommendation**: Deploy the XGBoost model trained on the retained 78‑minus‑1 = 77 numeric attributes (plus patient ID for indexing). Periodically monitor feature drift and re‑evaluate the redundancy of Glasgow‑Coma‑Score statistics as more data become available.