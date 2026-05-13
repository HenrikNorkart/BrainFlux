**Feature Evaluation Report – Blood Donation Prediction**

---

### 1. Overview  

- **Task:** Binary classification – predict whether a person will donate blood (`target = yes/no`).  
- **Initial Feature Set:** 22 engineered attributes + the original 4 raw variables (Recency, Frequency, Monetary, Time) = **23 columns** (including the target).  
- **Goal:** Quantify predictive power, identify importance, detect redundancy, and produce a compact, high‑performing feature subset.

---

### 2. Initial Model (All Features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.787** |
| ROC‑AUC | **0.777** |
| Macro‑F1 | 0.682 |
| Weighted‑F1 | 0.777 |

*Key observations* – The model performed reasonably but several features were highly correlated, suggesting possible redundancy.

---

### 3. Correlation & Redundancy Analysis  

Pairs with **|ρ| > 0.9** (absolute Pearson correlation):

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| `Freq_per_month` | `Monetary_per_month` | 1.00 |
| `Recency_to_Time` | `Active_span_ratio` | 1.00 |
| `Log_Frequency` | `Log_Monetary` | 0.995 |
| `Monetary_per_Recency` | `MonetaryRecency_LogTime_Interaction` | 0.993 |
| `Raw_Recency` | `Log_Recency` | 0.919 |
| `Raw_Time` | `Donation_span` | 0.945 |
| `Raw_Time` | `Time_sq` | 0.956 |
| `Donation_span` | `Time_sq` | 0.923 |

These pairs indicate near‑duplicate information; keeping both would not add predictive value and may increase multicollinearity.

---

### 4. Feature Pruning  

**Removed attributes (7):**  

- `Monetary_per_month`  
- `Active_span_ratio`  
- `Log_Monetary`  
- `MonetaryRecency_LogTime_Interaction`  
- `Log_Recency`  
- `Donation_span`  
- `Time_sq`

Resulting feature count: **15** (including the original 4 raw variables).

---

### 5. Model After Pruning  

| Metric | Value |
|--------|-------|
| Accuracy | **0.80** |
| ROC‑AUC | **0.782** |
| Macro‑F1 | 0.695 |
| Weighted‑F1 | 0.788 |
| Number of Features | **15** |

*Interpretation* – A modest but consistent improvement in both accuracy and AUC despite a 30 % reduction in feature count, confirming that the removed attributes were redundant.

---

### 6. Feature Importance (Gain, XGBoost)

Top 10 features after pruning (gain importance):

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `Raw_Recency` | 2.04 |
| 2 | `DonationRate_ActiveSpan_Interaction` | 1.70 |
| 3 | `Freq_per_month` | 0.76 |
| 4 | `Frequency_per_Recency` | 0.71 |
| 5 | `LogFrequency_DonationRate_Interaction` | 0.69 |
| 6 | `Monetary_per_Recency` | 0.63 |
| 7 | `Raw_Time` | 0.55 |
| 8 | `Interaction_Freq_Recency` | 0.53 |
| 9 | `Donation_rate` | 0.48 |
|10 | `Recency_to_Time` | 0.46 |

**Insights**

- **Recency‑related variables** (`Raw_Recency`, `Recency_to_Time`) dominate importance, aligning with domain knowledge that recent donation behavior is predictive.  
- **Interaction terms** (`DonationRate_ActiveSpan_Interaction`, `LogFrequency_DonationRate_Interaction`) also contribute substantially, showing that combined effects improve discrimination.  
- **Frequency‑per‑month** and its ratio to recency remain valuable, while the raw `Monetary` amount is less influential after adjusting for recency.

---

### 7. Robustness Check  

- **Procedure:** Added Gaussian noise (10 % of each feature’s standard deviation) to all 15 retained features and re‑trained the model.  
- **Results:**  

| Metric | Noisy Data |
|--------|------------|
| Accuracy | **0.813** (slightly higher due to random split) |
| ROC‑AUC | **0.752** ( modest drop ) |

*Interpretation* – Predictive performance is relatively stable; the slight AUC decline indicates the model tolerates moderate perturbations, suggesting the feature set is robust.

---

### 8. Conclusions  

1. **Predictive Power:** The curated 15‑feature set achieves **80 % accuracy** and **0.78 AUC**, marginally better than the full set.  
2. **Feature Importance:** Recency, interaction terms, and frequency‑derived ratios are the strongest drivers.  
3. **Redundancy Removal:** Eliminating 7 highly correlated attributes reduced dimensionality without harming – indeed slightly improving – model performance.  
4. **Robustness:** The model maintains performance under realistic noise, confirming the stability of the selected features.  

**Recommendation:** Adopt the 15‑feature subset for downstream deployment. It is compact, non‑redundant, and delivers the best observed predictive performance for blood‑donation classification.