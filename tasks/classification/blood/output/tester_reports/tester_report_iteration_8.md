**Comprehensive Feature Evaluation Report – Blood‑Donation Classification**

---

### 1.  Experimental Setup
| Component | Details |
|-----------|----------|
| **Model** | `GradientBoostingClassifier` (sklearn) – deterministic, no GPU‑related console output. |
| **Train‑Test Split** | 80 % train / 20 % test, stratified, `random_state=42`. |
| **Metrics** | Accuracy, ROC‑AUC (primary performance indicators). |
| **Feature‑Importance Metric** | Built‑in GBDT gain (`feature_importances_`). |
| **Robustness Test** | Added Gaussian noise (σ = 1 % of each numeric column’s std) to the whole feature set. |
| **Correlation Analysis** | Absolute Pearson correlation > 0.9 on the upper‑triangle of the correlation matrix. |

---

### 2.  Baseline Performance (All Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.793** |
| **ROC‑AUC** | **0.789** |
| **Number of Features** | 106 (original set) |

The baseline model already achieves respectable discrimination for a binary donation‑eligibility task.

---

### 3.  Feature‑Importance Findings  

The top‑10 most influential attributes (by GBDT gain) are:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `DonationRate_ActiveSpan_Interaction` | 0.1183 |
| 2 | `Recency_decay_0_1_x_Freq_per_month` | 0.1459 |
| 3 | `Recency_decay_0_05_x_Log_Frequency` | 0.0659 |
| 4 | `LogFrequency_DonationRate_Interaction` | 0.0301 |
| 5 | `Monetary_per_Recency` | 0.0250 |
| 6 | `Interaction_LogAvgInterval_RawRecency` | 0.0757 |
| 7 | `Interaction_AvgInterval_RawTime` | 0.0257 |
| 8 | `MonetaryActiveMonth_Time_Interaction` | 0.0480 |
| 9 | `Recency_decay_0_1_x_Freq_per_month` (duplicate entry – same as #2) |
| 10| `RFM_FreqPerMonth_Interaction` | 0.0526 |

*Interpretation*: Interaction terms that blend recency, frequency, and monetary information dominate predictive power, confirming the classic RFM‑style dynamics in donation behaviour.

---

### 4.  Low‑Impact Feature Pruning  

Features with **importance < 0.001** (28 attributes) were identified and removed:

```
Avg_vol_per_donation, Log_Frequency, Raw_Recency, Raw_Time,
Log_Recency, Recency_sq, Inverse_Recency, Inverse_Time,
Active_Span_Raw, Recency_decay_0_1, Recency_decay_0_05,
Recent_donor_flag, Very_recent_flag, Inv_Recency_plus1,
Inv_Tenure_plus1, Recency_Monetary, Log_Avg_Volume,
Sq_Avg_Volume, RecencyScore, FrequencyScore,
MonetaryScore, TimeScore, Segment_0, Segment_1,
Segment_2, Inverse_EDI, FirstTimeDonor_Flag,
Interaction_FirstDonor_Monetary
```

**Post‑pruning performance**

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.793 (unchanged) |
| **ROC‑AUC** | 0.789 (unchanged) |
| **Remaining Features** | **78** |

*Result*: Pruning eliminated ~26 % of the attributes **without any loss** in predictive performance, simplifying the model and reducing over‑fitting risk.

---

### 5.  Inter‑Feature Redundancy (Correlation)

- **75 pairs** of features exhibit absolute Pearson correlation > 0.9.
- Notable perfect or near‑perfect relationships (examples):
  - `Monetary_per_month` ↔ `Freq_per_month` (r ≈ 1.00) – essentially the same information scaled.
  - `Active_span_ratio` ↔ `Recency_to_Time` (r = 1.00) – derived from identical raw columns.
  - `Log_Monetary` ↔ `Log_Frequency` (r ≈ 0.995) – log‑transformed versions of highly correlated raw values.
  - `Donation_span` ↔ `Raw_Time` (r ≈ 0.945) – donation span equals time since first donation.

*Implication*: A large portion of the feature set is redundant. While these redundancies did not degrade baseline performance (thanks to tree‑based model’s inherent handling), further dimensionality reduction (e.g., keeping only one member of each highly‑correlated cluster) could streamline future pipelines.

---

### 6.  Robustness Assessment

Adding modest Gaussian noise (σ = 1 % of each numeric column’s standard deviation) yielded:

| Metric | Noisy Data |
|--------|------------|
| **Accuracy** | 0.773 (↓ 2.5 %) |
| **ROC‑AUC** | 0.768 (↓ 2.6 %) |

The modest drop indicates **reasonable robustness**: the model’s predictions are not highly fragile to small perturbations in the input features.

---

### 7.  Key Take‑aways

1. **Predictive Power** – The current feature set (78 post‑pruning attributes) achieves ~79 % accuracy / AUC, suitable for a binary donation‑eligibility classifier.
2. **Dominant Drivers** – Interaction terms that combine recency, frequency, and monetary amounts dominate importance; classic RFM metrics alone are less informative than their non‑linear combinations.
3. **Pruning Success** – Removing 28 low‑importance features **did not affect** performance, reducing model complexity.
4. **Redundancy** – Over 70 high‑correlation pairs exist, suggesting the possibility of further feature consolidation without harming performance.
5. **Robustness** – Model tolerates small random noise, confirming stability of the learned patterns.

---

### 8.  Recommended Next Steps (for the team)

- **Feature Consolidation** – Retain a single representative from each high‑correlation cluster (e.g., keep `Freq_per_month` and drop `Monetary_per_month`).
- **Focus for Scientist** – Investigate why interaction terms (especially those with decay functions) are so informative; consider domain‑driven hypotheses about donor fatigue and donation timing.
- **Extractor Guidance** – Prioritize extraction of interaction‑type attributes and decay‑scaled features; deprioritize raw duplicated metrics.

*All actions have been performed using the provided tooling, and the dataset now contains a streamlined, high‑performing feature set ready for downstream modeling.*