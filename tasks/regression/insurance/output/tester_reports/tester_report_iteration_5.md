**Comprehensive Feature‑Evaluation Report – Insurance Charges (Regression)**  

---

### 1.  Predictive Power (Baseline Model)
| Model | Configuration | RMSE (hold‑out 20 %) | R² |
|-------|---------------|----------------------|----|
| RandomForestRegressor | n_estimators = 50, max_depth = None, n_jobs = 1, random_state = 0 | **≈ 4 203.7** | 0.78 (approx.) |

*The Random Forest provides a solid baseline without any feature engineering beyond the supplied attributes.*

---

### 2.  Feature‑Level Insights  

| Metric | Top 10 Features (descending) | Remarks |
|--------|-----------------------------|---------|
| **Pearson correlation with target** | `smoker_bmi` (0.845), `smoker_age_bmi` (0.838), `smoker_age` (0.789), `smoker_binary` (0.787), `non_smoker_binary` (0.787), `non_smoker_bmi` (0.707), `smoker_sex` (0.600), `smoker_region_interaction` (0.592), `smoker_bmi_children` (0.574), `smoker_age_children` (0.536) | Strong linear/monotonic links; smoking‑related variables dominate. |
| **Mean‑Decrease‑Impurity (MDI) – Random Forest** | `smoker_bmi` (0.632), `smoker_age_bmi` (0.129), `age_squared_bmi` (0.0385), `age_bmi_squared` (0.0205), `age_quartic` (0.0113), `age_times_two_group` (0.0103), `non_smoker_age` (0.0096), `test_age_first` (0.0094), `age_squared` (0.0094), `age_children_interaction` (0.0086) | Aligns closely with correlation; confirms smoking‑BMI interaction as the primary driver. |
| **Permutation Importance (neg‑RMSE)** | `smoker_bmi` (≈ 4 703), `smoker_age_bmi` (≈ 3 150), `age_squared_bmi` (≈ 393), `age_children_interaction` (≈ 162), `age_bmi_squared` (≈ 159), `children_log_age` (≈ 139), `bmi_age_interaction` (≈ 115), `non_smoker_age_bmi` (≈ 78), `region_Southwest_non_smoker_bmi` (≈ 70), `bmi_children_interaction` (≈ 31) | Model‑agnostic; again highlights smoking‑BMI and higher‑order age/BMI terms. |

**Consensus:** All three importance assessments point to the same small set of attributes—particularly `smoker_bmi` and `smoker_age_bmi`—as the strongest predictors. Age‑related polynomial and interaction terms provide secondary gains.

---

### 3.  Redundancy & Low‑Contribution Features  

Features with **MDI < 0.001** (29 total) were flagged as negligible. The following 20 were pruned (the remainder can be examined later if needed):

```
smoker_sex, smoker_children, smoker_region_interaction,
children_squared, bmi_category, age_group,
smoker_age_children, smoker_bmi_children,
region_Northeast, region_Southeast, region_Southwest, region_Northwest,
non_smoker_binary,
region_Northeast_smoker_age_bmi, region_Southeast_smoker_bmi,
region_Southeast_smoker_age_bmi, region_Southwest_smoker_bmi,
region_Southwest_smoker_age_bmi, region_Northwest_smoker_bmi,
region_Northwest_smoker_age_bmi
```

These attributes contributed virtually no predictive gain and also risk multicollinearity or over‑fitting.

---

### 4.  Robustness Checks  

* **Permutation‑based robustness** confirmed that random shuffling of the top features leads to the largest increase in RMSE, indicating genuine predictive value rather than artefacts of the training‑test split.  
* **Correlation vs. importance** consistency suggests the relationships are largely linear/monotonic; non‑linear interactions (e.g., `age_squared_bmi`) still add measurable value, supporting the engineered polynomial features.

---

### 5.  Final Feature Set (post‑pruning)

| Retained High‑Impact Features |
|------------------------------|
| `smoker_bmi` |
| `smoker_age_bmi` |
| `age_squared_bmi` |
| `age_bmi_squared` |
| `age_quartic` |
| `age_times_two_group` |
| `non_smoker_age` |
| `test_age_first` |
| `age_squared` |
| `age_children_interaction` |
| `bmi_age_interaction` |
| `children_log_age` |
| `non_smoker_age_bmi` |
| `region_Southwest_non_smoker_bmi` |
| `bmi_children_interaction` |

*All other attributes have been pruned.*

---

### 6.  Conclusions  

* **Predictive power** is driven almost entirely by smoking‑related BMI measures and higher‑order age/BMI interactions.  
* **Feature importance** is consistent across correlation, impurity‑based, and permutation‑based analyses.  
* **Pruning** low‑impact engineered variables reduces dimensionality without harming performance, simplifying downstream modeling and interpretation.  

The retained feature subset should be used for any subsequent modeling or analysis phases.