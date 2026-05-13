**Comprehensive Evaluation Report – Bank Marketing Feature Set**

---

### 1.  Overall Predictive Power  
| Metric | Value (All 69 features) | Value (After pruning – 61 features) |
|--------|------------------------|--------------------------------------|
| **Accuracy** | **0.8966** | **0.8967** |
| **ROC‑AUC** | **0.9090** | **0.9080** |
| **Weighted F1 (class 1)** | 0.468 | 0.477 |
| **Macro F1** | 0.706 | 0.710 |

*The engineered feature set yields strong predictive performance for the “yes/​no” term‑deposit subscription task. Pruning the low‑value/redundant attributes does **not** degrade accuracy or AUC, confirming that the remaining 61 features retain the essential signal.*

---

### 2.  Feature Importance (Gain – XGBoost)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | **age_dur_prev_success_composite** | 63.83 |
| 2 | **comp_pdays_sq_interaction** | 59.89 |
| 3 | **housing_loan_binary** | 44.52 |
| 4 | **contact_freq** | 22.78 |
| 5 | **core_composite_contactfreq_interaction** | 16.21 |
| 6 | **core_composite_housing_interaction** | 8.98 |
| 7 | **comp_housing_interaction** | 8.82 |
| 8 | **comp_day_interaction** | 8.25 |
| 9 | **comp_contact_freq_sq_interaction** | 7.91 |
|10 | **age_dur_success_job_freq** | 7.81 |
|…|…|…|

*The top‑5 features alone account for more than **150 gain units**, dominating the model’s decision‑making.*

---

### 3.  Low‑Impact / Redundant Attributes  

| Reason for Removal | Attributes |
|--------------------|------------|
| **Negligible gain (< 0.5)** | `test_agg_size`, `groupby_size_test`, `age_dur_success_default_interaction` |
| **Exact duplicate** | `job_frequency` (identical to `job_freq`) |
| **Highly correlated (> 0.95) interaction blocks** | `comp_contact_freq_interaction`, `comp_contact_freq_sq_interaction`, `core_interaction_composite` (all > 0.99 correlated with `core_composite_contactfreq_interaction`) |
| **Redundant balance‑interaction** | `age_dur_success_balance_cubed` (correlated 0.99 with `age_dur_success_balance_squared`) |

These attributes contributed little unique information and inflated the dimensionality.

---

### 4.  Correlation Insights  

- **Job frequency** duplicated (`job_frequency` ↔ `job_freq` – correlation = 1.0).  
- **Contact‑frequency interaction suite** showed correlations > 0.99 among four features, justifying the retention of the single most informative (`core_composite_contactfreq_interaction`).  
- **Age‑balance composites** (`age_balance_interaction`, `age_squared_balance`) were moderately correlated (≈ 0.96) but each retained a distinct gain contribution, so both were kept.  
- **Duration‑age composites** (`duration_age_composite`, `log_age_times_duration`, etc.) displayed high pairwise correlations (≈ 0.96) yet each possessed non‑trivial gain; they remain in the set.

---

### 5.  Robustness Check  

Re‑training after pruning confirmed **stable performance** (Δaccuracy = +0.0001, ΔAUC = ‑0.001). This demonstrates that the model is robust to removal of the identified low‑value features and that the remaining set is compact yet expressive.

---

### 6.  Final Feature Set  

- **Total retained attributes:** **61** (including the original engineered variables and the original raw ones that survived pruning).  
- **Key retained high‑impact features:** `age_dur_prev_success_composite`, `comp_pdays_sq_interaction`, `housing_loan_binary`, `contact_freq`, `core_composite_contactfreq_interaction`, plus the moderate‑importance age/balance and duration composites.

---

### 7.  Conclusions  

1. **Predictive Power:** The engineered feature set provides strong classification ability for the target variable.  
2. **Feature Importance:** A small subset (≈ 10 %) drives the majority of predictive gain.  
3. **Redundancy:** Several engineered features were either exact duplicates or near‑perfectly correlated; pruning them yields a leaner model without performance loss.  
4. **Actionable Outcome:** The pruned 61‑feature collection should be adopted for downstream modeling, offering efficiency gains (faster training/inference) while preserving accuracy and AUC.

*All pruning actions have been executed via the `attribute_pruning_tool` and the final assessment confirms the effectiveness of the reduced feature set.*