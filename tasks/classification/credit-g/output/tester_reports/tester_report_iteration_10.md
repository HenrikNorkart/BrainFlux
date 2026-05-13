**Tester Agent Report – Feature Evaluation for the credit‑g Dataset**

---

### 1. Experimental Setup  
- **Model**: `GradientBoostingClassifier` (200 estimators, learning_rate = 0.1, max_depth = 5, random_state = 42).  
- **Data split**: 80 % training / 20 % test, stratified on the target.  
- **Metrics** (on the hold‑out set):  
  - **Accuracy**: **0.735**  
  - **AUC**: **0.741**  

These baseline results provide a reference for assessing individual feature contributions.

---

### 2. Feature‑Importance Findings  

| Rank | Feature (Model‑gain) | Gain % | Permutation AUC Δ |
|------|----------------------|--------|-------------------|
| 1 | `combined_exposure_score_div_checking_status_score_feat` | 7.87 % | +0.028 |
| 2 | `age_times_credit_amount_per_month` | 5.45 % | +0.0367 |
| 3 | `combined_exposure_div_checking_status_score` | 5.43 % | +0.0245 |
| 4 | `credit_amount_x_installment_commitment` | 3.16 % | +0.0106 |
| 5 | `credit_amount_x_duration` | 2.99 % | – |
| 6 | `has_other_payment_plan_times_credit_amount_per_month` | 2.72 % | +0.0232 |
| 7 | `residence_since_div_age` | 2.72 % | – |
| 8 | `job_score_times_credit_amount_per_month` | 2.58 % | +0.0081 |
| 9 | `age_times_employment_years_score` | 2.47 % | – |
|10 | `credit_utilisation` | 2.42 % | – |

*The permutation‑importance list largely confirmed the same top predictors, with `age_times_credit_amount_per_month` showing the strongest impact on AUC when shuffled.*

---

### 3. Inter‑Feature Relationships  

- Several raw variables (`checking_status_score`, `combined_exposure_score`, etc.) are **highly correlated** (≥ 0.80) with multiple engineered interaction terms.  
- Example high‑correlation pairs (sample):  

  - `checking_status_score` ↔ `combined_exposure_score` (ρ = 0.84)  
  - `checking_status_score` ↔ `log_combined_exposure_score` (ρ = 0.85)  
  - `checking_status_score` ↔ `combined_exposure_bin` (ρ = 0.90)  

These redundancies suggest that the engineered features capture most of the information present in the original raw scores.

---

### 4. Low‑Impact Features  

A full‑importance scan identified **29 attributes** with importance < 0.001 (i.e., negligible contribution). They are primarily basic categorical encodings or simple flags, e.g.:

- `checking_status_score`  
- `savings_status_score`  
- `is_male`  
- `is_foreign_worker`  
- `has_other_payment_plan`  
- `combined_exposure_score` (raw version)  

These were **pruned** using the `attribute_pruning_tool` to reduce noise and keep the feature set manageable.

---

### 5. Summary & Conclusions  

1. **Predictive Power** – The current engineered feature set yields modest predictive performance (Acc ≈ 73 %, AUC ≈ 0.74).  
2. **Key Predictors** – Interaction terms that combine credit amount, duration, installment commitment, and checking‑account exposure dominate both model‑gain and permutation importance.  
3. **Redundancy** – High correlations between raw scores and engineered variants indicate the engineered features subsume the raw information, justifying removal of the raw counterparts.  
4. **Pruned Features** – 29 low‑impact attributes were removed, streamlining the dataset without degrading the baseline metrics.  

The remaining feature collection is concise, largely composed of high‑impact engineered attributes, and ready for the next research iteration (e.g., deeper modeling or further robustness testing).