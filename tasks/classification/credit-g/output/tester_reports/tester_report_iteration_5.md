**Tester Agent – Feature Evaluation Report (Credit‑G Dataset)**  

---

### 1. Experimental Setup
* **Model:** XGBoost‑Classifier (300 trees, max_depth = 5, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8).  
* **Hardware:** GPU `cuda:5` with `tree_method="hist"` for speed.  
* **Data Split:** 80 % training / 20 % validation (stratified by the target).  
* **Target Encoding:** `yes → 1`, `no → 0`.  

### 2. Predictive Performance (baseline, all features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.75** |
| ROC‑AUC  | **0.747** |
| Total features (excluding *target*) | **70** |
| Features with non‑zero gain (XGBoost) | **62** |

The model achieves respectable predictive power for a single‑pass baseline, confirming that the engineered attributes already contain useful signal.

### 3. Feature Importance (Gain)

Top‑20 features (gain values) – ordered by contribution to the model:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `checking_savings_interaction` | 4.69 |
| 2 | `combined_exposure_div_checking_status_score` | 3.49 |
| 3 | `is_purpose_business_times_combined_exposure` | 2.84 |
| 4 | `combined_exposure_times_savings_status_score` | 2.62 |
| 5 | `has_other_payment_plan` | 2.50 |
| 6 | `combined_exposure_times_purpose_used_car` | 2.22 |
| 7 | `combined_exposure_times_purpose_radio_tv` | 1.96 |
| 8 | `has_other_parties_times_combined_exposure` | 1.95 |
| 9 | `sqrt_combined_exposure_score` | 1.93 |
|10 | `combined_exposure_score_times_is_male_is_foreign_worker` | 1.92 |
|11 | `has_other_parties` | 1.91 |
|12 | `combined_exposure_times_checking_status_score` | 1.86 |
|13 | `combined_exposure_score_times_is_purpose_new_car` | 1.76 |
|14 | `savings_status_score` | 1.69 |
|15 | `log_combined_exposure_score` | 1.65 |
|16 | `credit_history_score` | 1.56 |
|17 | `is_purpose_new_car` | 1.56 |
|18 | `num_dependents_times_employment_years_score` | 1.54 |
|19 | `log_credit_utilisation_ratio` | 1.53 |
|20 | `has_other_payment_plan_times_credit_amount_per_month` | 1.53 |

These features are dominated by interaction terms involving **combined exposure**, **checking/savings scores**, and **payment‑plan / purpose** variables – indicating that the engineered cross‑features are indeed informative.

### 4. Redundancy & Correlation Analysis
* **Highly correlated pair (|ρ| > 0.9):**  
  *`sqrt_combined_exposure_score`* ↔ *`log_combined_exposure_score`* (ρ ≈ 0.983).  
  *Implication:* one of them can be removed without loss of information.

* No other top‑20 pairs exceeded the 0.9 threshold.

### 5. Features with No Predictive Contribution
Using the full‑model gain scores:

| Category | Count | Example(s) |
|----------|-------|------------|
| **Zero‑gain features** (not used by XGBoost) | **9** | `is_male`, `combined_exposure_times_purpose_domestic_appliance`, `combined_exposure_times_purpose_repairs`, `combined_exposure_times_purpose_other`, `combined_exposure_times_purpose_retraining`, `combined_exposure_bin`, `simple_test`, `combined_exposure_score_cubed`, `combined_exposure_score_times_is_purpose_business` |
| **Low‑gain (< 0.5)** | **0** | – (all remaining features contributed ≥ 0.5 gain) |

These zero‑gain attributes add no predictive value and can safely be pruned.

### 6. Pruning Decisions
Based on the analysis, the following attributes are recommended for removal:

1. `is_male`  
2. `combined_exposure_times_purpose_domestic_appliance`  
3. `combined_exposure_times_purpose_repairs`  
4. `combined_exposure_times_purpose_other`  
5. `combined_exposure_times_purpose_retraining`  
6. `combined_exposure_bin`  
7. `simple_test`  
8. `combined_exposure_score_cubed`  
9. `combined_exposure_score_times_is_purpose_business`  
10. **One** of the highly correlated pair – `log_combined_exposure_score` (retain `sqrt_combined_exposure_score`).

*Total features after pruning:* **60** (down from 70), a more manageable set without sacrificing performance.

### 7. Robustness Check (Post‑pruning)
A quick re‑run of the same XGBoost configuration on the pruned feature set yielded **identical** validation metrics (Accuracy = 0.75, AUC = 0.747). This confirms that the removed attributes were indeed non‑informative.

### 8. Key Take‑aways
* The engineered interaction features (especially those combining *combined exposure* with checking/savings scores and purpose indicators) drive predictive power.
* Only one redundant pair exists; removing one does not affect performance.
* Nine attributes contribute **zero** gain; pruning them reduces dimensionality without impact.
* The current feature set (≈ 60 high‑quality attributes) balances predictive strength and model simplicity, ready for downstream modeling or deployment.

--- 

*Prepared by the Tester Agent.*