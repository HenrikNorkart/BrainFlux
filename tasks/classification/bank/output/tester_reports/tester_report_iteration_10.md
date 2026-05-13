**Comprehensive Feature Evaluation Report – Bank Marketing Classification**

---

### 1. Predictive Power  
- **Model:** XGBoost (binary:logistic, 200 trees, max_depth = 6, hist‑tree method).  
- **Evaluation:** 5‑fold stratified cross‑validation.  
- **Metric (ROC‑AUC):** **0.921 ± 0.0037** (mean ± std).  
- **Interpretation:** The current feature set yields strong discriminative ability for predicting whether a client subscribes to a term deposit.

### 2. Feature Importance (Gain)  
Top‑10 features by XGBoost gain (most contribution to loss reduction):

| Rank | Feature | Gain |
|------|----------------------------------------------|--------|
| 1 | **high_imp_pca1_log_balance_interaction** | 0.273 |
| 2 | **comp_pdays_sq_interaction** | 0.111 |
| 3 | **high_imp_pca1_month_freq_interaction** | 0.036 |
| 4 | **housing_loan_binary** | 0.035 |
| 5 | **high_imp_interaction_pca1_squared** | 0.034 |
| 6 | **contact_freq** | 0.027 |
| 7 | **age_duration_prev_success** | 0.025 |
| 8 | **age_dur_prev_success_composite** | 0.020 |
| 9 | **comp_contact_freq_interaction** | 0.015 |
|10 | **core_composite_marital_single_interaction** | 0.010 |

*Observation:* The most influential attributes are engineered interaction terms that combine principal‑component‑derived signals (e.g., balance, month, pdays) with other variables, indicating that non‑linear relationships drive performance.

### 3. Low‑Importance Features & Pruning  
- **Low‑gain threshold:** gain < 1e‑4.  
- **Identified low‑gain attributes (12 total):**  
  `job_freq, high_imp_pca1_log_shift, high_imp_pca1_day_interaction, high_imp_pca1_sqrt_shift, high_imp_interaction_composite2, groupby_size_test, high_imp_pca1_poutcomefreq_interaction, test_agg_size, age_dur_success_default_interaction, test_feature, high_imp_pca1_squared, high_imp_pca1_default_interaction`.

- **Action:** Pruned these attributes using the `attribute_pruning_tool`.

### 4. Impact of Pruning  
- **Features remaining:** 115 (down from 127).  
- **Post‑pruning CV AUC:** **0.92096 ± 0.00369** – virtually identical to the pre‑pruning score (difference ≈ 0.00004).  
- **Conclusion:** The removed features contributed negligibly; pruning improves model parsimony without sacrificing predictive ability.

### 5. Statistical Relationships (Brief)  
- Correlation analysis (not shown) revealed that many top interaction features are only weakly correlated with each other (|ρ| < 0.3), reducing redundancy.  
- Low‑gain features exhibited higher inter‑correlations among themselves, suggesting they offered overlapping, non‑informative information.

### 6. Key Take‑aways  

| Aspect | Finding |
|--------|----------|
| **Predictive strength** | High (AUC ≈ 0.92). |
| **Most valuable features** | Interaction terms involving PCA‑derived balance, month frequency, pdays, and housing loan. |
| **Redundant/useless features** | 12 low‑gain engineered attributes (mostly isolated interaction terms). |
| **Effect of pruning** | No measurable loss in performance; model becomes leaner (‑12 features). |
| **Recommendation** | Retain the top‑gain interaction features; discard the low‑gain list identified above. |

---

**Prepared by:** Tester Agent – Feature Evaluation Loop.