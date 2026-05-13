**Feature Evaluation Report – Credit‑G Dataset**

**1. Experimental Setup**  
- **Model:** GradientBoostingClassifier (200 trees, learning_rate 0.1, max_depth 3).  
- **Data Split:** 80 % train / 20 % test, stratified, random_state 42.  
- **Metrics:** Accuracy = 0.71, ROC‑AUC = 0.679.  
- **Importance Measures:**  
  - *Impurity‑based importance* from the fitted GBM.  
  - *Permutation importance* (10 repeats, ROC‑AUC scoring).  
- **Combined Score:** Normalised impurity + permutation scores, averaged to rank features.

**2. Key Findings**  

| Rank | Feature | Impurity % | Permutation % | Combined Score |
|------|---------|------------|---------------|----------------|
| 1 | **combined_exposure_score** | 37.2 % | 56.9 % | **0.471** |
| 2 | credit_amount_per_month | 8.3 % | 14.1 % | 0.112 |
| 3 | employment_years_score | 3.6 % | 13.5 % | 0.086 |
| 4 | total_credit_exposure | 10.4 % | 6.0 % | 0.082 |
| 5 | credit_history_score | 5.0 % | 10.0 % | 0.075 |
| … | … | … | … | … |
| **Low‑importance** (combined < 0.01) | high_installment_flag | 0.27 % | 1.03 % | 0.0065 |
|  | savings_status_score | 1.02 % | 0.24 % | 0.0063 |
|  | residence_years | 2.61 % | –3.17 % | –0.0028 |
|  | property_magnitude_score | 4.21 % | –5.55 % | –0.0067 |
|  | checking_property_interaction | 3.28 % | –7.48 % | –0.0210 |

- **Predictive Power:** The model’s moderate performance (AUC ≈ 0.68) indicates that the current feature set captures useful signal but leaves room for improvement.  
- **Feature Redundancy/Synergy:** Several low‑importance features show near‑zero or negative permutation impact, suggesting they either add noise or are highly correlated with higher‑ranked variables (e.g., *combined_exposure_score* already aggregates exposure‑related information).  

**3. Actions Taken**  
- Documented the experimental results and observations (see internal notes).  
- Pruned the following attributes due to negligible contribution:  
  `high_installment_flag, savings_status_score, residence_years, property_magnitude_score, checking_property_interaction`.  

**4. Recommendations for the Team**  
- **Focus future investigations** on the top‑ranked features, especially *combined_exposure_score* and derived credit‑amount metrics, as they drive most of the predictive signal.  
- **Consider robustness checks** (e.g., adding noise, cross‑validation) on the retained feature set to confirm stability.  
- **Avoid re‑introducing** the pruned attributes unless new evidence suggests they capture unique information not represented by existing variables.  

*End of report.*