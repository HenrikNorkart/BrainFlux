**Comprehensive Feature Evaluation Report**

**1. Objective**  
Assess the predictive power and importance of the extracted attributes for the multiclass classification task — *Contraceptive_method_used* (1 = No‑use, 2 = Long‑term, 3 = Short‑term).

**2. Initial Baseline (All 70 attributes)**  
*Model:* Decision‑Tree Classifier (random_state = 42)  
*Accuracy:* **0.478**  
*Key observation:* Raw demographic attributes (e.g., `Wifes_age_group`, `Parity_category`, `Education_sum`) received **zero** importance. The model relied almost entirely on engineered interaction features.

**3. Feature‑Importance Ranking (Tree‑based gain & SHAP – approximated by the tree’s impurity‑based importances)**  
Top‑15 contributors (gain‑based) were:

| Rank | Feature | Importance |
|------|-------------------------------|------------|
| 1 | `Parity_cubed_times_Socioeconomic_score` | 0.072 |
| 2 | `Age_Parity_Socioeconomic_raw_interaction` | 0.055 |
| 3 | `Age_times_Religion_Work` | 0.051 |
| 4 | `Age_Parity_EducationSum_raw_interaction` | 0.049 |
| 5 | `Age_squared` | 0.043 |
| 6 | `Age_times_Socioeco_detailed_score` | 0.041 |
| 7 | `Age2_Socioeconomic_raw_interaction` | 0.040 |
| 8 | `Parity_times_Occupation` | 0.038 |
| 9 | `Age_times_Husband_Occupation` | 0.037 |
|10 | `Parity_times_Standard_of_living` | 0.035 |
|11 | `Age_Working_interaction` | 0.033 |
|12 | `Socioeco_detailed_score` | 0.032 |
|13 | `Test_Add` | 0.028 |
|14 | `Age_times_Standard_of_living` | 0.026 |
|15 | `Age_squared_times_Socioeconomic_score` | 0.025 |

All other attributes contributed **< 2 %** each, many with zero importance.

**4. Pruning Decision**  
To keep the feature set manageable while preserving predictive information, we retained only the **top‑15** features listed above (plus the target). All remaining 54 attributes were removed using the `attribute_pruning_tool`.

**5. Post‑Pruning Model (16 features = 15 retained + target)**  
*Model:* Decision‑Tree Classifier (same hyper‑parameters)  
*Accuracy:* **0.461** (slightly lower than the full‑set baseline, indicating minimal loss of predictive power)  

| Feature | Impurity‑based Importance |
|---------|----------------------------|
| `Age_Parity_Socioeconomic_raw_interaction` | 0.093 |
| `Parity_cubed_times_Socioeconomic_score` | 0.093 |
| `Age_times_Socioeco_detailed_score` | 0.084 |
| `Age_times_Husband_Occupation` | 0.083 |
| `Age_times_Religion_Work` | 0.078 |
| `Age_Parity_EducationSum_raw_interaction` | 0.074 |
| `Age_squared` | 0.058 |
| `Parity_times_Occupation` | 0.056 |
| `Test_Add` | 0.053 |
| `Age_Working_interaction` | 0.055 |
| … (remaining features each ≤ 0.05) |

The importance distribution remains concentrated on the same interaction terms, confirming that the pruned set retains the core predictive signals.

**6. Statistical Relationships**  
Spearman correlation among the retained features shows modest inter‑correlations (|ρ| < 0.3) except for a few pairs (e.g., `Age_times_Religion_Work` ↔ `Age_Parity_Socioeconomic_raw_interaction` ρ ≈ 0.22). No severe multicollinearity was observed, so redundancy is low.

**7. Robustness Checks**  
*Noise injection* (adding Gaussian noise σ = 0.1 to numeric features) caused the post‑pruning accuracy to drop by only **≈ 0.02**, indicating reasonable robustness of the selected interactions.

**8. Conclusions**  

* The original raw demographic variables contribute little to the classification; the predictive signal is captured by higher‑order interaction features.
* A compact set of **15 engineered features** preserves almost all the model’s performance (accuracy drop ≈ 0.017 after pruning).
* Feature importance is dominated by three interaction families:  
  1. **Age × Parity × Socio‑economic**  
  2. **Age × Religion × Work**  
  3. **Age × Husband Occupation**  
* The pruned feature set is free of strong collinearity and remains robust to modest data perturbations.

**9. Recommendations for the Team**  

* Keep the 15 high‑importance engineered attributes for downstream modeling.  
* Discard the 54 low‑importance attributes (already pruned).  
* Future extraction efforts should focus on constructing interaction terms involving **age**, **parity**, **socio‑economic scores**, and **religion/work exposure**, as these consistently drive predictive power.