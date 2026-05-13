**Feature‑Importance Evaluation Report**  

**1. Data & Goal**  
- Classification task: predict *Contraceptive_method_used* (3 classes).  
- Original engineered feature set: 42 attributes (excluding the target).  

**2. Initial Assessment (all 42 features)**  
| Metric | Value |
|--------|-------|
| 5‑fold CV Accuracy | **0.529** |
| 5‑fold CV Log‑Loss | **0.969** |

**Feature‑importance (XGBoost gain)** – top contributors  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | Both_High_Education | 25.29 |
| 2 | Parity_category | 6.45 |
| 3 | Age_squared | 3.77 |
| 4 | Test_Wife_Edu | 3.57 |
| 5 | Age_times_Parity_category | 2.71 |
| … | … | … |

**Permutation‑importance (accuracy drop on a hold‑out set)** – top contributors  

| Rank | Feature | Δ Accuracy |
|------|---------|------------|
| 1 | Parity_category | **0.0678** |
| 2 | Age_times_Wife_Education | **0.0508** |
| 3 | Age_times_Parity_category | **0.0305** |
| 4 | Test_Mul_Other | **0.0237** |
| 5 | Age_squared | **0.0203** |
| … | … | … |

*Observation*: Some features (e.g., Both_High_Education) received very high gain scores but caused only a tiny accuracy drop when permuted, indicating that their importance is largely due to interaction effects captured by the tree splits rather than standalone predictive power.

**3. Feature Pruning Strategy**  
- Combined the two importance views and kept the **15 features** that consistently showed the largest permutation impact or gain.  
- All remaining 27 attributes were deemed low‑impact and were pruned using the `attribute_pruning_tool`.

**Remaining Feature Set (15 attributes)**  

1. Parity_category  
2. Both_High_Education  
3. Education_disparity  
4. Combined_Edu_Occupation_sum  
5. Test_Wife_Edu  
6. Socioeco_religion_interaction  
7. Test_Mul_Other  
8. Socioeco_product_religion_interaction  
9. Age_squared  
10. Age_times_Education_sum  
11. Age_times_Socioeconomic_score  
12. Age_times_Parity_category  
13. Age_times_Parity_fine_category  
14. Age_times_Wife_Education  
15. Age_times_Husband_Occupation  

**4. Re‑evaluation with Reduced Set**  

| Metric | Value |
|--------|-------|
| 5‑fold CV Accuracy | **0.540** (↑ ~1.1 pp) |
| 5‑fold CV Log‑Loss | **0.960** (↓ ~0.009) |

*Interpretation*: Removing the low‑impact attributes **improved** predictive performance slightly, confirming that the pruned features added noise rather than useful signal.

**5. Key Take‑aways**  

| Insight | Evidence |
|---------|----------|
| **Parity_category** is the single most predictive attribute (largest accuracy drop when shuffled). | Permutation Δ Accuracy = 0.0678 |
| **Age‑based interaction terms** (e.g., Age_times_Wife_Education, Age_times_Parity_category) provide meaningful incremental information. | All rank within top 5‑10 of permutation importance. |
| **Both_High_Education** shows a huge gain in the tree model but negligible standalone effect; it mainly contributes through interactions. | Gain = 25.3, Δ Accuracy ≈ 0.0068 |
| **Many engineered interaction features** (e.g., Socioeco_product, Test_Add, etc.) do not improve the model and can be safely removed. | Pruned 27 attributes; model performance improved. |
| **A compact set (15 features)** yields a **more robust** model (higher CV accuracy, lower log‑loss) while simplifying interpretation and reducing computational cost. | Direct comparison of full vs. reduced models. |

**6. Final Recommendation (Tester’s Scope)**  
- **Retain** the 15 features listed above for downstream modelling.  
- **Discard** the 27 pruned attributes (see pruning list).  
- No further feature engineering is required; the current set already balances predictive power and parsimony.  

*All steps, code, and notes are recorded via the provided tooling.*