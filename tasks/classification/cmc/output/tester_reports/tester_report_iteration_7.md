**Tester‑Agent Report – Feature‑Importance Evaluation for *Contraceptive_method_used***  

---

### 1.  Experimental Setup  

| Step | Description |
|------|-------------|
| **Data** | `df_attributes` –  ≈ 70 engineered columns + `target` (3‑class label). |
| **Model** | `RandomForestClassifier` (n_estimators = 300, `random_state` = 42, `n_jobs` = ‑1). |
| **Train‑Test split** | 80 % / 20 % stratified on the target. |
| **Metric** | Overall classification accuracy (multiclass). |
| **Feature‑importance** | – Built‑in Gini‑based importance from the Random Forest (mean decrease impurity). <br>– Ranked features from highest to lowest importance. |
| **Redundancy check** | Pearson correlation matrix (|ρ| > 0.9) to spot near‑duplicate columns. |
| **Pruning criterion** | Features with importance < 0.01 were deemed non‑contributory and removed from further consideration. |

---

### 2.  Model Performance  

| Metric | Value |
|--------|-------|
| **Accuracy** (20 % hold‑out) | **0.536** |
| *Interpretation* | Baseline predictive power is modest (≈ 53 % correct). The model is far from perfect, reflecting the difficulty of the task and the heavy redundancy in the engineered feature set. |

---

### 3.  Feature‑Importance Findings  

| Rank | Feature (short name) | Importance |
|------|----------------------|------------|
| 1 | `Age_Parity_EducationSum_raw_interaction` | 0.0408 |
| 2 | `Age_Parity_Socioeconomic_raw_interaction` | 0.0401 |
| 3 | `Parity_times_Socioeconomic_score` | 0.0341 |
| 4 | `Age_times_Socioeco_detailed_score` | 0.0335 |
| 5 | `Age_times_Socioeconomic_score` | 0.0323 |
| 6 | `Age_times_Standard_of_living` | 0.0320 |
| 7 | `Age3_Socioeconomic_raw_interaction` | 0.0314 |
| 8 | `Parity_times_Education_sum` | 0.0310 |
| 9 | `Age2_Socioeconomic_raw_interaction` | 0.0308 |
| 10 | `Age_times_Wife_Education` | 0.0307 |
| … | *(remaining top‑20 shown in the execution log)* | … |

**Key observations**

* The **most predictive variables are high‑order interaction terms** that combine **age, parity, education‑sum, and socioeconomic indices**.  
* Simple, raw descriptors (e.g., `Wifes_age_group`, `Parity_category`, `Education_sum`) receive **very low importance** (≤ 0.01) and contribute little beyond the engineered interactions.  
* Several interaction features that involve *media exposure* or *religion‑work* also score low and are redundant with other columns.

---

### 4.  Redundancy & Correlation  

* 72 feature pairs exhibited |ρ| > 0.9 (e.g., `Wifes_age_group` ↔ `Age_squared`, `Education_sum` ↔ `Education_product`, `Parity_category` ↔ `Parity_fine_category`).  
* In almost every highly‑correlated pair, the **interaction‑rich version** (e.g., `Age_times_AgeGroup`) held **higher importance** than the raw base variable.

---

### 5.  Pruning Decisions  

All features with importance **\< 0.01** were removed from the candidate set (26 attributes).  
The removed list (alphabetical) includes, for example:

```
Age_MediaExposure_interaction,
Both_High_Education,
Both_midhigh_or_higher,
Education_disparity,
Education_media_interaction,
Education_product,
Education_sum,
Husband_high_Wife_low,
Parity_category,
Parity_fine_category,
Parity_times_Media_exposure,
Parity_times_Education_disparity,
Religion_Edu_Work_score,
Religion_Work_interaction,
Socioeco_partial_sum,
Socioeco_three_sum,
Test_Add,
Test_Add2,
Test_Feature_Add,
Test_Mul_Two,
Test_Three_Sum_Paren,
Test_Wife_Edu,
Wifes_age_group,
Wife_age_decade_group,
Wife_high_Husband_low,
...
```

*The pruning was performed via the `attribute_pruning_tool`; the underlying data frame remains unchanged, but the pruned attributes are excluded from any subsequent analysis.*

---

### 6.  Summary & Take‑aways  

| Aspect | Finding |
|--------|---------|
| **Predictive power** | The current feature set yields **≈ 53 % accuracy** – modest, indicating substantial room for improvement (e.g., better modeling, more informative raw variables). |
| **Most useful features** | Complex interaction terms that blend *age*, *parity*, *education‑sum*, and *socio‑economic scores* dominate predictive importance. |
| **Least useful features** | Raw demographic descriptors and many engineered interactions (especially those tied to media exposure, religion‑work, or simple sums) have negligible contribution and were pruned. |
| **Redundancy** | Over 70 highly correlated pairs exist; many low‑importance features are near‑duplicates of higher‑importance interaction terms. |
| **Action taken** | Pruned 26 low‑importance attributes, keeping the feature space focused on the ~30 most informative interaction variables. |

**Next steps for the team** (outside the scope of this report) could include:

* Re‑training a model on the **pruned feature set** to verify whether accuracy improves (less noise, lower variance).  
* Exploring alternative algorithms (e.g., Gradient Boosting, Multiclass Logistic Regression) to see if they can better exploit the interaction features.  
* Considering dimensionality‑reduction techniques (e.g., PCA) on the remaining interaction space to further condense information.

--- 

*Report compiled by the Tester Agent – all observations are based on the experiments executed with the provided tools.*