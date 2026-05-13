**Feature‑Evaluation Report – Contraceptive Method Prediction**

---

### 1.  Experimental Setup
* **Model:** RandomForestClassifier (300 trees, `n_jobs=4`, `random_state=42`).  
* **Data split:** 80 % train – 20 % test, stratified by the target.  
* **Metric:** Overall classification accuracy (macro‑averaged F1 also inspected).  
* **Tools used:** Python execution, `take_note_tool` for logging, `attribute_pruning_tool` for removal of low‑impact attributes, `attribute_lookup_tool` for explanations (not required here).

---

### 2.  Baseline (All 81 features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.518** |
| Macro‑avg F1 | 0.506 |
| Weighted‑avg F1 | 0.520 |

*Feature‑importance (RandomForest) identified 18 attributes with importance < 0.005 – mainly raw categorical variables (e.g., `Wifes_age_group`, `Parity_category`, `Both_High_Education`, …). These contributed little to predictive power.*

---

### 3.  Pruning Low‑Impact Features
**Removed attributes (18):**  

```
Wifes_age_group, Parity_category, Education_media_interaction,
Religion_Work_interaction, Both_High_Education, Both_midhigh_or_higher,
Wife_high_Husband_low, Husband_high_Wife_low, Parity_fine_category,
Wife_age_decade_group, Test_Wife_Edu, Test_Add, Test_Add2,
Socioeco_partial_sum, Socioeco_three_sum, Test_Feature_Add,
Test_Three_Sum_Paren, Test_Mul_Two
```

*Rationale:* each had a RandomForest importance < 0.005, indicating negligible contribution and potential redundancy.

---

### 4.  Post‑Pruning Performance (63 features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.532** |
| Macro‑avg F1 | 0.519 |
| Weighted‑avg F1 | 0.533 |
| Features used | **63** (down from 81)

*Result:* modest but consistent improvement (≈ 1.4 % absolute gain) despite a 22 % reduction in feature count, confirming that the pruned attributes were non‑informative.

---

### 5.  Key Predictive Features  

**RandomForest importance (top 20)**  

| Feature | Importance |
|------------------------------|------------|
| Age_Parity_Husband_Occupation_triple | 0.0305 |
| Age_Parity_Standard_of_living_triple | 0.0281 |
| Age_Parity_EducationSum_raw_interaction | 0.0251 |
| Age_Parity_Socioeconomic_triple | 0.0251 |
| Age_times_Wife_Education | 0.0248 |
| Age_times_Socioeco_detailed_score | 0.0247 |
| Age_times_Education_sum | 0.0246 |
| Age_squared_times_Parity_times_Socioeconomic_score | 0.0242 |
| Parity_squared_times_Age_times_Socioeconomic_score | 0.0233 |
| Age_times_Standard_of_living | 0.0233 |
| … (remaining top‑20 also interaction‑heavy) | |

**Permutation importance (top 10)**  

| Feature | Mean Δ Accuracy |
|------------------------------|----------------|
| Age_times_Wife_Education | 0.0149 |
| Age_times_Religion_Work | 0.0122 |
| Age2_Socioeconomic_raw_interaction | 0.0122 |
| Socioeconomic_score_times_Wifes_now_working | 0.0095 |
| Age_squared_times_Socioeconomic_score | 0.0095 |
| Age_times_Socioeco_detailed_score | 0.0095 |
| Age_times_Standard_of_living | 0.0088 |
| Age_Religion_interaction | 0.0088 |
| Age_times_Parity_category | 0.0075 |
| Age_Parity_Religion_Work_triple | 0.0075 |

*Interpretation:*  
- **Age** and **Parity** consistently appear in the most influential interactions.  
- **Socio‑economic score**, **Standard of living**, and **Education‑related sums** amplify the predictive signal when combined with age or parity.  
- Simple raw categorical variables (e.g., raw age group, basic education levels) are largely redundant once these higher‑order interaction features are present.

---

### 6.  Robustness Insight
Permutation‑importance results show that perturbing the top interaction terms leads to the greatest drop in accuracy, confirming that the model’s performance relies heavily on those non‑linear relationships. Features with very low permutation impact (e.g., `Test_Mul_Other` – importance ≈ 0.022 in RF but negligible in permutation) could be candidates for further pruning if model size is a concern.

---

### 7.  Conclusions & Recommendations for the Team
1. **Keep** the 63‑feature set after pruning; it yields the best trade‑off between performance and dimensionality.  
2. **Prioritise** the interaction features listed above for any downstream analysis or model‑explanation work – they carry the bulk of predictive power.  
3. **Discard** the 18 low‑importance raw attributes (already pruned) – they add noise without benefit.  
4. **Future work** (outside the scope of this test) could explore dimensionality‑reduction on the interaction space (e.g., PCA) or regularised linear models to verify that the same interactions dominate.

*All findings have been recorded in the internal notes for the Scientist Agent’s reference.*