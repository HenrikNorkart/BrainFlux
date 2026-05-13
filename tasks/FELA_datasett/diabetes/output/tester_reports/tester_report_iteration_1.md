**Comprehensive Feature‑Evaluation Report**  
*Diabetes type prediction (3‑class classification)*  

---

### 1. Experimental Setup  

| Step | Description |
|------|-------------|
| **Data** | `df_attributes` – 139 524 rows, 16 predictor columns + `target`. |
| **Target** | 3 classes (0 = no diabetes, 1 = type I, 2 = type II). |
| **Model** | XGBoost (`objective='multi:softprob'`, `num_class=3`, `device='cuda:5'`, `tree_method='hist'`). |
| **Train‑test split** | 80 % train / 20 % test, stratified, `random_state=42`. |
| **Metrics** | Accuracy, macro‑averaged F1, per‑class precision/recall. |
| **Additional analyses** | – Feature‑importance (gain). <br>– Pearson correlation matrix. <br>– Robustness test (Gaussian noise on BMI & Age). |

---

### 2. Baseline Results (all 16 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8444** |
| **Macro‑F1** | **0.3615** |
| **Per‑class performance** | <ul><li>Class 0: Precision 0.854, Recall 0.985, F1 0.915</li><li>Class 1: Precision 0, Recall 0, F1 0 (never predicted)</li><li>Class 2: Precision 0.501, Recall 0.102, F1 0.169</li></ul> |
| **Confusion (test)** | `[[23169, 0, 357], [470, 0, 37], [3477, 0, 395]]` |

**Interpretation** – The model reliably distinguishes the majority “no‑diabetes” class (0) but fails to identify the minority type‑I class (1). Class‑II (2) is detected with modest precision but very low recall, reflecting the imbalance.

---

### 3. Feature‑Importance (gain)

Top 10 contributors (gain):

| Rank | Feature | Gain |
|------|---------|------|
| 1 | **HighBP** | 40.57 |
| 2 | **HighChol** | 12.37 |
| 3 | **BMI** | 4.13 |
| 4 | **HighBP_Sex** | 3.42 |
| 5 | **Age** | 3.31 |
| 6 | **HeavyAlcohol** | 3.01 |
| 7 | **Age_Sex** | 2.25 |
| 8 | **PhysActivity** | 2.03 |
| 9 | **HighChol_Sex** | 1.54 |
|10| **BMI_Sex** | 1.28 |

All remaining features have gain ≤ 1.5, many close to 1.0.

---

### 4. Correlation Analysis  

Only one pair showed a strong linear relationship (|r| > 0.8):

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| **BMI_Sex** | **Age_Sex** | **0.85** |

All other pairs were ≤ 0.8, indicating low redundancy.

---

### 5. Feature Pruning  

**Rationale**  

* Remove highly correlated interaction (`BMI_Sex` vs `Age_Sex`).  
* Drop interaction terms whose gain is near the low‑importance floor and add little predictive value.

**Pruned attributes**

```text
BMI_Sex
HeavyAlcohol_Sex
Smoker_Sex
FruitVegScore_Sex
PhysActivity_Sex
```

Pruning was executed via `attribute_pruning_tool`.

---

### 6. Post‑Pruning Results  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8446** (virtually unchanged) |
| **Macro‑F1** | **0.3635** (slight improvement) |
| **Per‑class performance** | Class 0 unchanged; Class 1 still never predicted; Class 2 F1 ↑ 0.176 vs 0.170 |

The modest gain in macro‑F1 suggests that the removed interactions contributed noise rather than signal.

---

### 7. Robustness Test  

*Added Gaussian noise (σ = 0.5) to the two continuous predictors (BMI, Age).*

| Metric | Noisy Data |
|--------|------------|
| **Accuracy** | **0.8457** |
| **Macro‑F1** | **0.3641** |

Performance remained stable, indicating that the retained feature set is robust to modest perturbations in the most influential numeric variables.

---

### 8. Key Take‑aways  

1. **Predictive power is driven primarily by a handful of clinical variables** – high blood pressure, high cholesterol, BMI, age, and heavy alcohol consumption (and their sex‑interaction terms).  
2. **Interaction features add limited value**; several low‑gain interactions can be safely removed without hurting performance.  
3. **Class imbalance is the dominant limitation** – the model never predicts the minority type‑I class. Addressing this (e.g., class‑weighting, oversampling) is required for any improvement in macro‑F1, but falls outside the current “feature‑only” evaluation scope.  
4. **Feature set after pruning (12 attributes)** is concise, low‑redundancy, and robust:

   ```
   HighBP, HighChol, BMI, HighBP_Sex, Age, HeavyAlcohol,
   Age_Sex, PhysActivity, HighChol_Sex, Smoker,
   FruitVegScore, HeavyAlcohol (main), Smoker (main)
   ```

   *(All remaining interaction terms have measurable gain.)*

5. **Correlation issue resolved** – only one high‑correlation pair existed; we removed the weaker of the two (`BMI_Sex`).

---

### 9. Recommended Final Feature Subset  

| Feature | Type |
|---------|------|
| HighBP | Binary (0/1) |
| HighChol | Binary |
| BMI | Continuous |
| HighBP_Sex | Interaction |
| Age | Ordinal (1‑13) |
| HeavyAlcohol | Binary |
| Age_Sex | Interaction |
| PhysActivity | Binary |
| HighChol_Sex | Interaction |
| Smoker | Binary |
| FruitVegScore | Integer (0‑2) |
| Smoker (main) – retained (gain ≈ 1.18) |

*All other attributes (including the pruned interactions) can be omitted for a more manageable model without sacrificing predictive accuracy.*

---

### 10. Next Steps (outside current scope)

* Apply class‑imbalance mitigation (e.g., `scale_pos_weight`, SMOTE) to improve detection of type‑I diabetes.  
* Explore calibrated probability thresholds for minority classes.  
* Validate on an external hold‑out set to confirm generalisation.

---  

*Prepared by the Tester Agent – Feature Evaluation Loop*  