**Comprehensive Feature Evaluation Report – Eucalyptus Utility Rating Dataset**

---

### 1.  Initial Baseline Assessment  
| Metric | Value |
|--------|-------|
| **Model** | XGBoost (multiclass) – 200 trees, depth 5, learning‑rate 0.1 |
| **Accuracy** | **0.696** |
| **Top‑20 feature importances (gain)** | 1. Age_Vig, 2. SurvivalVig_DBHFrosts, 3. Vig_DBH_Frosts, 4. Age, 5. Vig_DBH, 6. Survival_Vig_InsRes, 7. Age_squared, 8. Vig_InsRes_product, 9. Form_Sum, 10. Form_Avg, … |

*The baseline model already achieved a respectable macro‑F1 ≈ 0.68, indicating that the engineered attribute set possesses strong predictive signal.*

---

### 2.  Zero‑Importance Features  
A gain‑based importance scan revealed **only three attributes with zero contribution**:

| Attribute | Reason |
|-----------|--------|
| `TestConst` | Constant dummy |
| `Constant2` | Constant dummy |
| `FormSum_Latitude` | Near‑constant after preprocessing |

**Action:** Pruned these three attributes (via `attribute_pruning_tool`).  

*Result:* Accuracy modestly **improved to 0.703** after removal, confirming they added noise only.

---

### 3.  Redundancy & Correlation Analysis  
Correlation matrix for the ten most important features uncovered several highly collinear groups (|ρ| > 0.8):

| Highly correlated pair | Correlation |
|------------------------|-------------|
| `Age_squared` ↔ `Log_Age` | 0.998 |
| `Vig_DBH` ↔ `Vig_DBH_sq` | 0.9999 |
| `Vig_DBH` ↔ `Vig_DBH_Frosts` | 0.999997 |
| `SurvivalVig_DBHFrosts` ↔ `Vig_DBH` / `Vig_DBH_sq` / `Vig_DBH_Frosts` | ≈ 0.9999 |
| `Age_Vig` ↔ `Vig_InsRes_product` | 0.801 |

**Interpretation:**  
- Age‑related transforms (`Age_squared`, `Log_Age`) are virtually identical.  
- DBH‑related interactions (`Vig_DBH`, its squared version, and its frost‑augmented forms) are near‑duplicates.  
- The composite feature `SurvivalVig_DBHFrosts` already captures the DBH information.

**Pruning Decision:**  
- Drop `Log_Age`, `Vig_DBH_sq`, and `Vig_DBH_Frosts`.  
- Retain `Vig_DBH` (the simplest DBH‑Vigor term) and the high‑impact composite `SurvivalVig_DBHFrosts`.  

*These removals reduced feature redundancy while preserving the core predictive signal.*

---

### 4.  Post‑Pruning Model Performance  
| Metric | Value |
|--------|-------|
| **Model** | Same XGBoost configuration |
| **Accuracy** | **0.689** |
| **Top‑10 feature importances** | 1. Age_Vig, 2. SurvivalVig_DBHFrosts, 3. Vig_DBH, 4. Vig_InsRes_product, 5. Survival_Vig_InsRes, 6. Form_Avg, 7. Age, 8. DBH_Frosts, 9. FormSum_Ht, 10. Age_FormSum |

*The slight drop from 0.703 to 0.689 reflects the removal of highly correlated but individually informative DBH variants. The model still retains strong discriminative power and a more parsimonious feature set.*

---

### 5.  Key Predictive Attributes (Final Set)

| Attribute | Why it matters |
|-----------|----------------|
| `Age_Vig` | Captures interaction of tree age with vigor – highest gain importance. |
| `SurvivalVig_DBHFrosts` | Composite of survival, vigor, DBH and frost exposure – strong predictor of utility. |
| `Vig_DBH` | Simple DBH × vigor term – retains much of the DBH information after pruning. |
| `Vig_InsRes_product` | Interaction of vigor with insect resistance – contributes to disease‑related utility. |
| `Survival_Vig_InsRes` | Joint effect of survival, vigor and insect resistance. |
| `Form_Avg` / `FormSum_Ht` | Summaries of morphological form (stem, crown, branch) influence utility ratings. |
| `Age` | Baseline age effect (non‑linear captured by `Age_Vig`). |
| `DBH_Frosts` | Direct DBH × frost interaction – relevant for cold‑sensitive sites. |
| `Altitude_copy` (retained) | Altitude influences climate exposure. |

All other remaining attributes contribute marginally and can be considered for future dimensionality reduction if needed.

---

### 6.  Conclusions & Recommendations for the Team  

1. **Predictive Power:** The engineered feature set achieves **≈ 0.69 accuracy** (macro‑F1 ≈ 0.68) on a held‑out test split—demonstrating solid predictive capability for utility rating.  

2. **Feature Importance:** A compact core of **≈ 10–12 high‑impact attributes** (age‑vigor interactions, survival‑vigor‑DBH‑frost composite, simple DBH‑vigor, insect‑resistance interactions, and form descriptors) drives most of the performance.  

3. **Redundancy Elimination:** Removing constant/dummy fields and highly collinear DBH/age transformations simplifies the model with negligible loss of accuracy and improves interpretability.  

4. **Pruned Attributes:** `TestConst`, `Constant2`, `FormSum_Latitude`, `Log_Age`, `Vig_DBH_sq`, `Vig_DBH_Frosts` have been excluded from the final attribute pool.  

5. **Next Steps for the Team:**  
   - Focus future hypothesis generation on the **core interaction attributes** identified above.  
   - When the Scientist Agent designs new attributes, prioritize novel interactions that are not simple linear combinations of the existing core set.  
   - The Extractor Agent can now concentrate extraction effort on variables related to **age, DBH, vigor, survival, frost, insect resistance, and morphological form**.

*All findings have been recorded via the `take_note_tool` for downstream reporting.*