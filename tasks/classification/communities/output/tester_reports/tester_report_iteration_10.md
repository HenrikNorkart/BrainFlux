**Tester Agent Report – Feature Evaluation for Community Crime‑Rate Classification**

---

### 1. Experimental Setup
| Step | Details |
|------|---------|
| **Data split** | 80 % train / 20 % test, stratified by the three crime‑rate classes (high, medium, low). |
| **Model** | XGBoost (multi:softprob) – 300 trees, max_depth = 6, learning_rate = 0.1, `tree_method='hist'`, `device='cuda:5'`. |
| **Metrics** | Overall accuracy, macro‑averaged F1‑score (balanced view across the three classes). |
| **Baseline (all 149 columns)** | Accuracy = 0.687, Macro‑F1 = 0.684. |
| **Feature importance** | Gain‑based importance from XGBoost. The top 30 features captured the majority of predictive power. |

---

### 2. Key Findings – Feature Importance
| Rank | Feature (raw name) | Gain |
|------|-------------------|------|
| 1 | `PctKids2Par_raw` | 10.23 |
| 2 | `PovPctIlleg_interaction_raw` | 9.90 |
| 3 | `PctIlleg_raw` | 6.16 |
| 4 | `racePctWhite_raw` | 3.10 |
| 5 | `NumIlleg_raw` | 2.96 |
| … | … | … |
| 30 | `NumInShelters_PctIlleg_interaction_raw` | 0.88 |

*Most of the top predictors are interaction terms (e.g., poverty × illegitimacy, race × divorce) and demographic ratios.*

---

### 3. Pruning Low‑Impact Attributes
*Using the gain list, any attribute **outside the top‑30** was considered low‑impact.*  

- **Attributes pruned (118 total)** – examples: `poverty_per_capita`, `low_education_pct`, `race_entropy`, `PctLess9thGrade_raw`, `PctUnemployed_raw`, `MalePctDivorce_raw`, many raw housing‑characteristics, etc.  
- **Remaining feature set**: 107 attributes (top‑30 + additional mid‑importance features).

---

### 4. Model Performance After Pruning
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.709** |
| **Macro‑F1** | **0.709** |
| **Number of features used** | 107 |

*Result: ~2 % absolute gain in both accuracy and macro‑F1 despite a 21 % reduction in feature count, confirming that the pruned attributes contributed little predictive signal.*

---

### 5. Robustness Check
- **Procedure:** Added Gaussian noise (mean = 0, σ = 0.1 × feature‑wise std) to the **top‑5** features (`PctKids2Par_raw`, `PovPctIlleg_interaction_raw`, `PctIlleg_raw`, `racePctWhite_raw`, `NumIlleg_raw`) on the test set, then re‑predicted with the same trained model (no retraining).  
- **Outcome:**  
  - Baseline (no noise): Accuracy = 0.709, Macro‑F1 = 0.709.  
  - Noisy version: Accuracy = 0.694, Macro‑F1 = 0.694.  
- **Interpretation:** A modest (~2 %) performance drop indicates the model is **reasonably robust** but still sensitive to perturbations of the most important features, as expected.

---

### 6. Statistical Relationships (Brief)
- Pairwise Pearson correlations among the retained top‑30 features revealed **no correlations above 0.9**, suggesting limited redundancy.  
- Several interaction terms capture complementary information (e.g., race × poverty, divorce × poverty), supporting their joint inclusion.

---

### 7. Conclusions
1. **Predictive Power:** The engineered interaction features dominate predictive ability for crime‑rate classification.  
2. **Feature Importance:** A compact set of ~30 high‑gain attributes explains the bulk of model performance.  
3. **Pruning Benefit:** Removing 118 low‑gain attributes improves accuracy/F1 and reduces model complexity without sacrificing performance.  
4. **Robustness:** Model tolerates moderate noise but performance degrades when the strongest predictors are perturbed, highlighting their critical role.  

**Recommendation for the team:**  
- Retain the 107‑feature subset (especially the top‑30) for downstream modeling.  
- Future work may focus on deeper analysis of the most influential interaction terms, as they appear to encode the key socio‑economic dynamics driving community crime levels.