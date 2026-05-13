**Feature Evaluation Report – Heart Disease Classification**

---

### 1. Overview  
The dataset contains 14 engineered attributes derived from the original clinical variables, plus the binary target **‘target’** (presence of heart disease).  

### 2. Initial Model (All Features)  
| Metric | Value |
|--------|-------|
| **Algorithm** | RandomForest (300 trees, `n_jobs=4`) |
| **Accuracy** | **0.755** |
| **ROC‑AUC** | **0.824** |
| **Number of Features** | 13 (excluding target) |

**Top‑10 Feature Importances** (gain from the forest)  

| Feature | Relative Importance |
|---------|----------------------|
| Age_X_ExerciseAngina | 0.183 |
| MaxHR_per_age | 0.125 |
| Cholesterol_per_age | 0.119 |
| Age_div_MaxHR | 0.116 |
| Age_X_Cholesterol | 0.106 |
| Age_X_MaxHR | 0.083 |
| Age_X_RestingBP | 0.081 |
| RestBP_per_age | 0.071 |
| Age_zscore | 0.049 |
| Age_squared | 0.047 |

**Statistical Redundancy** – Pairwise absolute correlations > 0.9  

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| Age_group | Age_zscore | 0.905 |
| Age_squared | Age_zscore | 0.994 |
| MaxHR_per_age | Age_div_MaxHR | 0.911 |

These highly correlated pairs suggest redundancy that could inflate model complexity without adding predictive value.

---

### 3. Feature Pruning  
**Removed Features (redundant):**  

- `Age_group`  
- `Age_squared`  
- `MaxHR_per_age`

Rationale: each is strongly correlated with another retained feature (`Age_zscore` or `Age_div_MaxHR`). Their removal should reduce multicollinearity while preserving information.

---

### 4. Model After Pruning (10 Features)  
| Metric | Value |
|--------|-------|
| **Algorithm** | RandomForest (same hyper‑parameters) |
| **Accuracy** | **0.755** (unchanged) |
| **ROC‑AUC** | **0.831** (slight improvement) |
| **Number of Features** | 10 |

**Top‑5 Feature Importances**  

| Feature | Relative Importance |
|---------|----------------------|
| Age_X_ExerciseAngina | 0.201 |
| Age_div_MaxHR | 0.165 |
| Cholesterol_per_age | 0.134 |
| Age_X_Cholesterol | 0.127 |
| Age_X_MaxHR | 0.105 |

**Correlation Check (post‑pruning):** No feature pairs exceed a correlation of 0.9, indicating the retained set is free of high multicollinearity.

---

### 5. Key Findings  

1. **Predictive Power** – The engineered feature set achieves solid predictive performance (≈ 75 % accuracy, > 0.83 AUC) on a hold‑out test split.  
2. **Feature Importance** – Interaction terms that combine **Age** with other clinical measurements dominate the model, especially `Age_X_ExerciseAngina`. Simple ratios (`Age_div_MaxHR`) and age‑normalized cholesterol (`Cholesterol_per_age`) also contribute markedly.  
3. **Redundancy Elimination** – Removing three highly correlated attributes did **not** degrade accuracy and modestly improved AUC, confirming that they added little unique information.  
4. **Statistical Relationships** – After pruning, the remaining features exhibit low inter‑correlation, reducing the risk of over‑fitting and simplifying interpretability.  

---

### 6. Recommendations (Tester‑only)  

- **Retain** the 10‑feature subset identified above for downstream modeling.  
- **Discard** the pruned attributes (`Age_group`, `Age_squared`, `MaxHR_per_age`) from any further analyses.  
- No further pruning is advised at this stage; all remaining features show distinct contributions and low collinearity.  

*End of Report.*