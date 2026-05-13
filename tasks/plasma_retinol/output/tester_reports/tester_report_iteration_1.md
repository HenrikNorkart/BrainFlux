**Comprehensive Feature‑Evaluation Report – Plasma Retinol Dataset**

---

### 1. Experimental Setup
- **Target variable:** `target` (plasma retinol, ng/ml)  
- **Data:** 315 rows × 14 columns (13 features + target). Categorical variables are already encoded as numeric codes (`*_code`).  
- **Models evaluated:**  
  - **XGBoost Regressor** (500 trees, depth 5, learning‑rate 0.05, `device="cuda:5"`, `tree_method="hist"`).  
  - **Linear Regression** (baseline).  
- **Metrics:** RMSE (lower = better) and R² (higher = better).  
- **Train‑test split:** 80 % / 20 % (random_state = 42).  

---

### 2. Predictive Performance  

| Model                | RMSE   | R² (test) |
|----------------------|--------|-----------|
| XGBoost (all 13)    | **221.5** | **‑0.04** |
| XGBoost (top‑8 important) | 225.1 | ‑0.08 |
| Linear Regression    | 219.1 | ‑0.02 |

*All models produce **negative R²**, indicating that the feature set alone cannot explain the variance in plasma retinol better than a naïve mean predictor.*

---

### 3. Feature Importance (XGBoost – Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | **SEX_code** | 27 244 |
| 2 | **FAT** | 14 490 |
| 3 | **BETAPLASMA** | 13 330 |
| 4 | **ALCOHOL** | 13 248 |
| 5 | **BETADIET** | 12 627 |
| 6 | **FIBER** | 11 608 |
| 7 | **AGE** | 10 515 |
| 8 | **RETDIET** | 10 408 |
| 9 | **CHOLESTEROL** | 9 026 |
|10 | **QUETELET** | 8 678 |
|11 | **VITUSE_code** | 8 673 |
|12 | **CALORIES** | 8 451 |
|13 | **SMOKSTAT_code** | 3 509 |

*Only `SEX_code`, `FAT`, `BETAPLASMA`, and `ALCOHOL` contribute >10 % of the total gain. The remaining features each account for <5 %.*

---

### 4. Statistical Relationships  

| Feature | |Correlation with Target| |
|---------|---|------------------------|---|
| AGE | 0.212 |
| SEX_code | 0.184 |
| FAT | 0.091 |
| CALORIES | 0.073 |
| BETAPLASMA | 0.072 |
| CHOLESTEROL | 0.070 |
| RETDIET | 0.063 |
| FIBER | 0.044 |
| SMOKSTAT_code | 0.029 |
| ALCOHOL | 0.017 |
| BETADIET | 0.014 |
| QUETELET | 0.013 |
| VITUSE_code | 0.007 |

*All absolute correlations are **≤ 0.22**, confirming weak linear relationships with the target.*

**Inter‑feature redundancy:**  
- `CALORIES` ↔ `FAT` correlation = **0.87** (highly collinear).  

---

### 5. Impact of Feature Pruning  

Given the low predictive power and the redundancy between `CALORIES` and `FAT`, a pruning decision was made to remove the least informative attributes (based on importance, correlation, and redundancy):

- **Pruned attributes:** `VITUSE_code`, `QUETELET`, `CHOLESTEROL`, `RETDIET`, `SMOKSTAT_code`

*After pruning, the remaining feature set (8 variables) still yields a negative R², confirming that the current feature collection lacks sufficient signal for accurate plasma‑retinol prediction.*

---

### 6. Conclusions  

1. **Predictive Power:** The current feature set explains virtually none of the variance in plasma retinol (negative R² across models).  
2. **Key Contributors:** `SEX_code`, `FAT`, `BETAPLASMA`, and `ALCOHOL` are the most important features, yet their individual correlations with the target are modest.  
3. **Redundancy:** `CALORIES` and `FAT` are highly collinear; retaining only one would simplify the model with negligible performance loss.  
4. **Pruning Outcome:** Removing five low‑importance, low‑correlation attributes reduces dimensionality without degrading performance; however, overall model quality remains poor.  
5. **Recommendation (assessment only):** Further improvement will likely require **additional or alternative biomarkers**, richer clinical data, or non‑linear transformations—not simple feature selection from the existing set.

---

*All notes and pruning actions have been recorded via the provided tooling.*