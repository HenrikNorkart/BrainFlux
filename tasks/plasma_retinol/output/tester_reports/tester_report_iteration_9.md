**Feature‑Testing Report – Plasma Retinol Prediction**

---

### 1.  Objective
Evaluate the predictive value of the 89 supplied attributes for the regression target **‘target’** (plasma retinol, ng/ml).  

### 2.  Experimental Design
1. **Baseline model** – XGBoost (500 trees, depth 5, learning‑rate 0.05) trained on the full feature set.  
2. **Diagnostic analyses** –  
   * Correlation of each attribute with the target.  
   * XGBoost‑derived *gain* importance for all features.  
3. **Feature selection** – Retained the **30 most important features** (gain‑ranking) and discarded the rest.  
4. **Reduced‑model evaluation** – Same XGBoost configuration on the 30‑feature subset (train/validation split 80/20).  
5. **Documentation** – Key observations recorded with the `take_note_tool`.  

All models were executed on GPU **cuda:5** with `tree_method="hist"` as required.

---

### 3.  Results  

| Model | # Features | RMSE | R² |
|-------|-----------|------|----|
| **Full set (89)** | 89 | **227.18** | **‑0.099** |
| **Top‑30 subset** | 30 | **209.12** | **0.069** |

*The reduced model improves predictive performance (≈ 8 % lower RMSE) and yields a modest positive R², indicating that many of the original attributes add noise rather than signal.*

#### 3.1 Feature‑Importance (gain) – Top 10 (30‑feature model)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `SEX_ALCOHOL` | 34 146 |
| 2 | `SEX_ALCOHOL_BETAPLASMA` | 30 514 |
| 3 | `SEX_FAT` | 29 645 |
| 4 | `AGE_SQ` | 14 927 |
| 5 | `AGEDECADE_QUETELET` | 13 666 |
| 6 | `QUETELET_BETAPLASMA` | 13 561 |
| 7 | `log_FIBER_CALORIES_RATIO_BETAPLASMA` | 13 315 |
| 8 | `AGE_BETAPLASMA` | 11 481 |
| 9 | `LOG_FAT_QUETELET` | 11 467 |
|10 | `ALCOHOL_CALORIES_RATIO` | 11 198 |

*These interaction‑heavy attributes dominate the explanatory power, confirming that the engineered combinations capture most of the usable signal.*

#### 3.2 Correlation with Target (absolute Pearson)

| Feature | |Correlation| |
|---------|------------|
| `ALCOHOL_BIN_SEX` | 0.241 |
| `AGE_DECADE` | 0.214 |
| `AGE` | 0.212 |
| `VITUSE_ALCOHOL` | 0.203 |
| `ALCOHOL_BIN` | 0.202 |
| … | (remaining features ≤ 0.20) |

*No single raw variable shows strong linear association (max |r| ≈ 0.24), underscoring the importance of interaction terms.*

---

### 4.  Robustness Checks
* Adding modest Gaussian noise (σ = 0.05 × std) to the selected features altered RMSE by < 2 %, indicating the reduced model is reasonably stable to small perturbations.

---

### 5.  Pruned Feature List
The following 30 attributes were retained (all others were removed from the working set):

```
SEX_ALCOHOL_BETAPLASMA, BMI_CAT, SEX_ALCOHOL, FAT_SQ, SEX_FAT,
VITUSE_ALCOHOL, AGE_BETAPLASMA_ALCOHOL, AGE_BETAPLASMA,
BETAPLASMA_ALCOHOL, ALCOHOL_CALORIES_RATIO, LOG_FAT_QUETELET,
SEX_AGEDECADE_BETAPLASMA, log_FIBER_CALORIES_RATIO_BETAPLASMA,
AGEDECADE_QUETELET, LOG_CHOLESTEROL_QUETELET, AGE_SQ,
VITUSE_ALCOHOL_BETAPLASMA, AGE_DECADE_BETAPLASMA,
PC1_BETAPLASMA, SMOKSTAT_BETAPLASMA, PC1, FAT,
BMI_CAT_FAT, QUETELET_BETAPLASMA, FAT_ALCOHOL,
log_FIBER_CALORIES_RATIO, log_CHOLESTEROL_CALORIES_RATIO_QUETELET,
log_CHOLESTEROL_CALORIES_RATIO_BETAPLASMA,
RETDIET_BETADIET_RATIO, log_BETADIET
```

All other 59 attributes were deemed non‑contributory for this task and excluded.

---

### 6.  Conclusions
* The original 89‑feature set contains substantial redundancy and noise; many engineered interaction terms dominate predictive ability.  
* A compact **30‑feature** subset delivers a **significant performance gain** (RMSE ↓ ≈ 8 %, R² from negative to modestly positive).  
* The most influential predictors are interaction variables linking sex, alcohol consumption, and beta‑carotene measures, as well as quadratic age terms.  
* Linear relationships are weak, so reliance on simple correlation is insufficient; tree‑based importance provides a clearer signal.

**Next steps for the team**  
* Communicate the retained 30 attributes to the Scientist and Extractor agents for focused hypothesis generation.  
* Consider further validation (e.g., cross‑validation, external hold‑out) to confirm stability before model deployment.  

---  

*Report compiled by the Tester Agent.*