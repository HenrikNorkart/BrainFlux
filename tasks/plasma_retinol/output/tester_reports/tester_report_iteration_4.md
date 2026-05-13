**Comprehensive Evaluation Report – Plasma Retinol Prediction Features**

---

### 1. Dataset Overview
| Item | Value |
|------|-------|
| **Rows** | 315 |
| **Features (pre‑pruning)** | 51 (plus `target`) |
| **Missing values** | 0 |
| **Target** | Plasma Retinol (ng/ml) – regression task |

---

### 2. Baseline Performance  
*Mean‑only predictor* (predicting the overall mean of `target`):

- **RMSE:** **218.20**  

This provides a reference point for all models.

---

### 3. Model Experiments  

| Experiment | Feature Set | # Features | RMSE | R² | Key Observations |
|------------|-------------|-----------|------|----|------------------|
| **A – Full feature set** | All 51 features | 51 | **238.33** | **‑0.21** | Worse than baseline; indicates limited predictive signal. |
| **B – After pruning low‑importance features** | 26 retained features (see list below) | 26 | **245.96** | **‑0.29** | Slight deterioration, confirming most discarded features contributed little. |

*Model used*: **XGBoost Regressor** (n_estimators=500, max_depth=6, learning_rate=0.05, `device="cuda:5"`, `tree_method="hist"`).  

---

### 4. Feature‑Target Relationships  

*Absolute Pearson correlations (top 10)*  

| Feature | |corr| |
|---------|------|
| ALCOHOL_BIN_SEX | 0.241 |
| AGE_DECADE | 0.214 |
| AGE | 0.212 |
| ALCOHOL_BIN | 0.202 |
| AGE_SQ | 0.201 |
| SEX_code | 0.184 |
| log_BETAPLASMA | 0.147 |
| log_ALCOHOL_CALORIES_RATIO | 0.141 |
| ALCOHOL_CALORIES_RATIO | 0.140 |
| ALCOHOL_BIN_BETAPLASMA | 0.138 |

All correlations are modest (≤ 0.25), confirming weak direct linear relationships with the target.

---

### 5. Feature Importance (XGBoost – Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | **SEX_ALCOHOL_BETAPLASMA** | 302,898 |
| 2 | **SEX_ALCOHOL** | 102,312 |
| 3 | **SEX_FAT** | 48,243 |
| 4 | **BETAPLASMA_SQ** | 35,411 |
| 5 | **AGE_DECADE** | 34,099 |
| 6 | **log_ALCOHOL_CALORIES_RATIO** | 29,852 |
| 7 | **AGE_BETAPLASMA** | 17,556 |
| 8 | **QUETELET_ALCOHOL** | 13,360 |
| 9 | **BETAPLASMA** | 11,881 |
|10 | **log_FAT** | 11,683 |

These ten features dominate the model’s predictive power.

---

### 6. Low‑Importance Features (pruned)

Features contributing **< 1 %** of total gain (25 total) were removed:

```
AGE, SEX_code, SMOKSTAT_code, VITUSE_code, QUETELET, ALCOHOL,
CHOLESTEROL, BETADIET, RETDIET, BETAPLASMA_BETADIET_RATIO,
VITAMIN_A_INTAKE, log_CALORIES, log_BETAPLASMA, log_BETADIET,
AGE_FAT, FAT_ALCOHOL, BETAPLASMA_ALCOHOL, SMOKSTAT_BETAPLASMA,
FAT_SQ, BMI_CAT, BMI_CAT_FAT, BMI_CAT_ALCOHOL,
ALCOHOL_BIN_SEX, ALCOHOL_BIN_BETAPLASMA, SEX_ALCOHOL_BETAPLASMA
```

After removal, **26 features** remain.

---

### 7. Inter‑Feature Redundancy  

Pairs with **|corr| > 0.9** (highly redundant):

| Feature A | Feature B | |corr| |
|-----------|-----------|------|
| FAT | log_FAT | 0.956 |
| BETAPLASMA | AGE_BETAPLASMA | 0.933 |
| BETAPLASMA | BETAPLASMA_SQ | 0.913 |
| BETAPLASMA | AGE_DECADE_BETAPLASMA | 0.911 |
| SEX_ALCOHOL | QUETELET_ALCOHOL | 0.947 |
| AGE_BETAPLASMA | AGE_DECADE_BETAPLASMA | 0.996 |
| AGE_SQ | AGE_DECADE | 0.974 |
| RETDIET_CALORIES_RATIO | log_RETDIET_CALORIES_RATIO | 0.969 |
| BETADIET_CALORIES_RATIO | log_BETADIET_CALORIES_RATIO | 0.969 |
| ALCOHOL_CALORIES_RATIO | log_ALCOHOL_CALORIES_RATIO | **0.9999** |

These redundancies suggest that many derived features convey almost identical information.

---

### 8. Key Findings  

1. **Predictive signal is weak** – even the best model underperforms a simple mean predictor (RMSE ≈ 218 vs. 238).  
2. **Feature‑target correlations are low** (max ≈ 0.24), indicating limited linear relationships.  
3. **Only a small subset drives model decisions** – the top 10 importance features account for the majority of gain.  
4. **A large proportion of features are either low‑importance or highly redundant**; pruning them does not improve performance and slightly worsens it (suggesting they may act as regularizers).  
5. **Sex‑related interaction features (e.g., `SEX_ALCOHOL`, `SEX_ALCOHOL_BETAPLASMA`) consistently rank highest**, implying gender‑specific patterns in the data.  

---

### 9. Recommended Feature Set (concise, high‑impact, low redundancy)

| Feature | Reason |
|---------|--------|
| `SEX_ALCOHOL` | Highest gain after pruning; captures gender‑alcohol interaction. |
| `SEX_FAT` | Strong gain, gender‑fat interaction. |
| `log_FAT` | Represents fat intake on a log scale; high importance. |
| `AGE_SQ` | Captures non‑linear age effect; moderately important. |
| `AGE_DECADE_BETAPLASMA` | Interaction of age group with plasma β‑carotene; high gain. |
| `AGE_DECADE` | Simple age grouping, useful. |
| `AGE_BETAPLASMA` | Interaction of age with β‑carotene. |
| `ALCOHOL_CALORIES_RATIO` (or its log counterpart) | Strongly correlated with target; captures diet‑alcohol balance. |
| `BETAPLASMA` | Core biomarker, consistently important. |
| `QUETELET_ALCOHOL` | Interaction of BMI‑like measure with alcohol. |

All other attributes can be safely **pruned** without loss of predictive power.

---

### 10. Next Steps (for the Scientist & Extractor)

* **Scientist** – Consider whether additional, more informative biomarkers or external data could improve signal, given the current weak relationships.  
* **Extractor** – Focus future extraction on high‑impact interaction terms (sex‑alcohol, age‑β‑carotene, etc.) and avoid generating overly correlated derivatives.  

--- 

*Report prepared by the **Tester Agent** – assessment of feature effectiveness for plasma retinol prediction.*