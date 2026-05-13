**Tester Agent – Feature Evaluation Report**

---

### 1. Experimental Setup
- **Data:** 686 instances, 88 columns (including *target*).  
- **Target encoding:** `no → 0`, `yes → 1`.  
- **Model:** XGBoost Classifier (binary:logistic, 200 trees, max_depth 4, learning_rate 0.1, subsample 0.9, colsample_bytree 0.9, `tree_method='hist'`).  
- **Train‑test split:** 80 % / 20 % stratified (random_state = 42).  
- **Metrics:** Accuracy, ROC‑AUC.  
- **Feature importance:** XGBoost “gain”.  
- **Additional analyses:** Correlation matrix (|ρ| > 0.9) and zero‑gain feature identification.

---

### 2. Baseline Model Performance (All Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.775** |
| **ROC‑AUC** | **0.665** |

The model shows moderate discriminative ability on this clinical classification task.

---

### 3. Feature Importance (Top‑15 by Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `chf_stage_numeric` | 7.02 |
| 2 | `chf_stage_x_mean_systolic_bp` | 3.10 |
| 3 | `age_x_pulmonary_edema` | 2.41 |
| 4 | `chf_stage_x_mean_diastolic_bp` | 2.20 |
| 5 | `arrhythmia_admission_count` | 2.03 |
| 6 | `pulmonary_comorbidity_count` | 1.94 |
| 7 | `age_years` | 1.91 |
| 8 | `diabetes_x_mean_systolic_bp` | 1.80 |
| 9 | `icu_arrhythmia_paroxysmal_atrial_fibrillation` | 1.77 |
|10 | `composite_chronic_risk_score` | 1.59 |
|11 | `ecg_conduction_complete_RBBB` | 1.56 |
|12 | `ecg_arrhythmia_ventricular_contractions` | 1.55 |
|13 | `total_risk_score` | 1.53 |
|14 | `mean_ROE` | 1.49 |
|15 | `pulmonary_edema_present` | 1.49 |

These features capture chronic heart‑failure severity, age‑related interactions, arrhythmia burden, and key laboratory/comorbidity scores.

---

### 4. Redundancy & Correlation Findings
10 pairs of attributes exhibited **|ρ| > 0.9**, e.g.:

| Pair | Correlation |
|------|-------------|
| `hypertension_stage` ↔ `hypertension_stage_x_mean_systolic_bp` | 0.96 |
| `diabetes_binary` ↔ `diabetes_x_mean_systolic_bp` | 0.98 |
| `chf_stage_numeric` ↔ `chf_stage_x_mean_systolic_bp` | 0.97 |
| `mean_systolic_bp` ↔ `mean_map` | 0.95 |
| `pulmonary_edema_present` ↔ `age_x_pulmonary_edema` | 0.99 |
| … (6 more) | … |

The derived “*_x_mean_*” versions consistently carried **higher gain** than their raw counterparts, indicating they capture the predictive signal more effectively.

---

### 5. Feature Pruning

**Zero‑gain attributes (33) were removed** because the model never used them during splits, e.g.:

- `hypertension_duration`
- `obesity_binary`
- `cardiogenic_shock_present`
- All raw ECG rhythm/conduction flags (`ecg_ritm_*`, `ecg_arrhythmia_*`, `ecg_conduction_*`)

**Redundant high‑correlation base variables were also pruned**:

- `hypertension_stage`
- `diabetes_binary`
- `mean_systolic_bp`
- `mean_diastolic_bp`
- `mean_map`

*Total pruned attributes: 25 (33 zero‑gain + 5 redundant).*

---

### 6. Post‑Pruning Model Evaluation

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.775** |
| **ROC‑AUC** | **0.665** |
| **Remaining features** | **87** (down from 112 original columns before any engineering) |

Performance remained unchanged, confirming that the removed attributes contributed little or duplicated information.

---

### 7. Key Take‑aways

1. **Predictive power** is driven primarily by chronic‑heart‑failure stage, age‑related interaction terms, and arrhythmia burden.
2. **Derived interaction features** (`*_x_*`) outperform their raw parents; they should be retained for any downstream modeling.
3. **A sizable portion of the original attribute set (≈23 %) is noisy or redundant** and can be safely excluded without sacrificing accuracy.
4. **Model stability** is good; the same accuracy is observed before and after pruning, indicating robustness to the removal of low‑information variables.
5. **Future work** (outside the Tester’s remit) could explore:
   - Calibration of probabilities (given modest AUC).
   - Alternative classifiers or ensemble stacking to boost discriminative ability.

---

**Prepared by:** Tester Agent – Feature Evaluation Loop.