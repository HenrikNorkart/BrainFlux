**Comprehensive Feature Evaluation Report – Heart Disease Classification**

---

### 1. Initial Baseline (All Engineered Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.755** |
| **ROC‑AUC** | **0.818** |
| **Number of Features** | 25 (including the target) |
| **Top‑5 XGBoost Gain Importance** | 1. Age_X_ExerciseAngina_X_Sex (26.2) <br>2. Age_X_ExerciseAngina (6.12) <br>3. ExerciseAngina_X_RestBP_per_age (4.53) <br>4. Age_zscore (2.64) <br>5. Age_div_MaxHR_X_Sex (2.42) |

**Observations**

* Several engineered attributes had **zero gain importance**: `Age_over_60`, `Age_over_50`, `Age_bin`, `log_age`, `Age_bin_X_ExerciseAngina`.
* Pearson correlation analysis showed **strong multicollinearity** among age‑derived features (e.g., `Age_group` ↔ `Age_squared` r≈0.89, `Age_squared` ↔ `Age_cubic` r≈0.99, etc.).  
* High‑importance features were largely interaction terms involving **Age** and **ExerciseAngina** or **Cholesterol**.

---

### 2. Feature Pruning & Re‑evaluation  

**Pruned Attributes (zero importance & redundant interaction)**  

```text
Age_over_60
Age_over_50
Age_bin
log_age
Age_bin_X_ExerciseAngina
```

These were removed directly from `df_attributes`.

**Performance after Pruning (20 remaining features)**  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.761** (↑ 0.006) |
| **ROC‑AUC** | **0.823** (↑ 0.005) |
| **Top‑5 XGBoost Gain Importance** | 1. Age_X_ExerciseAngina_X_Sex (25.2) <br>2. ExerciseAngina_X_RestBP_per_age (7.80) <br>3. Cholesterol_per_age (2.40) <br>4. Age_div_MaxHR_X_Sex (2.18) <br>5. Age_X_Sex (1.90) |

*Pruning eliminated noise without harming – indeed slightly improving – predictive performance.*

---

### 3. Statistical Relationships  

* **High‑correlation clusters** (|r| > 0.8) centered on age transformations (`Age_group`, `Age_squared`, `Age_zscore`, `Age_cubic`, `log_age`).  
* Interaction terms (`Age_X_ExerciseAngina`, `Age_X_ExerciseAngina_X_Sex`) are strongly correlated with the base interaction (`ExerciseAngina_X_RestBP_per_age`, `ExerciseAngina_X_MaxHR_per_age`).  
* The retained set still contains some correlated pairs, but XGBoost’s tree‑based learning mitigates redundancy by selecting split points that capture the shared information.

---

### 4. Robustness Check  

**Experiment:** Added Gaussian noise (10 % of the standard deviation) to the most important feature `Age_X_ExerciseAngina_X_Sex`.

| Metric | Noisy Data |
|--------|------------|
| **Accuracy** | **0.728** (−0.033) |
| **ROC‑AUC** | **0.801** (−0.022) |

*Result:* The model’s performance degrades modestly, indicating that while the top interaction feature contributes meaningfully, the overall feature set is **reasonably robust** – predictions remain above 0.70 accuracy even with perturbed key information.

---

### 5. Key Take‑aways  

| Insight | Implication |
|---------|-------------|
| **Zero‑gain features** (`Age_over_60`, `Age_over_50`, `Age_bin`, `log_age`, `Age_bin_X_ExerciseAngina`) add no predictive value. | Safe to drop – reduces dimensionality and overfitting risk. |
| **Age‑derived multicollinearity** is high, yet the most informative age representation is `Age_zscore` and interaction terms. | Retaining a single, well‑scaled age metric (`Age_zscore`) suffices. |
| **Interaction terms** involving **ExerciseAngina** and **Age** dominate importance. | Future feature‑engineering efforts should focus on clinically plausible interactions of these variables. |
| **Model robustness** is acceptable; performance does not collapse with moderate noise on top features. | The feature set is stable for deployment scenarios with measurement variability. |

---

### 6. Final Feature Set (20 columns)

```text
Age_group
Age_squared
Cholesterol_per_age
RestBP_per_age
MaxHR_per_age
Age_X_ExerciseAngina
Age_X_Cholesterol
Age_X_RestingBP
Age_X_MaxHR
Age_zscore
Age_div_MaxHR
ExerciseAngina_X_Cholesterol_per_age
ExerciseAngina_X_RestBP_per_age
ExerciseAngina_X_MaxHR_per_age
Age_X_Sex
Age_X_ExerciseAngina_X_Sex
Age_zscore_X_Sex
Age_cubic
Age_div_MaxHR_X_Sex
target
```

These attributes collectively deliver **~76 % accuracy** and **~0.82 ROC‑AUC**, with a clear ranking of importance and manageable redundancy.

---

**Prepared by:** Tester Agent – Feature Evaluation Loop.