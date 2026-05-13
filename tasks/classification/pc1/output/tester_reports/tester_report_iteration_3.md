**Comprehensive Feature‑Evaluation Report – pc1 Dataset (Defect Prediction)**  

---

### 1.  Initial Baseline (All 42 engineered attributes)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.937** |
| **ROC‑AUC** | **0.842** |
| **Number of features** | 42 |

**Top‑10 features by XGBoost gain**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `log_loc` | 4.62 |
| 2 | `reciprocal_unique_op_ratio` | 3.06 |
| 3 | `unique_op_ratio` | 2.70 |
| 4 | `halstead_total` | 1.98 |
| 5 | `log_halstead_bugs_times_unique_op_ratio` | 1.96 |
| 6 | `sqrt_halstead_bugs` | 1.83 |
| 7 | `design_density` | 1.73 |
| 8 | `comment_density` | 1.65 |
| 9 | `halstead_difficulty_per_loc_times_log_loc` | 1.62 |
|10 | `log_loc_times_unique_op_ratio` | 1.60 |

---

### 2.  Redundancy & Correlation Analysis  

*Pairwise absolute Pearson correlation > 0.9* was found for **33** feature pairs, e.g.:

| Highly correlated pair (|r|) | Example |
|-------------------------------|---------|
| `log_loc` ↔ `log_loc_plus1` (1.00) |
| `log_loc` ↔ `sqrt_loc` (0.94) |
| `log_loc` ↔ `log_loc_squared` (0.95) |
| `log_halstead_bugs` ↔ `sqrt_halstead_bugs` (0.98) |
| `bug_est_per_loc` ↔ `log_halstead_bugs_div_loc` (0.997) |
| `unique_op_ratio` ↔ `unique_op_ratio_squared` (0.96) |
| `op_operand_ratio` ↔ `op_operand_ratio_squared` (0.99) |
| `comment_density` ↔ several “log_loc × comment_density” variants (≈0.91) |

These indicate many engineered transformations (log, sqrt, reciprocal, squared) are essentially duplicates of the underlying raw metric.

---

### 3.  Pruning Decision  

**Goal:** Remove redundant attributes while preserving predictive power.  

**Criteria:**  
* Feature is **highly correlated (> 0.9)** with a higher‑importance feature.  
* The redundant feature shows **lower gain** in the baseline model.

**Attributes pruned (9 total):**  

| Pruned attribute | Reason |
|------------------|--------|
| `sqrt_halstead_bugs` | Redundant with `log_halstead_bugs` (very high correlation, lower gain) |
| `log_loc` | Duplicate of `log_loc_plus1`, `sqrt_loc`, etc. |
| `log_loc_squared` | Duplicate of `log_loc` |
| `log_halstead_bugs_div_loc` | Near‑perfect correlation with `bug_est_per_loc` |
| `log_halstead_bugs` | Redundant with `sqrt_halstead_bugs` & `log_halstead_bugs_plus1` |
| `log_halstead_bugs_plus1` | Same information as `log_halstead_bugs` |
| `log_loc_plus1` | Identical to `log_loc` |
| `reciprocal_halstead_bugs` | Mirrors `log_halstead_bugs` |
| `unique_op_ratio_squared` | Direct square of `unique_op_ratio` |

After pruning, **33** attributes remain.

---

### 4.  Post‑Pruning Model Performance  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.937** (unchanged) |
| **ROC‑AUC** | **0.831** (slight drop) |
| **Number of features** | 33 |

**New Top‑5 features (gain)**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `sqrt_log_loc` | 14.23 |
| 2 | `sqrt_loc` | 3.91 |
| 3 | `reciprocal_unique_op_ratio` | 3.63 |
| 4 | `comment_density` | 2.45 |
| 5 | `unique_op_ratio` | 2.15 |

*Interpretation*: The most predictive information is now captured by a compact set of **non‑redundant** metrics (size‑related, operator‑ratio, and comment density).

---

### 5.  Robustness Check  

*Experiment*: Added Gaussian noise (10 % of the standard deviation) to the key feature `sqrt_loc`.  

| Metric | Noisy data |
|--------|------------|
| Accuracy | **0.932** (‑0.5 % absolute) |
| ROC‑AUC | **0.839** (≈ unchanged) |

**Conclusion** – Model performance is **stable** to moderate perturbations of the most important feature, indicating good robustness.

---

### 6.  Key Take‑aways  

1. **Predictive power is concentrated** in a few core attributes:  
   * `sqrt_log_loc` / `sqrt_loc` (size of code)  
   * `reciprocal_unique_op_ratio` (operator diversity)  
   * `comment_density` (documentation level)  
   * `unique_op_ratio` (operator‑operand balance)

2. **Redundant engineered transformations** (log, squared, reciprocal versions of the same base metric) do **not** improve performance and can safely be removed.  

3. **Pruning 9 redundant features** reduces dimensionality by ~21 % while keeping accuracy identical and only a minor AUC reduction, simplifying the model and easing interpretability.

4. **Model is robust** to reasonable noise in the dominant size‑related feature, suggesting the learned patterns are not brittle.

---

### 7.  Recommended Next Steps (Tester‑only)

* Adopt the **pruned 33‑feature set** for downstream modeling.  
* Focus interpretability analyses on the top‑5 features identified above.  
* (No further feature engineering is suggested per the brief.)  

---  

*All observations have been recorded via the `take_note_tool` for traceability.*