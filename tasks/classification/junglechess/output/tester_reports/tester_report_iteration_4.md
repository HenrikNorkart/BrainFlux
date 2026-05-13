**Comprehensive Feature‑Evaluation Report – JungleChess Encounter Prediction**

---

### 1.  Experimental Setup
* **Model:** `XGBClassifier` (n_estimators = 200, max_depth = 4, learning_rate = 0.1, objective = binary:logistic, eval_metric = logloss, tree_method = ‘hist’, verbosity = 0).  
* **Data split:** Stratified 20 % hold‑out (train‑test split, `random_state=42`).  
* **Target encoding:** `white` → 1, `black` → 0.  
* **Metrics recorded:** Accuracy, ROC‑AUC, Log‑Loss.  
* **Feature‑importance methods:**  
  * XGBoost **gain** (tree‑based importance).  
  * **Permutation importance** (scikit‑learn, scoring = negative log‑loss).  

---

### 2.  Baseline Performance (all features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.9549** |
| ROC‑AUC | **0.9928** |
| Log‑Loss | **0.1320** |

The model already exhibits very strong predictive power for the binary outcome.

---

### 3.  Importance Analysis  

#### 3.1 Gain (top 10)

| Feature | Gain |
|---------|------|
| `white_dist_to_black_den` | 162.86 |
| `black_strength_4` | 152.35 |
| `white_strength_6` | 134.32 |
| `strength_difference` | 117.85 |
| `test_attr` | 109.63 |
| `black_dist_to_white_den` | 108.76 |
| `white_near_trap` | 102.12 |
| `black_near_trap` | 70.12 |
| `black_strength_6` | 63.33 |
| `effective_strength_difference` | 57.58 |

#### 3.2 Permutation Importance (top 10)

| Feature | Mean Δ (log‑loss) |
|---------|-------------------|
| `white_dist_to_black_den` | **0.957** |
| `black_dist_to_white_den` | 0.447 |
| `manhattan_distance` | 0.160 |
| `strength_difference` | 0.108 |
| `black_near_trap` | 0.077 |
| `effective_strength_difference` | 0.067 |
| `white_near_trap` | 0.038 |
| `white_strength_6` | 0.024 |
| `effective_strength_diff_times_rank_offset` | 0.021 |
| `black_strength_6` | 0.017 |

*Both methods agree that distance‑based and strength‑derived features dominate predictive relevance.*  

#### 3.3 Low‑Impact Features (gain < 5)

| Feature | Gain |
|---------|------|
| `white_in_river` | 2.80 |
| `black_strength_0` | 2.62 |
| `white_stronger_flag` | 0.00 |
| `mouse_vs_elephant_white` | 0.00 |
| `mouse_vs_elephant_black` | 0.00 |
| `white_in_den` | 0.00 |
| `effective_white_stronger_flag` | 0.00 |
| `black_in_den` | 0.00 |
| `black_mouse_vs_white_elephant_safe` | 0.00 |

These attributes contributed negligibly to model performance.

---

### 4.  Feature Pruning  

Using the **attribute_pruning_tool**, the nine low‑impact attributes listed above were removed.

**Post‑pruning performance**

| Metric | Value |
|--------|-------|
| Accuracy | **0.9566** |
| ROC‑AUC | **0.9929** |
| Log‑Loss | **0.1310** |

*Performance is essentially unchanged (slight improvement), confirming that the pruned features were non‑informative.*

---

### 5.  Key Take‑aways  

1. **Predictive Power:** The engineered distance and strength interaction features enable near‑perfect discrimination (AUC ≈ 0.993).  
2. **Dominant Predictors:**  
   * `white_dist_to_black_den` and `black_dist_to_white_den` (spatial proximity).  
   * `strength_difference` and derived “effective strength” metrics.  
   * `test_attr` (a high‑gain engineered attribute).  
3. **Redundant / No‑Value Features:** The original raw flags and several “mouse vs elephant” binaries have zero gain and can be safely removed without harming accuracy.  
4. **Feature Set Size:** After pruning, 31 high‑impact attributes remain, a manageable number for downstream modeling or interpretability work.  

---

### 6.  Recommendations for the Team  

* **Keep** all high‑gain distance/strength features and the `test_attr` column.  
* **Discard** the nine pruned attributes (they add noise and increase dimensionality).  
* **Future work** (outside the scope of this evaluation) could explore whether any additional engineered interactions among the retained features further boost robustness, but the current set already yields excellent predictive performance.  

---  

*All observations have been recorded via the `take_note_tool` for reference.*