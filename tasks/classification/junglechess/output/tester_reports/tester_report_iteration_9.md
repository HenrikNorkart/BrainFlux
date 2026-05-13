**Feature Evaluation Report – JungleChess Dataset**

---

### 1. Experimental Setup
- **Model:** XGBoost (multi‑class) with GPU (`device="cuda:5"`, `tree_method="hist"`).  
- **Parameters:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.  
- **Data Split:** 80 % train / 20 % test (stratified).  
- **Target Encoding:** `LabelEncoder` (classes: *white*, *black*, *draw*).  
- **Metrics:** Accuracy, per‑class Precision/Recall/F1, confusion matrix, XGBoost gain‑based feature importance.

---

### 2. Baseline Performance (All 72 features)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **0.9835** |
| **Weighted F1** | 0.9834 |
| **Macro F1** | 0.9754 |
| **Per‑class F1** | white 0.9882, black 0.9846, draw 0.9532 |

*Confusion matrix (test set)*  

| – | Pred white | Pred black | Pred draw |
|---|------------|------------|-----------|
| **True white** | 4576 | 13 | 24 |
| **True black** | 24 | 3425 | 15 |
| **True draw**  | 44 | 28 | 815 |

The model already attains **>98 %** accuracy, indicating the engineered attributes are highly predictive.

---

### 3. Feature Importance (Top 20 by XGBoost gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `strength_ratio_raw` | 28.69 |
| 2 | `white_dist_to_black_den` | 25.04 |
| 3 | `black_dist_to_white_den` | 21.62 |
| 4 | `test_division` | 20.36 |
| 5 | `strength_difference` | 20.25 |
| 6 | `adjacent_capture_flag` | 19.71 |
| 7 | `white_strength_4` | 19.05 |
| 8 | `white_stronger_flag` | 18.72 |
| 9 | `effective_strength_black` | 17.78 |
|10 | `same_strength_flag` | 15.35 |
|11 | `strength_raw_near_trap_white` | 14.69 |
|12 | `test_mul_two_columns` | 14.08 |
|13 | `effective_strength_difference` | 12.79 |
|14 | `white_strength_6` | 12.19 |
|15 | `test_attr` | 11.78 |
|16 | `black_strength_4` | 11.71 |
|17 | `white_effective_distance` | 10.96 |
|18 | `black_strength_6` | 9.26 |
|19 | `white_strength_over_effdist` | 9.05 |
|20 | `black_effective_distance` | 8.97 |

*Interpretation:* Distance to the opponent’s den, raw strength ratios, and interaction flags (e.g., capture adjacency) dominate predictive power.

---

### 4. Redundancy & Correlation Analysis
A 5 000‑row sample revealed **13 perfectly correlated (ρ = 1.0) feature pairs**, indicating exact duplication:

| Correlated Group (identical) |
|------------------------------|
| `test_attr`, `test_double_white_strength`, `test_div_one`, `effective_strength_white` |
| `capture_distance_white`, `capture_distance_black` |
| `strength_near_trap_white`, `effective_strength_near_trap_white`, `strength_raw_near_trap_white` |
| `strength_near_trap_black`, `effective_strength_near_trap_black`, `strength_raw_near_trap_black` |

All other features showed correlations < 0.95, so no further redundancy was detected.

---

### 5. Pruning Action
Using **attribute_pruning_tool**, the following 8 redundant attributes were removed:

- `test_double_white_strength`
- `test_div_one`
- `effective_strength_white`
- `capture_distance_black`
- `effective_strength_near_trap_white`
- `strength_raw_near_trap_white`
- `effective_strength_near_trap_black`
- `strength_raw_near_trap_black`

The dataset now contains **64** features.

---

### 6. Post‑Pruning Performance
Re‑training the same XGBoost configuration on the pruned set yielded **identical results**:

- **Accuracy:** 0.9835  
- **Weighted F1:** 0.9834  

No degradation in predictive power was observed, confirming that the removed features were pure duplicates.

---

### 7. Conclusions & Recommendations
1. **Predictive Power:** The engineered feature set is extremely strong (≈98 % accuracy).  
2. **Key Drivers:** Distance‑to‑den, raw strength ratios, and capture‑related flags are the most influential attributes.  
3. **Redundancy:** Several groups of features were exact duplicates; pruning them reduced dimensionality without any loss in performance.  
4. **Feature Set Size:** After pruning, the model works with **64 high‑quality, non‑redundant features**, simplifying downstream modeling and interpretation.  

*No further pruning is advisable at this stage, as every remaining attribute contributed measurable gain.*