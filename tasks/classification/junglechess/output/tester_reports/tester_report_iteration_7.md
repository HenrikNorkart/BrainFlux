**Comprehensive Feature Evaluation Report – Jungle Chess Encounter Dataset**

---

### 1. Dataset Overview
- **Rows:** 44,819  
- **Columns (including target):** 55  
- **Target:** `target` (categorical – “white” vs “black”)

The raw feature set already contains many engineered attributes (distance, strength differences, interaction flags, etc.).

---

### 2. Experimental Setup
| Step | Methodology |
|------|--------------|
| **Train‑test split** | 80 % / 20 % stratified on the target |
| **Model** | XGBoost (binary:logistic) – GPU (`device="cuda:5"`, `tree_method="hist"`), 200 trees, max depth 6, learning rate 0.1 |
| **Metric** | Classification accuracy (primary) and full classification report |
| **Feature importance** | XGBoost gain importance (`model.get_booster().get_score(importance_type='gain')`) |
| **Correlation analysis** | Pearson absolute correlation, flagging pairs > 0.90 |

All code was executed via the provided Python executor, ensuring reproducibility.

---

### 3. Baseline Results (All 55 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.9856** |
| **Weighted F1** | 0.9856 |
| **Key observations** | The model already achieved near‑perfect discrimination. |

**Top‑20 most important features (gain):**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `test_double_white_strength` | 106.81 |
| 2 | `black_strength_4` | 76.64 |
| 3 | `white_dist_to_black_den` | 57.94 |
| 4 | `white_effective_distance` | 45.64 |
| 5 | `black_dist_to_white_den` | 37.58 |
| 6 | `white_strength_7` | 35.45 |
| 7 | `strength_difference` | 34.45 |
| 8 | `white_near_trap` | 30.67 |
| 9 | `black_effective_distance` | 29.99 |
| 10 | `white_strength_6` | 24.43 |
| … | … | … |

**Low‑importance / zero‑importance features (gain ≤ 0.5):**  

`white_in_den`, `mouse_vs_elephant_black`, `black_in_den`, `mouse_vs_elephant_white`, `lion_tiger_jump_capture_flag`, `capture_distance_difference`, `white_first_to_capture`, `eff_strength_diff_times_capture_dist_diff`, `black_strength_0` (gain 0.46), plus several very low contributors.

**Highly correlated pairs (|ρ| > 0.90):**  

- `test_attr` ↔ `test_double_white_strength` ↔ `test_div_one` (perfect correlation)  
- `capture_distance_white` ↔ `capture_distance_black` (identical)  
- `strength_difference` ↔ `effective_strength_difference` (ρ ≈ 0.91)  
- `white_dist_to_black_den` ↔ `white_effective_distance` (ρ ≈ 0.95)  
- `manhattan_distance` ↔ `capture_distance_white/black` (ρ ≈ 0.94)  

These redundancies suggest that some attributes can be removed without harming performance.

---

### 4. Feature Pruning

**Attributes removed (zero or negligible importance, or perfect redundancy):**  

```
white_in_den,
mouse_vs_elephant_black,
black_in_den,
mouse_vs_elephant_white,
lion_tiger_jump_capture_flag,
capture_distance_difference,
white_first_to_capture,
eff_strength_diff_times_capture_dist_diff,
black_strength_0,
test_attr,
test_div_one,
capture_distance_black
```

Total features after pruning: **43**.

---

### 5. Post‑pruning Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.9848** (drop of < 0.2 %) |
| **Weighted F1** | ≈ 0.985 |
| **Remaining feature set** | 43 attributes (list provided below) |

The negligible loss in predictive power confirms that the pruned attributes contributed little or were duplicate information.

**Remaining key features (still top contributors):**  

`test_double_white_strength`, `black_strength_4`, `white_dist_to_black_den`, `white_effective_distance`, `black_dist_to_white_den`, `white_strength_7`, `strength_difference`, `white_near_trap`, `black_effective_distance`, `white_strength_6`, …

---

### 6. Conclusions & Recommendations for the Team

1. **Predictive Power** – The current engineered feature set is highly predictive (≈ 98 % accuracy).  
2. **Feature Importance** – A relatively small subset (≈ 15 %) drives the majority of the model’s decisions; these should be the focus of any downstream analysis.  
3. **Redundancy** – Several groups of features are perfectly correlated; keeping only the highest‑importance member (e.g., `test_double_white_strength` instead of `test_attr`/`test_div_one`) simplifies the model without loss.  
4. **Pruned Feature Set** – After removing 12 low‑value/redundant attributes, the model retains its performance while being more compact (43 → 43 features).  
5. **Next Steps for Scientist & Extractor** –  
   * **Scientist:** Concentrate hypothesis generation on the top‑ranked attributes (strength differentials, distance to opponent’s den, mobility counts, and the `test_double_white_strength` composite).  
   * **Extractor:** Future extractions can prioritize these high‑impact attributes and avoid generating the now‑pruned duplicates.  

---

### 7. Final Feature List (post‑pruning)

```
strength_difference, white_stronger_flag, file_offset, rank_offset,
manhattan_distance, strength_diff_times_distance, effective_strength_difference,
effective_white_stronger_flag, white_in_trap, white_in_river, black_in_river,
white_near_trap, black_near_trap, white_dist_to_black_den, black_dist_to_white_den,
effective_strength_diff_times_manhattan, effective_strength_diff_times_rank_offset,
effective_strength_diff_times_file_offset, white_strength_0, white_strength_4,
white_strength_5, white_strength_6, white_strength_7, black_strength_4,
black_strength_5, black_strength_6, black_strength_7, white_own_den_distance,
black_own_den_distance, white_mobility_count, black_mobility_count,
white_mouse_vs_black_elephant_safe, black_mouse_vs_white_elephant_safe,
white_effective_distance, black_effective_distance, test_double_white_strength,
test_mul_two_columns, white_strength_over_effdist, black_strength_over_effdist,
adjacent_capture_flag, capture_distance_white, capture_distance_ratio
```

These 43 attributes constitute an efficient, high‑performing feature set for predicting jungle‑chess encounter outcomes.