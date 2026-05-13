**Comprehensive Evaluation Report – Jungle Chess Encounter Prediction**

---

### 1. Baseline Model (All Features)

| Metric | Value |
|--------|-------|
| **Model** | XGBoost (200 trees, depth 5, `tree_method='hist'`) |
| **Features Used** | 22 (original + engineered) |
| **Accuracy** | **0.9391** |
| **Top‑10 Feature Importances** (gain) | 1. **white_dist_to_black_den** (15.3 %)<br>2. **black_dist_to_white_den** (12.9 %)<br>3. **strength_difference** (12.7 %)<br>4. **white_near_trap** (7.99 %)<br>5. **effective_strength_difference** (7.11 %)<br>6. **black_near_trap** (6.62 %)<br>7. **effective_strength_diff_times_rank_offset** (6.37 %)<br>8. **white_in_trap** (6.28 %)<br>9. **manhattan_distance** (6.28 %)<br>10. **strength_diff_times_distance** (6.16 %) |

---

### 2. Statistical Relationship Analysis  

*Correlation matrix of the top 10 features* (rounded to 3 dp) highlighted:

| Feature Pair | Correlation |
|--------------|-------------|
| `strength_difference` ↔ `effective_strength_difference` | **0.906** |
| `strength_difference` ↔ `strength_diff_times_distance` | **0.897** |
| `strength_difference` ↔ `effective_strength_diff_times_rank_offset` | **0.723** |
| `effective_strength_difference` ↔ `strength_diff_times_distance` | **0.811** |
| `effective_strength_difference` ↔ `effective_strength_diff_times_rank_offset` | **0.816** |
| `strength_diff_times_distance` ↔ `effective_strength_diff_times_rank_offset` | **0.821** |

All other pairwise correlations among the top features are ≤ 0.23, indicating low redundancy elsewhere.

**Interpretation** – The four strength‑related engineered attributes are highly collinear; they convey essentially the same information.

---

### 3. Feature Pruning  

Based on the redundancy analysis, the following three attributes were removed:

* `effective_strength_difference`  
* `strength_diff_times_distance`  
* `effective_strength_diff_times_rank_offset`

> **Action** – Executed `attribute_pruning_tool` to prune the three attributes.

**Resulting Feature Set:** 19 attributes (original 22 – 3 pruned).

---

### 4. Post‑Pruning Model Performance  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.9386** (Δ = ‑0.0005) |
| **Number of Features** | 19 |
| **Top‑10 Importances** (post‑pruning) | 1. `white_dist_to_black_den` (18.6 %)<br>2. `black_dist_to_white_den` (15.7 %)<br>3. `strength_difference` (14.5 %)<br>4. `white_near_trap` (10.1 %)<br>5. `black_near_trap` (8.2 %)<br>6. `manhattan_distance` (7.44 %)<br>7. `white_in_trap` (7.30 %)<br>8. `effective_strength_diff_times_manhattan` (6.22 %)<br>9. `file_offset` (3.48 %)<br>10. `black_in_river` (2.63 %) |

*The predictive power is essentially unchanged, confirming that the pruned attributes were redundant.*

---

### 5. Robustness Testing  

**Procedure:** Added Gaussian noise (σ = 0.1 × feature std) to the four original strength‑related features (the highly correlated group) and re‑trained the model.

| Scenario | Accuracy |
|----------|----------|
| Baseline (no noise) | **0.9391** |
| Noisy strength features | **0.9308** |

*Impact:* Accuracy dropped by only **0.0083**, indicating the model is robust to moderate perturbations of these features.

---

### 6. Conclusions & Recommendations  

1. **Predictive Power** – The engineered feature set yields high classification performance (≈ 94 % accuracy).  
2. **Key Predictors** – Distance‑related attributes (`white_dist_to_black_den`, `black_dist_to_white_den`) and the raw `strength_difference` dominate importance, suggesting spatial relationship is crucial.  
3. **Redundancy** – Four strength‑derived features are almost collinear; removing three of them does **not** degrade performance.  
4. **Feature Set Size** – Pruning reduced the dimensionality from 22 → 19 without loss, simplifying the model and interpretation.  
5. **Robustness** – The model tolerates realistic noise in the strength features, confirming stability.  

**Next Step for the Team:**  
- Retain the pruned 19‑feature set for downstream modeling.  
- Focus future investigations on additional orthogonal attributes (e.g., trap‑related dynamics) that may capture unexplored variance.  

*All observations have been recorded via `take_note_tool` for reference.*