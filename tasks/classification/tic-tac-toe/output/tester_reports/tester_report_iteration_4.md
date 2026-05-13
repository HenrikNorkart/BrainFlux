**Comprehensive Feature Evaluation Report – Tic‑Tac‑Toe Dataset**

---

### 1. Initial Model Performance  
- **Algorithm:** XGBoost (GPU, `device="cuda:5"`, `tree_method="hist"`).  
- **Train‑test split:** 80 % / 20 % (stratified).  
- **Target encoding:** `positive → 1`, `negative → 0`.  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.9948** |
| **Precision (positive)** | 0.9921 |
| **Recall (positive)** | 1.0000 |
| **F1‑score (positive)** | 0.9960 |
| **Macro‑avg F1** | 0.9942 |

*The model already achieves near‑perfect discrimination.*

---

### 2. Feature Importance (Gain) – Top 10  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | **X_minus_O_diff** | 85.04 |
| 2 | **total_X_count** | 4.28 |
| 3 | **blocked_line_count** | 1.11 |
| 4 | **edge_X_count** | 0.95 |
| 5 | **center_is_X** | 0.76 |
| 6 | **O_wins_any_line** | 0.59 |
| 7 | **corner_O_count** | 0.49 |
| 8 | **O_count_anti_diag** | 0.37 |
| 9 | **X_wins_any_line** | 0.33 |
|10 | **X_count_main_diag** | 0.33 |

*The dominant predictor is the difference between X and O counts (`X_minus_O_diff`). All other important features relate to the presence of X/O in strategic board positions (wins, blocks, edges, centre).*

---

### 3. Inter‑Feature Correlations (selected top features)

| Feature Pair | Correlation |
|--------------|-------------|
| X_minus_O_diff ↔ O_wins_any_line | **‑0.72** |
| X_minus_O_diff ↔ X_wins_any_line | **0.65** |
| edge_X_count ↔ corner_O_count | **0.52** |
| total_X_count ↔ blocked_line_count | **0.66** |
| center_is_X ↔ X_count_main_diag | **0.45** |

*Most correlations are modest; only the strong negative link between `X_minus_O_diff` and `O_wins_any_line` reflects the logical opposition of X vs. O winning lines.*

---

### 4. Pruning Low‑Impact Features  

- **Criterion:** Gain < 0.1 (20 features).  
- **Pruned attributes:**  

`X_potential_win_any_line, O_potential_win_any_line, total_O_count, O_count_top_row, O_count_left_col, O_count_mid_col, O_count_right_col, O_count_main_diag, center_is_O, corner_X_count, edge_O_count, corner_X_minus_O, edge_X_to_corner_X_ratio, edge_O_to_corner_O_ratio, X_fork_possible, total_edge_to_corner_ratio, X_two_in_row_blank_count, O_two_in_row_blank_count, X_fork_strict, total_blank_squares`

- **Resulting feature count:** **22** (down from 42).

**Effect of pruning**

| Metric | Before Pruning | After Pruning |
|--------|----------------|---------------|
| Accuracy | 0.9948 | **0.9948** |
| No. of features | 42 | **22** |

*Performance remained identical, confirming that the removed attributes contributed negligible predictive information.*

---

### 5. Robustness Test – Noise Injection  

- Added Gaussian noise (σ = 0.1) to all numeric features.  
- Re‑trained XGBoost on the noisy data.

| Metric | Value |
|--------|-------|
| **Accuracy (noisy)** | **0.9844** |

*Even with moderate perturbation the model retains >98 % accuracy, indicating strong robustness of the retained feature set.*

---

### 6. Key Take‑aways  

1. **Predictive Power:** The current feature set (22 attributes) predicts the game outcome with ~99.5 % accuracy—essentially optimal for this dataset.  
2. **Dominant Feature:** `X_minus_O_diff` alone captures the majority of the signal (gain ≈ 85).  
3. **Redundant/Low‑Impact Features:** 20 attributes contributed virtually no gain and were safely removed, simplifying the model without loss of performance.  
4. **Feature Interactions:** Moderate correlations exist, but no severe multicollinearity; important features complement each other (e.g., counts, win‑line flags, centre/edge presence).  
5. **Robustness:** Model tolerates added noise, confirming stability of the selected attributes.  

---

### 7. Recommendations (internal)

- **Maintain** the 22‑feature subset for downstream modeling or deployment.  
- **Monitor** `X_minus_O_diff` and `total_X_count` as primary drivers; any drift in these may signal data distribution changes.  
- **Future work** (outside current scope) could explore interaction terms, but the current set already achieves near‑perfect classification.  

--- 

*All observations have been recorded in the internal notes for reference.*