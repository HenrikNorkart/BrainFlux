**Tester Agent – Feature Evaluation Report**  
*Dataset: junglechess (binary outcome – “white” vs “black”)*  

---

### 1. Data‑quality checks  

| Check | Result |
|-------|--------|
| Total rows | **44 819** |
| Total columns (pre‑pruning) | **78** (including *target*) |
| Zero‑variance attributes | **16** (e.g., `test_constant`, `white_piece_in_river`, `white_is_rat`, …) |
| Missing values | **None** |

*Action:* All 16 constant columns were removed with **attribute_pruning_tool**.

---

### 2. Predictive power – univariate analysis  

**Correlation with target (white = 0, black = 1)** – top absolute values  

| Feature | Pearson r |
|---------|-----------|
| `opponent_den_dist_diff` | **+0.592** |
| `white_opponent_den_dist` | **+0.448** |
| `black_opponent_den_dist` | **‑0.396** |
| `white_own_den_dist` | **‑0.396** |
| `black_own_den_dist` | **+0.346** |
| `black_trap_distance` | **‑0.287** |
| `black_adjacent_opponent_den` | **+0.245** |
| `strength_ratio_times_opponent_dist_diff` | **+0.242** |
| `black_can_capture_opponent_den` | **+0.237** |
| `white_adjacent_opponent_den` | **‑0.227** |

These distance‑related attributes dominate the linear relationship with the outcome.

---

### 3. Simple predictive rule (baseline)

*Rule:* predict **black** if `opponent_den_dist_diff > 0`, otherwise **white**.  

- **Accuracy:** **74.5 %**  
- **Interpretation:** A single engineered distance feature already captures a large portion of the signal.

A weighted sum of the five strongest features (using their correlation signs as weights) gave **71.9 %** accuracy, confirming that the leading feature alone is the most informative.

---

### 4. Inter‑feature relationships (redundancy check)

Correlation matrix of the 10 most predictive features:

|                     | opponent_den_dist_diff | white_opponent_den_dist | black_opponent_den_dist | white_own_den_dist | black_own_den_dist | black_trap_distance | black_adjacent_opponent_den | strength_ratio_times_opponent_dist_diff | black_can_capture_opponent_den | white_adjacent_opponent_den |
|---------------------|-----------------------|------------------------|------------------------|--------------------|--------------------|---------------------|-----------------------------|------------------------------------------|-------------------------------|-----------------------------|
| **opponent_den_dist_diff** | 1.00 | **0.70** | **‑0.70** | **‑0.63** | **0.65** | **‑0.48** | 0.29 | 0.42 | 0.24 | **‑0.29** |
| **white_opponent_den_dist** | 0.70 | 1.00 | 0.01 | **‑0.90** | **‑0.02** | 0.00 | 0.00 | 0.33 | 0.00 | **‑0.41** |
| **black_opponent_den_dist** | **‑0.70** | 0.01 | 1.00 | **‑0.02** | **‑0.93** | **0.68** | **‑0.41** | **‑0.27** | **‑0.34** | 0.00 |
| **white_own_den_dist** | **‑0.63** | **‑0.90** | **‑0.02** | 1.00 | 0.02 | **‑0.01** | 0.01 | **‑0.29** | 0.00 | 0.35 |
| **black_own_den_dist** | 0.65 | **‑0.02** | **‑0.93** | 0.02 | 1.00 | **‑0.46** | 0.33 | 0.25 | 0.22 | 0.01 |
| **black_trap_distance** | **‑0.48** | 0.00 | 0.68 | **‑0.01** | **‑0.46** | 1.00 | **‑0.30** | **‑0.18** | **‑0.30** | 0.00 |
| **black_adjacent_opponent_den** | 0.29 | 0.00 | **‑0.41** | 0.01 | 0.33 | **‑0.30** | 1.00 | 0.11 | **0.85** | 0.00 |
| **strength_ratio_times_opponent_dist_diff** | 0.42 | 0.33 | **‑0.27** | **‑0.29** | 0.25 | **‑0.18** | 0.11 | 1.00 | 0.09 | **‑0.14** |
| **black_can_capture_opponent_den** | 0.24 | 0.00 | **‑0.34** | 0.00 | 0.22 | **‑0.30** | **0.85** | 0.09 | 1.00 | 0.00 |
| **white_adjacent_opponent_den** | **‑0.29** | **‑0.41** | 0.00 | 0.35 | 0.01 | 0.00 | 0.00 | **‑0.14** | 0.00 | 1.00 |

*Observations*  

* `opponent_den_dist_diff` is strongly correlated with both white and black opponent‑den distances (≈ ±0.7).  
* Several “own_den_dist” and “opponent_den_dist” features are highly collinear (|r| > 0.6).  
* `black_adjacent_opponent_den` and `black_can_capture_opponent_den` are tightly linked (r = 0.85).  

These redundancies suggest that a compact subset (e.g., `opponent_den_dist_diff`, `white_own_den_dist`, `black_adjacent_opponent_den`) could capture most of the signal while reducing dimensionality.

---

### 5. Summary of Findings  

1. **Feature quality** – 16 constant attributes were removed; the remaining 62 attributes are informative.  
2. **Predictive strength** – Distance‑based features, especially `opponent_den_dist_diff`, exhibit the highest linear association with the outcome (|r| ≈ 0.59).  
3. **Baseline performance** – A single‑feature rule reaches **≈ 74 % accuracy**, establishing a strong baseline without any model training.  
4. **Redundancy** – Several top features are moderately to highly correlated; careful selection can keep predictive power while simplifying the model.  
5. **Next steps for the Scientist/Extractor** –  
   * Focus on refining distance‑related attributes (e.g., engineered ratios, binary flags for “adjacent opponent den”).  
   * Consider removing highly collinear pairs to avoid over‑parameterisation.  
   * Explore interaction terms between river‑crossing flags and distance measures, as they may capture the strategic nuance of jungle chess.

---

**Prepared by:** Tester Agent  
*All notes have been recorded via `take_note_tool`.*
