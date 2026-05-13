**Feature‑Testing Report – Wine Quality Regression**

---

### 1. Experimental Setup  
* **Model:** XGBoostRegressor (GPU, `device="cuda:5"`, `tree_method="hist"`).  
* **Data Split:** 80 % training / 20 % test (random_state = 42).  
* **Metrics:** RMSE (lower = better) and R² (higher = better).  
* **Procedures:**  
  1. Baseline model with all 18 engineered attributes.  
  2. Computed **gain‑based feature importance** and **permutation importance**.  
  3. Evaluated **pairwise Pearson correlations** (|ρ| > 0.8 flagged as redundant).  
  4. For each highly correlated pair, retained the attribute with the higher gain score and pruned the other.  
  5. Re‑trained the model on the reduced set and re‑measured performance.

---

### 2. Baseline Results (18 features)

| Metric | Value |
|--------|-------|
| RMSE   | **0.665** |
| R²     | **0.397** |

**Top gain importance (baseline)**  
1. `alcohol_squared` (14.66)  
2. `log_alcohol` (14.15)  
3. `sulphates_to_volatile_ratio` (5.56)  
4. `alcohol_vs_volatile_ratio` (3.79)  
5. `log_free_so2` (3.40) …  

**High‑correlation pairs (|ρ| > 0.8)**  

| Pair | Correlation | Higher‑gain keeper |
|------|--------------|--------------------|
| `sulphates_times_alcohol` vs `log_sulphates` | 0.90 | **sulphates_times_alcohol** |
| `alcohol_squared` vs `log_alcohol` | 0.99 | **alcohol_squared** |
| `volatile_acidity_squared` vs `alcohol_times_volatile` | 0.93 | **volatile_acidity_squared** |
| `volatile_acidity_squared` vs `fixed_acidity_times_volatile_acidity` | 0.89 | **volatile_acidity_squared** |
| `volatile_acidity_squared` vs `pH_times_volatile_acidity_squared` | 0.998 | **pH_times_volatile_acidity_squared** |
| `alcohol_times_volatile` vs `fixed_acidity_times_volatile_acidity` | 0.88 | **fixed_acidity_times_volatile_acidity** |
| `alcohol_times_volatile` vs `pH_times_volatile_acidity_squared` | 0.92 | **pH_times_volatile_acidity_squared** |
| `log_total_so2` vs `log_total_so2_squared` | 0.99 | **log_total_so2_squared** |
| `fixed_acidity_times_volatile_acidity` vs `pH_times_volatile_acidity_squared` | 0.88 | **pH_times_volatile_acidity_squared** |

---

### 3. Pruned Feature Set  

Attributes removed (lower gain within each redundant pair):

* `log_sulphates`  
* `log_alcohol`  
* `alcohol_times_volatile`  
* `volatile_acidity_squared`  
* `fixed_acidity_times_volatile_acidity`  
* `log_total_so2`

**Remaining features (12):**  
`alcohol_squared`, `alcohol_vs_volatile_ratio`, `pH_times_volatile_acidity_squared`, `pH_times_log_alcohol`, `sulphates_to_volatile_ratio`, `log_free_so2`, `free_to_total_so2_ratio`, `total_so2_times_volatile_acidity_squared`, `sulphates_times_alcohol`, `citric_acid_times_log_alcohol`, `log_total_so2_squared`, `fixed_acidity_times_log_total_so2`.

---

### 4. Post‑Pruning Results (12 features)

| Metric | Value |
|--------|-------|
| RMSE   | **0.668** |
| R²     | **0.393** |

**Top gain importance (pruned set)**  

1. `alcohol_squared` (11.33)  
2. `alcohol_vs_volatile_ratio` (4.89)  
3. `pH_times_volatile_acidity_squared` (3.53)  
4. `pH_times_log_alcohol` (3.44)  
5. `sulphates_to_volatile_ratio` (3.43) …  

*Performance impact* – RMSE increased by **≈0.003** (0.5 % relative) and R² dropped by **≈0.004** (1 % relative). The loss is negligible given the 33 % reduction in feature count.

---

### 5. Conclusions & Recommendations  

1. **Predictive Power:** The engineered attributes collectively provide moderate predictive ability (R² ≈ 0.39). The strongest predictors are the alcohol‑related transformations (`alcohol_squared`, `alcohol_vs_volatile_ratio`) and interaction terms involving pH and volatile acidity.  

2. **Feature Redundancy:** High pairwise correlations created redundant information. Removing the lower‑importance member of each correlated pair retained essentially the same predictive performance while simplifying the model.  

3. **Pruned Feature Set:** The 12‑feature subset is **recommended** for downstream modeling – it is more parsimonious, easier to interpret, and incurs no meaningful loss in accuracy.  

4. **Next Steps (outside current scope):** Further robustness checks (noise injection, cross‑validation) could confirm stability, but the current evidence already supports the reduced feature set.

--- 

*All observations have been recorded via the note‑taking tool for traceability.*