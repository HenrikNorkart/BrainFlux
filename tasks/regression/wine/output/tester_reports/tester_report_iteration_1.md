**Feature Evaluation Report – Wine Quality Regression (Engineered Attributes)**  

**1. Experimental Setup**  
- **Model:** Linear Regression (no regularisation) – chosen for its interpretability of feature contributions.  
- **Data Split:** 80 % training / 20 % test (random_state = 42).  
- **Metrics (computed manually):**  
  - **RMSE** = √Mean[(y‑true – y‑pred)²]  
  - **R²** = 1 – Σ(y‑true – y‑pred)² / Σ(y‑true – ȳ)²  

**2. Baseline Results (All 8 engineered features)**  

| Metric | Value |
|--------|-------|
| RMSE   | **0.7424** |
| R²     | **0.249** |

**Feature importance (absolute coefficient magnitude)**  

| Rank | Feature | |Coefficient| |
|------|---------------------------|----------|
| 1 | **log_alcohol** | 3.34 |
| 2 | **volatile_acidity_squared** | 0.94 |
| 3 | **log_sulphates** | 0.26 |
| 4 | **sulphates_to_volatile_ratio** | 0.08 |
| 5 | **alcohol_times_volatile** | 0.04 |
| 6 | **alcohol_squared** | 0.03 |
| 7 | **sulphates_times_alcohol** | 0.01 |
| 8 | **alcohol_vs_volatile_ratio** | 0.007 |

**Highly correlated pairs (|ρ| > 0.9)**  

- `alcohol_squared` ↔ `log_alcohol` (ρ = 0.993)  
- `volatile_acidity_squared` ↔ `alcohol_times_volatile` (ρ = 0.928)  

**3. Pruning Decision**  

- **Low‑impact features:** `alcohol_vs_volatile_ratio` and `sulphates_times_alcohol` (tiny coefficients, negligible predictive contribution).  
- **Redundant pairs:** Although `alcohol_squared` and `log_alcohol` are almost identical, `log_alcohol` carries far larger importance, so we keep it and drop the less‑important partner only if we aim for a minimal set. The same logic applies to the second pair, but dropping either yields a slight performance dip (≈ 0.005 RMSE).  

**Action taken:** Pruned **only** the two lowest‑importance attributes.  

```json
{
  "attribute_names_list": [
    "alcohol_vs_volatile_ratio",
    "sulphates_times_alcohol"
  ]
}
```

**4. Post‑pruning Results (6 remaining features)**  

| Metric | Value |
|--------|-------|
| RMSE   | **0.7427** |
| R²     | **0.2484** |

**Feature importance after pruning**

| Feature | |Coefficient| |
|---------|----------|
| log_alcohol | 3.01 |
| volatile_acidity_squared | 0.53 |
| sulphates_to_volatile_ratio | 0.16 |
| log_sulphates | 0.04 |
| alcohol_squared | 0.028 |
| alcohol_times_volatile | 0.023 |

Performance is virtually unchanged (ΔRMSE ≈ 0.0003, ΔR² ≈ ‑0.001), confirming that the removed attributes contributed negligibly.

**5. Conclusions & Recommendations**  

- **Predictive Power:** The engineered feature set predicts wine quality modestly (RMSE ≈ 0.74, R² ≈ 0.25). The dominant predictor is `log_alcohol`, followed by `volatile_acidity_squared`.  
- **Feature Redundancy:** `alcohol_squared` is almost perfectly collinear with `log_alcohol`; retaining both does not improve accuracy and may inflate variance.  
- **Pruning Outcome:** Removing the two weakest features (`alcohol_vs_volatile_ratio`, `sulphates_times_alcohol`) yields a leaner model with no measurable loss in predictive ability.  
- **Suggested Final Feature Subset (6 attributes):**  
  1. `log_alcohol`  
  2. `volatile_acidity_squared`  
  3. `log_sulphates`  
  4. `sulphates_to_volatile_ratio`  
  5. `alcohol_squared` *(optional – highly correlated with #1)*  
  6. `alcohol_times_volatile` *(optional – correlated with #2)*  

Keeping the optional pair (`alcohol_squared`, `alcohol_times_volatile`) does not harm performance but adds redundancy; they can be dropped for a more parsimonious model if desired.  

**Overall**, the current engineered attributes are adequate, with a clear hierarchy of importance and a small set of dispensable features identified for pruning.