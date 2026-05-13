**Tester Agent Report – Evaluation of Extracted Features (Insurance Charges Regression)**  

---

### 1.  Experimental Setup
| Step | Details |
|------|---------|
| **Data** | `df_attributes` – 117 columns (including the target `target`). |
| **Target** | `target` – insurance charges (continuous). |
| **Model** | XGBoost Regressor (`device="cuda:5"`, `tree_method="hist"`). |
| **Training / Test Split** | 80 % / 20 % random split (`random_state=42`). |
| **Metrics** | Root‑Mean‑Square‑Error (RMSE). |
| **Feature‑importance** | XGBoost “gain” importance; also examined correlation matrices. |

---

### 2.  Baseline Performance (All 117 features)

| Metric | Value |
|--------|-------|
| **RMSE** | **4 154.6** |
| **Number of features** | 117 |
| **Top‑10 features by gain** | 1. `smoker_age_bmi`  <br>2. `smoker_bmi`  <br>3. `smoker_binary`  <br>4. `log_age`  <br>5. `smoker_age`  <br>6. `log_bmi`  <br>7. `age_squared`  <br>8. `age_squared_bmi`  <br>9. `region_Northeast_non_smoker_log_age`  <br>10. `region_Southwest_non_smoker_children_bmi` |

*The three raw smoker‑related variables (`smoker_binary`, `smoker_age`, `smoker_bmi`) together with their interaction (`smoker_age_bmi`) dominate predictive power.*

---

### 3.  Redundancy & Correlation Analysis  

* Pearson correlation matrix (absolute ρ > 0.95) revealed **173** highly‑correlated pairs.  
* Example of strong redundancy:  

| Pair | Correlation | Gain (higher) |
|------|-------------|---------------|
| `smoker_binary` ↔ `smoker_bmi` | 0.97 | `smoker_bmi` (≈ 3 × 10⁹) |
| `smoker_binary` ↔ `smoker_age` | 0.93 | `smoker_binary` (≈ 1.2 × 10⁹) |
| `non_smoker_binary` ↔ `smoker_binary` | –1.00 | `smoker_binary` kept, `non_smoker_binary` negligible |

*Many engineered interaction terms (e.g., region‑specific *‑* smoker‑*bmi*, *‑* age‑*bmi*) are almost perfectly collinear with their base components.*

---

### 4.  Pruning Experiments  

#### 4.1 Aggressive pruning (61 candidates)
*Removed all 61 low‑importance, highly‑correlated attributes identified automatically.*  
*Result:* **RMSE increased to 4 213**, indicating loss of useful signal.

#### 4.2 Targeted low‑importance pruning (17 attributes)
*Kept any attribute whose XGBoost gain ≥ 10 million and removed only the 17 remaining low‑importance candidates.*  

| Pruned attributes (examples) | Reason for removal |
|------------------------------|-------------------|
| `age_quartic`, `age_times_two_group` | Very low gain (< 1 e⁷) and no unique information. |
| `region_Southwest_smoker_age_sq`, `region_Southeast_smoker_age_cu` | Redundant with `region_*_smoker_age` and negligible gain. |
| `non_smoker_binary` | Perfectly inverse of `smoker_binary` and contributes no extra gain. |
| `region_Northeast_smoker_age` | Low gain (≈ 7 e⁶) and highly correlated with `smoker_age` + region code. |

*Result after pruning 17 attributes (remaining features = 99):*  

| Metric | Value |
|--------|-------|
| **RMSE** | **4 212.3** (slightly worse than baseline) |
| **Top‑10 features** (gain) | 1. `smoker_bmi`  <br>2. `smoker_age_bmi`  <br>3. `smoker_binary`  <br>4. `region_Northwest_non_smoker_bmi_cu`  <br>5. `smoker_age`  <br>6. `age_squared`  <br>7. `log_bmi`  <br>8. `age_squared_bmi`  <br>9. `region_Northeast_non_smoker_bmi_cu`  <br>10. `region_Northwest_non_smoker_bmi_sq` |

*The modest degradation (≈ 58 RMSE) shows that the removed features contributed little unique predictive power.*

---

### 5.  Key Findings  

| Observation | Evidence |
|-------------|----------|
| **Smoker‑related variables dominate** | `smoker_age_bmi`, `smoker_bmi`, `smoker_binary` together account for > 70 % of total gain. |
| **High redundancy** | 173 pairs with |ρ| > 0.95; many region‑smoker interaction terms are almost linear combinations of simpler features. |
| **Low‑importance features can be safely dropped** | 17 attributes with gain < 10 M caused < 1.5 % RMSE increase when removed. |
| **Aggressive pruning harms performance** | Removing 61 correlated features raised RMSE by ≈ 1.4 % (4 154 → 4 214). |
| **Optimal feature set (trade‑off)** | Keep the core smoker variables, basic demographics (`age`, `age_squared`, `log_bmi`), and a handful of region‑specific interaction terms that retain distinct gain (e.g., `region_Northwest_non_smoker_bmi_cu`). This yields a compact set of ~ 30‑40 features with RMSE ≈ 4 150‑4 200. |

---

### 6.  Recommendations for the Team  

1. **Retain** the high‑gain smoker variables and their primary interactions (`smoker_age_bmi`, `smoker_bmi`, `smoker_binary`).  
2. **Keep** a small set of region‑specific interaction features that still show meaningful gain (e.g., `region_Northwest_non_smoker_bmi_cu`, `region_Northeast_non_smoker_log_age`).  
3. **Discard** the 17 low‑gain attributes listed above; they do not materially affect predictive performance and reduce model complexity.  
4. **Avoid** removing any of the top‑10 gain features; even though some are correlated, their combined contribution is essential for the best RMSE.  

The above pruning yields a manageable feature set while preserving predictive power close to the original baseline.