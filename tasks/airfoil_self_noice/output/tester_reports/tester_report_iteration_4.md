**Feature Evaluation Report – Airfoil Self‑Noise Dataset**

---

### 1. Objective  
Assess the predictive usefulness of the provided attribute set for estimating the sound‑pressure level (`target`) and identify a compact, high‑performing subset of features.

---

### 2. Methodology  

| Step | Description |
|------|-------------|
| **Data inspection** | Loaded the full `df_attributes` (49 columns, including `target`). |
| **Initial modelling** | Trained an XGBoost regressor (500 trees, depth 6, learning‑rate 0.05, GPU `cuda:5`, `tree_method='hist'`) on an 80/20 train‑test split. Evaluated RMSE and gain‑based feature importance. |
| **Correlation analysis** | Computed absolute Pearson correlations among all features. Identified many pairs with > 0.95 correlation (e.g., `angle_of_attack` ↔ `angle_in_radians`, `free_stream_velocity` ↔ `Mach_number`, etc.). |
| **Pruning decision** | Kept raw physical variables and the most important derived features (as indicated by gain). Removed 28 highly redundant or low‑importance attributes (e.g., `log_frequency`, `Reynolds_number`, `angle_rad_*`, `cos_angle_*`, etc.). |
| **Post‑pruning modelling** | Retrained the same XGBoost configuration on the reduced set (20 features). Compared performance to the full‑feature baseline. |
| **Documentation** | Recorded observations via `take_note_tool`. |

---

### 3. Results  

| Model | # Features (incl. target) | RMSE (lower is better) |
|-------|---------------------------|------------------------|
| **Full feature set** (48 predictors) | 48 | **1.36** |
| **Pruned feature set** (20 predictors) | 20 | **1.51** |

**Top 10 gain‑importance features after pruning**

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `angle_strouhal_interaction` | 120.63 |
| 2 | `sin_angle_strouhal_interaction` | 115.65 |
| 3 | `freq_chord_product` | 33.36 |
| 4 | `log_chord_length` | 28.52 |
| 5 | `strouhal_number` | 20.36 |
| 6 | `freq_disp_interaction` | 17.23 |
| 7 | `angle_in_radians` | 16.82 |
| 8 | `sin_angle` | 15.10 |
| 9 | `chord_length` | 12.46 |
|10 | `angle_disp_interaction` | 12.31 |

**Key observations (from notes)**  

* The original model achieved an excellent RMSE of **1.36**.  
* Strong multicollinearity existed among angle‑derived variables (`angle_of_attack`, `angle_in_radians`, `sin_angle`, `cos_angle`) and velocity‑derived variables (`free_stream_velocity`, its log, square, and Mach number).  
* Pruning these redundant attributes reduced the feature count by **~58 %** while only modestly degrading performance (RMSE rose to **1.51**).  
* The most predictive signals are interaction terms that couple angle with the Strouhal number (`*_strouhal_interaction`) and the frequency‑chord product, confirming the physical relevance of aerodynamic‑acoustic coupling.  
* Simple raw variables (`chord_length`, `angle_of_attack`) still retain importance after pruning, indicating they contribute unique information beyond the derived interactions.

---

### 4. Conclusions  

* **Predictive power** remains high with a streamlined set of 20 features (RMSE ≈ 1.5).  
* **Feature importance** is dominated by angle‑Strouhal interactions, frequency‑chord products, and logarithmic chord length.  
* **Redundancy** was effectively removed by discarding highly correlated derived metrics, simplifying the model without major loss of accuracy.  
* The pruned attribute list is ready for downstream modelling or for hand‑off to the Scientist/Extractor agents for further hypothesis generation.

--- 

*Prepared by the Tester Agent*