**Feature Evaluation Report – NATOPS Time‑Series Classification**

---

### 1. Baseline Model (RandomForest)

| Metric | Value |
|--------|-------|
| **Accuracy** (30 % hold‑out, stratified) | **0.759** |
| **Confusion matrix** (rows = true, cols = predicted) | `[[8,1,0,0,0,0],[1,5,3,0,0,0],[0,4,5,0,0,0],[0,0,0,7,2,0],[0,0,0,1,8,0],[0,0,0,0,1,8]]` |
| **Per‑class F1‑scores** (macro‑avg ≈ 0.76) | – |

**Top‑10 most important features (Mean Decrease Impurity)**  

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `vel_3_std` | 0.0441 |
| 2 | `vel_mag_mean` | 0.0425 |
| 3 | `coord_0_fft_power_sum` | 0.0386 |
| 4 | `vel_0_std` | 0.0384 |
| 5 | `vel_4_std` | 0.0369 |
| 6 | `coord0_10pct` | 0.0321 |
| 7 | `vel_0_max` | 0.0293 |
| 8 | `vel_0_min` | 0.0292 |
| 9 | `corr_handX_left_right` | 0.0276 |
|10 | `vel_4_max` | 0.0268 |

These features are mainly **velocity statistics** (standard deviations, means) and a few **frequency‑domain** descriptors (`coord_0_fft_power_sum`, `coord0_10pct`).

---

### 2. Statistical Relationships

* **Highly correlated pairs (|ρ| > 0.9)**: 67 pairs detected.  
  *Examples*: `vel_0_std` ↔ `vel_1_std` (ρ ≈ 0.96), `vel_mag_mean` ↔ `vel_0_std` (ρ ≈ 0.93), `vel_mag_std` ↔ `vel_mag_mean` (ρ ≈ 0.93).  
  *Implication*: many velocity‑derived metrics capture overlapping information.

* **Perfectly correlated identifiers**: `double_id` ↔ `test_feature` (ρ = 1.0).  
  *These are pure IDs, not predictive.*

* **Constant columns**: `const_one`, `const_one_float`, `group_size` (no variance).

---

### 3. Low‑Importance Features (Importance < 0.005)

- `double_id`, `test_feature` (identifiers)  
- `const_one`, `const_one_float`, `group_size` (constants)  
- `band_power_ratio_vel_0`  
- `vel_2_mean`, `vel_0_mean`, `vel_1_mean`  
- `dist_elbowleft_wristleft_mean`, `dist_elbowleft_wristleft_max`, `dist_handleft_wristleft_min`  
- `angle_wrist_left_mean`  
- `coord0_50pct`  
- `acc_0_skew`  
- `vel_mag_min`  

These contributed negligibly to predictive performance.

---

### 4. Feature Pruning

Using **attribute_pruning_tool**, the 16 low‑importance attributes listed above were removed.

**Post‑pruning evaluation** (same RandomForest settings):

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.759** (unchanged) |
| **Top‑10 features** | Identical to baseline |

*Result*: Pruning did **not** degrade model performance, confirming the removed attributes were non‑informative.

---

### 5. Conclusions & Recommendations for the Team

1. **Predictive Power** – The current engineered features achieve moderate classification performance (≈ 76 % accuracy). The most discriminative signals are velocity variability and certain frequency‑domain measures.

2. **Redundancy** – A large subset of velocity‑derived statistics are highly correlated. Future work could consolidate these (e.g., keep a representative std/mean per axis) to reduce dimensionality without loss.

3. **Non‑Informative Attributes** – Identifier and constant columns, plus several low‑importance derived metrics, can be safely excluded from the feature set, simplifying models and reducing computational load.

4. **Robustness** – Since pruning did not affect accuracy, the model appears robust to removal of noisy or irrelevant features.

5. **Next Steps for Scientist/Extractor** –  
   * Focus on generating additional **interaction** or **non‑linear** descriptors that capture relationships between hand, wrist, and thumb motions (e.g., joint angles, cross‑correlations).  
   * Consider dimensionality reduction (PCA) on the highly correlated velocity block to produce compact latent features.  

These findings should guide the next round of hypothesis generation and attribute extraction.