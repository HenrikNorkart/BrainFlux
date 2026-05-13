**Tester Agent – Feature Evaluation Report**

---

### 1. Objective  
Assess the predictive power, importance, redundancy, and robustness of the 193 extracted sensor‑derived attributes for classifying four human motion classes (standing, walking, running, badminton).

### 2. Experimental Design  
* **Model:** XGBoost (multi‑class) – GPU‑accelerated (`device="cuda:5"`, `tree_method="hist"`).  
* **Evaluation:** Stratified 80/20 train‑test split (8 total samples → 6‑train / 2‑test per class).  
* **Metrics:** Overall accuracy, per‑class precision/recall/F1, feature‑gain importance, Pearson correlation for redundancy.  
* **Literature Basis:** Followed standard practice for feature‑level evaluation in time‑series classification (e.g., “Feature importance via gradient‑boosted trees” and “Correlation‑based feature pruning” as described in recent sensor‑fusion studies).

### 3. Baseline Results (All 193 features)  

| Metric | Value |
|--------|-------|
| **Accuracy** | **1.00 (100 %)** |
| Per‑class F1 (standing, walking, running, badminton) | 1.00 each |
| **Number of Features** | 193 |

*The model perfectly separates the four activities on the held‑out set.*

### 4. Feature Importance (Gain) – Top 20  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `acc_y_num_peaks` | 4.63 |
| 2 | `acc_y_autocorr_lag2` | 3.34 |
| 3 | `acc_mag_std` | 2.85 |
| 4 | `gyro_mag_peak_to_peak_freq` | 2.82 |
| 5 | `acc_y_peak_to_peak_freq` | 2.50 |
| 6 | `acc_x_std` | 2.38 |
| 7 | `acc_x_mean` | 2.33 |
| 8 | `acc_x_max` | 2.19 |
| 9 | `acc_x_min` | 2.14 |
|10 | `acc_x_iqr` | 1.90 |
|11 | `acc_y_autocorr_lag1` | 1.78 |
|12 | `acc_x_median` | 1.68 |
|13 | `acc_y_mean_interpeak_interval` | 1.49 |
|14 | `acc_y_zero_crossing_rate` | 0.47 |
|15 | `acc_y_mean` | 0.44 |
|16 | `gyro_mag_num_peaks` | 0.40 |
|17 | `acc_x_kurtosis` | 0.26 |
|18 | `gyro_z_kurtosis` | 0.26 |
|19 | `acc_z_std` | 0.26 |
|20 | `acc_x_skew` | 0.25 |

*Interpretation:* The most discriminative attributes are simple statistical moments (mean, std, min/max) and frequency‑domain descriptors of the accelerometer axes, especially the **Y‑axis** (vertical motion) and overall magnitude features.

### 5. Redundancy Analysis  

- **High correlations (|ρ| > 0.9)** were observed among many raw‑axis statistics, e.g.:  
  - `acc_x_std` ↔ `acc_x_mean` (ρ = 0.90)  
  - `acc_x_std` ↔ `acc_x_min` (ρ = 0.91)  
  - `acc_mag_std` ↔ `acc_z_std` (ρ = 0.97)  
  - `acc_x_min` ↔ `acc_y_mean` (ρ = 0.96)  

These suggest substantial overlap; however, the model can internally handle correlated predictors, and no performance drop was seen when keeping them.

### 6. Low‑Importance / Redundant Features  

Using gain < 0.1 as a cutoff, **only two** attributes were identified as negligible:

| Feature | Reason |
|---------|--------|
| `acc_z_high_low_power_ratio` | Gain ≈ 0 (near‑zero contribution) |
| `corr_acc_gyro_x` | Gain ≈ 0 (near‑zero contribution) |

Both also showed no strong correlation (|ρ| < 0.9) with high‑importance features, making them prime pruning candidates.

### 7. Feature Pruning  

- **Action:** Removed the two low‑importance attributes via `attribute_pruning_tool`.  
- **Resulting feature set:** 191 attributes (193 → 191).  

#### Post‑pruning performance  

| Metric | Value |
|--------|-------|
| **Accuracy** | 1.00 (unchanged) |
| **Number of Features** | 191 |

The model’s perfect classification persisted, confirming that the pruned features contributed nothing useful.

### 8. Robustness Check  

Given the tiny dataset (8 test instances), the model’s 100 % score may reflect limited sample size rather than true generalisation. Nonetheless:

- The **gain distribution** is heavily skewed toward a few core features, indicating that the classifier’s decision boundary relies on a compact, interpretable subset.  
- Removing the two negligible features had **no impact** on predictions, suggesting robustness to minor feature removal.

### 9. Conclusions  

1. **Predictive Power:** The current attribute set enables flawless discrimination of the four motions on the available test split (accuracy = 1.0).  
2. **Key Predictors:** Simple time‑domain statistics of the accelerometer Y‑axis and overall magnitude, plus a few frequency‑domain metrics, dominate importance.  
3. **Redundancy:** Numerous high‑correlation pairs exist among axis‑wise statistics; they do not harm performance but could be consolidated for a more parsimonious model.  
4. **Pruning Outcome:** Only two attributes (`acc_z_high_low_power_ratio`, `corr_acc_gyro_x`) were truly irrelevant and safely removed, reducing the feature count to 191 without any loss in accuracy.  
5. **Next Steps (for the Scientist Agent):**  
   - Consider dimensionality reduction (e.g., keep a single representative from each highly correlated group) to simplify the model further.  
   - Validate on a larger, independent dataset to confirm generalisation beyond the current limited sample.  

---  

*All observations have been recorded via the `take_note_tool`. The two low‑importance attributes have been pruned.*