**Feature‑Testing Report – NATOPS Hand‑Sensor Classification**

---

### 1.  Data Overview
| Item | Value |
|------|-------|
| Instances (rows) | **180** |
| Attributes (features) | **73** (plus the target column) |
| Classes | 6 (labels 1‑6) |
| Target distribution | Balanced – 30 samples per class |

The attributes are derived from 3‑D positions of hands, elbows, wrists and thumbs (velocity, acceleration, distances, angles, spectral metrics, etc.).

---

### 2.  Predictive Power (Random‑Forest Baseline)

| Metric | Value |
|--------|-------|
| **Overall accuracy** (30 % hold‑out) | **0.722** |
| Macro‑averaged F1‑score | 0.724 |
| Weighted‑averaged F1‑score | 0.724 |

**Confusion matrix (rows = true, columns = predicted)**  

|   | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1** | 8 | 1 | 0 | 0 | 0 | 0 |
| **2** | 1 | 5 | 3 | 0 | 0 | 0 |
| **3** | 0 | 5 | 4 | 0 | 0 | 0 |
| **4** | 0 | 0 | 0 | 6 | 3 | 0 |
| **5** | 0 | 0 | 0 | 1 | 8 | 0 |
| **6** | 0 | 0 | 0 | 0 | 1 | 8 |

*The model discriminates classes 1, 4, 5 and 6 well; classes 2 and 3 are more frequently confused.*

---

### 3.  Feature Importance (Mean Decrease in Impurity)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **vel_3_std** | 0.0506 |
| 2 | **vel_mag_mean** | 0.0449 |
| 3 | **coord0_10pct** | 0.0351 |
| 4 | **vel_0_std** | 0.0328 |
| 5 | **corr_handX_left_right** | 0.0327 |
| 6 | **vel_0_min** | 0.0317 |
| 7 | **vel_4_std** | 0.0301 |
| 8 | **vel_0_max** | 0.0300 |
| 9 | **vel_4_min** | 0.0298 |
|10 | **coord_0_fft_power_sum** | 0.0294 |

*These ten attributes together explain > 30 % of the total impurity reduction and are the strongest predictors of the six command gestures.*

---

### 4.  Low‑Impact Features (Importance < 0.005)

The following 19 attributes contributed negligible information to the Random‑Forest model:

- `dist_handleft_wristleft_std`
- `coord0_50pct`
- `band_power_ratio_vel_0`
- `acc_0_25pct`
- `dist_handleft_wristleft_acc_mean`
- `acc_0_min`
- `dist_elbowleft_wristleft_std`
- `vel_6_mean`
- `vel_1_mean`
- `dist_handleft_wristleft_mean`
- `dist_elbowleft_wristleft_mean`
- `vel_2_mean`
- `dist_elbowright_wristright_mean`
- `acc_0_skew`
- `vel_0_mean`
- `jerk_0_mean`
- `dist_handleft_wristleft_min`
- `dist_elbowleft_wristleft_max`
- `vel_mag_min`

These features can be **pruned** without harming (and potentially improving) model efficiency and interpretability.

*Pruning was performed via the `attribute_pruning_tool`; the attribute‑explanations dictionary was updated accordingly.*

---

### 5.  Robustness Check (Post‑Pruning)

A second Random‑Forest run **after pruning** the low‑impact attributes yielded the **same accuracy (0.722)** and an identical ranking of the top‑10 important features. This confirms that the removed attributes do not carry useful predictive signal.

---

### 6.  Key Take‑aways

1. **Predictive strength** – The current feature set yields respectable classification performance (≈ 72 % accuracy) on a balanced six‑class problem.  
2. **High‑impact attributes** are dominated by velocity‑related statistics (`vel_*_std`, `vel_*_min/max`) and a few derived metrics (FFT power, correlation between left/right hand X‑coordinates, and a percentile of the first coordinate).  
3. **Redundant / noisy attributes** – 19 features with near‑zero importance can be safely removed, reducing dimensionality and computational load.  
4. **Model stability** – Removing those low‑impact features does **not** degrade performance, indicating the remaining feature pool is robust.  

---

### 7.  Suggested Next Steps (for the Scientist & Extractor)

- **Focus extraction** on velocity‑based descriptors and inter‑hand correlation measures, as they consistently rank highest.  
- **Discard** the listed low‑importance attributes in future extraction pipelines.  
- **Consider** more sophisticated temporal models (e.g., LSTM, Temporal Convolution) that can exploit the underlying time‑series nature, now that the feature space is trimmed to the most informative signals.  

*All observations have been recorded via `take_note_tool` for the team’s reference.*