**Comprehensive Feature Evaluation Report**  
*Predicting out‑lier patients (ineligible for EEG suppression‑ratio monitoring)*  

---

### 1. Data Overview
| Item | Description |
|------|-------------|
| **Feature matrix** | `df_attributes` – 1 082 rows, 31 numeric features (plus patient `id`). |
| **Target** | Binary vector `y` (0 = eligible, 1 = ineligible), shape (1 082,). |
| **Goal** | Assess predictive power, importance, redundancy, and robustness of the provided features. |

---

### 2. Predictive Power (Baseline Model)

- **Model**: XGBoost (`n_estimators=300`, `max_depth=5`, `learning_rate=0.1`, GPU `cuda:3`, `tree_method='hist'`).
- **Train/Test split**: 80 % / 20 % stratified on `y`.
- **Performance**  
  - **AUC‑ROC**: **0.821**  
  - **Accuracy**: **0.829**  

These scores indicate **strong discriminative ability** of the current feature set.

---

### 3. Feature Importance (Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `pulse_peak_count` | 4.98 |
| 2 | `gcs_peep_product` | 3.48 |
| 3 | `max_low_gcs_episode_duration` | 1.78 |
| 4 | `map_fft_sum5` | 1.53 |
| 5 | `count_ventilator_mode_changes` | 1.51 |
| 6 | `peep_mean` | 1.49 |
| 7 | `max_low_gcs_value` | 1.34 |
| 8 | `gcs_pulse_corr` | 1.22 |
| 9 | `slope_FiO2_vent` | 1.21 |
|10 | `fft4_Pulse` | 1.10 |
| … | … | … |

*Features with **negligible gain** (≤ 0.03):*  

- `time_to_first_overlap_peep` (gain = 0)  
- `overlap_low_gcs_high_peep_mean_episode_length` (gain ≈ 0.001)  
- `overlap_low_gcs_peep_max_episode_length` (gain ≈ 0.025)  

These contributed virtually nothing to model decisions.

---

### 4. Inter‑Feature Correlations & Redundancy

- **Highly correlated pairs (|ρ| > 0.9)**  

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| `overlap_low_gcs_fio2_time_to_first` | `time_to_first_overlap_fio2` | **≈ 1.0** |
| `overlap_low_gcs_high_peep_mean_episode_length` | `overlap_low_gcs_peep_max_episode_length` | **0.979** |

Both pairs involve features already identified as low‑importance, suggesting they are **redundant** and safe to drop.

---

### 5. Feature Pruning

**Removed attributes** (low importance &/or redundant):

1. `time_to_first_overlap_peep`  
2. `overlap_low_gcs_high_peep_mean_episode_length`  
3. `overlap_low_gcs_peep_max_episode_length`  
4. `overlap_low_gcs_fio2_time_to_first`

*Resulting feature count*: **27** (down from 31).

---

### 6. Post‑Pruning Model Performance

- **AUC‑ROC**: **0.820** (virtually unchanged)  
- **Accuracy**: **0.816** (slight drop, still high)  

Top‑ranked features after pruning remain the same high‑impact set, confirming that the removed attributes were **non‑contributory**.

---

### 7. Robustness Assessment

**Procedure**: Added Gaussian noise (σ = 0.1 × feature standard deviation) to every retained feature, re‑trained the same XGBoost model.

- **AUC‑ROC with noisy data**: **0.797**  

The modest decline (~0.023) demonstrates that the predictive signal is **fairly robust** to moderate perturbations.

---

### 8. Summary of Findings

| Aspect | Observation |
|--------|-------------|
| **Predictive Power** | Baseline AUC ≈ 0.82, accuracy ≈ 0.83 – strong capability to separate eligible vs. ineligible patients. |
| **Key Predictors** | `pulse_peak_count`, `gcs_peep_product`, `max_low_gcs_episode_duration`, `map_fft_sum5`, `count_ventilator_mode_changes`, `peep_mean`, etc. |
| **Redundant / Irrelevant Features** | Four attributes (listed above) showed near‑zero gain and/or perfect collinearity; pruning did not harm performance. |
| **Feature Inter‑relationships** | Only two high‑correlation pairs, both involving the pruned variables. |
| **Robustness** | 10 % noise injection reduces AUC to ~0.80 – acceptable stability. |
| **Overall Recommendation** | Retain the **27‑feature subset** (excluding the four pruned attributes). This compact set preserves predictive power, simplifies the model, and reduces risk of over‑fitting. |

---

**Prepared by:** Tester Agent  
*Date:* 2025‑12‑10  

*All notes referenced in this report are available via the internal note‑keeping system.*