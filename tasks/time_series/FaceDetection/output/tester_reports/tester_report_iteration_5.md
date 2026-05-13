**Comprehensive Evaluation Report – Feature Set for FaceDetection Classification**

---

### 1.  Experimental Setup
- **Task:** Binary classification (Face = 1, Scrambled = 0) on the provided MEG‑derived feature matrix (`df_attributes`, 5 890 rows × 1 832 features + target).  
- **Model:** XGBoost Classifier (binary:logistic) with GPU (`device="cuda:5"`, `tree_method="hist"`), 200 trees, max depth 6, learning rate 0.1, subsample 0.8, colsample_bytree 0.8.  
- **Validation:** Stratified train‑test split (80 % / 20 % hold‑out).  

---

### 2.  Baseline Predictive Performance
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.650** |
| **ROC‑AUC** | **0.695** |

These figures represent the predictive power of the **full, unpruned** feature set.

---

### 3.  Feature Importance (Gain) – Top 20
| Rank | Feature | Gain |
|------|---------|------|
| 1 | `ch14_ptp` | 18.03 |
| 2 | `ch116_var` | 16.21 |
| 3 | `ch83_early_mean` | 15.57 |
| 4 | `ch124_m170_peak_amp` | 15.05 |
| 5 | `ch92_latency` | 14.10 |
| 6 | `ch104_mean` | 13.14 |
| 7 | `ch69_spec_entropy` | 12.64 |
| 8 | `ch80_mean` | 11.95 |
| 9 | `ch56_var` | 11.86 |
|10 | `ch79_spec_entropy` | 11.83 |
|11 | `mean_spec_entropy_all_channels` | 11.19 |
|12 | `ch88_max` | 10.88 |
|13 | `ch72_ptp` | 9.74 |
|14 | `ch72_mean` | 9.74 |
|15 | `ch8_late_mean` | 9.74 |
|16 | `ch122_latency` | 9.74 |
|17 | `ch17_spec_entropy` | 9.74 |
|18 | `ch53_ptp` | 9.74 |
|19 | `ch35_late_mean` | 9.74 |
|20 | `ch97_max` | 9.74 |

*Gain values are the XGBoost “gain” importance metric (higher = more influence).*

---

### 4.  Redundancy & Inter‑Feature Relationships
- **High‑correlation pairs (|ρ| > 0.8) among the top 20:**  
  - `ch104_mean` ↔ `ch80_mean` (ρ ≈ 0.85)  
  - `ch8_late_mean` ↔ `ch35_late_mean` (ρ ≈ 0.84)  

These pairs suggest potential redundancy; retaining only one of each could simplify the model with minimal loss.

---

### 5.  Feature Pruning Experiments
| Pruning Strategy | Features Removed | Remaining Features | Accuracy | ROC‑AUC |
|------------------|------------------|--------------------|----------|----------|
| **Zero‑gain only** (37 features) | Features with gain = 0 (e.g., `delta_power_mean`, `ch0_delta_power`, …) | 1 795 | 0.650 | 0.695 |
| **Low‑gain (< 0.1)** (38 features) | Same as above plus a few low‑gain items | 1 794 | **0.620** | **0.689** |

*Removing only zero‑gain attributes preserves baseline performance. Adding the low‑gain set slightly degrades accuracy, indicating those features still contribute modestly.*

---

### 6.  Robustness Check
- **Procedure:** Added Gaussian noise (σ = 10 % of each top‑20 feature’s standard deviation) to the test set.  
- **Result:**  
  - Baseline → Accuracy 0.650, AUC 0.695  
  - Noisy → Accuracy 0.646, AUC 0.692  

The negligible drop demonstrates that the model’s predictions are **robust** to modest perturbations in the most important features.

---

### 7.  Key Take‑aways
1. **Predictive Power:** The current feature set yields moderate performance (≈ 65 % accuracy, 0.70 AUC).  
2. **Important Features:** A small subset (≈ 20) of channel‑level statistics (peak‑to‑peak, variance, spectral entropy, latency, M170 peak amplitude) drives most of the predictive signal.  
3. **Redundancy:** Some top features are highly correlated; collapsing these could reduce dimensionality without harming performance.  
4. **Pruning:** Eliminating **zero‑gain** attributes is safe and reduces the feature count by ~2 %. More aggressive pruning (gain < 0.1) harms accuracy, so retain those low‑gain features for now.  
5. **Robustness:** Model predictions are stable under realistic noise, indicating reliable feature extraction.  

---

### 8.  Recommendations for the Next Iteration (for the Scientist & Extractor Agents)
- **Focus extraction** on the top‑20 (or top‑30) high‑gain attributes, especially those related to **peak‑to‑peak amplitude, spectral entropy, and M170 latency/peak amplitude**.  
- **Consider merging** highly correlated pairs (e.g., keep `ch104_mean` *or* `ch80_mean`).  
- **Retain low‑gain features** (gain ≥ 0.01) as they still contribute modestly; future experiments may reveal synergistic effects.  
- **Maintain robustness** checks when new attributes are added.

*All observations above have been recorded via `take_note_tool` for reference.*