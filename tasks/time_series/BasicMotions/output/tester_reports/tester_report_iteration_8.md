**Comprehensive Feature Evaluation Report**

**1. Predictive Power (Whole Feature Set)**
- 5‑fold stratified CV using XGBoost (multi:softprob, `device="cuda:5"`, `tree_method="hist"`):  
  **Mean Accuracy = 0.975 ± 0.05** (four activity classes).  
  The full 260‑feature set already provides very high classification performance.

**2. Feature Importance (Gain)**
- XGBoost gain importance identified **22 attributes** with non‑zero contribution.  
- The top 20 (by gain) were:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `acc_y_num_peaks` | 12.05 |
| 2 | `acc_y_autocorr_lag2` | 9.40 |
| 3 | `acc_x_max` | 3.20 |
| 4 | `acc_x_mean` | 3.19 |
| 5 | `acc_x_std` | 2.93 |
| 6 | `acc_y_autocorr_lag1` | 2.90 |
| 7 | `gyro_mag_peak_to_peak_freq` | 2.18 |
| 8 | `acc_y_zero_crossing_rate` | 2.03 |
| 9 | `acc_x_min` | 1.96 |
|10 | `acc_y_mean_interpeak_interval` | 1.84 |
|…|…|…|

**3. Redundancy & Correlation Analysis**
- Pairwise Pearson correlations (|r| > 0.9) revealed strong redundancy among many of the important features, e.g.:
  - `acc_x_max` ↔ `acc_mag_std` (0.95)  
  - `acc_x_std` ↔ `gyro_mag_median` (0.96)  
  - `gyro_x_max` ↔ `gyro_x_std` (0.92)  
  - `gyro_x_max` ↔ `yaw_std` (0.94)  
  - etc.

**4. Pruned, Non‑Redundant Feature Subset**
- Applying a correlation‑threshold of 0.9 and retaining the highest‑gain member of each cluster yielded a compact set of **16 features**:

```
['acc_y_num_peaks',
 'acc_y_autocorr_lag2',
 'acc_x_max',
 'acc_x_mean',
 'acc_y_autocorr_lag1',
 'gyro_mag_peak_to_peak_freq',
 'acc_y_zero_crossing_rate',
 'acc_x_min',
 'acc_y_mean_interpeak_interval',
 'y_autocorr_lag2_over_lag1',
 'acc_mag_peak_to_peak_freq',
 'gyro_z_peak_to_peak_freq',
 'acc_max_diff_xy',
 'gyro_x_max',
 'acc_y_dom_freq',
 'max_corr_x']
```

- These retain the most informative signal while removing duplicated information.

**5. Predictive Power (Reduced Set)**
- Re‑training XGBoost on only the 16 selected attributes reproduced the original performance:  
  **Mean Accuracy = 0.975 ± 0.05** – no loss of predictive ability.

**6. Robustness Check**
- Adding Gaussian noise (σ = 1 % of each feature’s standard deviation) to the reduced set did **not** affect accuracy (still 0.975).  
- Indicates the selected features are stable to small perturbations.

**7. Feature Pruning Outcome**
- **Zero‑gain attributes** (238 of 260) were removed via `attribute_pruning_tool`.  
- **Highly correlated, lower‑gain attributes** (6 of the original 22) were also omitted, leaving the 16‑feature core.

**8. Final Feature Set Summary**
| Feature | Brief Description |
|---------|-------------------|
| `acc_y_num_peaks` | Count of peaks in Y‑axis acceleration |
| `acc_y_autocorr_lag2` | Autocorrelation of Y‑acc at lag 2 |
| `acc_x_max` | Maximum X‑axis acceleration |
| `acc_x_mean` | Mean X‑axis acceleration |
| `acc_y_autocorr_lag1` | Autocorrelation of Y‑acc at lag 1 |
| `gyro_mag_peak_to_peak_freq` | Peak‑to‑peak frequency of gyroscope magnitude |
| `acc_y_zero_crossing_rate` | Zero‑crossing rate of Y‑acc |
| `acc_x_min` | Minimum X‑axis acceleration |
| `acc_y_mean_interpeak_interval` | Mean interval between Y‑acc peaks |
| `y_autocorr_lag2_over_lag1` | Ratio of Y‑axis autocorr lag 2 / lag 1 |
| `acc_mag_peak_to_peak_freq` | Peak‑to‑peak frequency of acceleration magnitude |
| `gyro_z_peak_to_peak_freq` | Peak‑to‑peak frequency of Z‑axis gyroscope |
| `acc_max_diff_xy` | Max difference between X and Y acceleration |
| `gyro_x_max` | Maximum X‑axis gyroscope reading |
| `acc_y_dom_freq` | Dominant frequency component of Y‑acc |
| `max_corr_x` | Maximum cross‑correlation of X‑axis signals |

**9. Conclusions**
- The original feature collection is highly predictive but contains extensive redundancy.  
- A **compact 16‑feature subset** captures the full discriminatory power, simplifies the model, and is robust to minor noise.  
- Pruning the 238 zero‑gain attributes and the 6 redundant low‑gain attributes dramatically reduces dimensionality while preserving classification performance.  

*All non‑contributory attributes have been successfully pruned from the global attribute dictionary.*