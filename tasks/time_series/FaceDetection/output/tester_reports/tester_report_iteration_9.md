**Tester Agent – Feature Evaluation Report (FaceDetection MEG Classification)**  

---

### 1. Experimental Set‑up  
| Step | Description |
|------|-------------|
| **Data** | `df_attributes` – 5 890 trials, 1 866 columns (185 + target). |
| **Target** | `target` (0 = scrambled, 1 = face). |
| **Baseline model** | XGBoost (`objective='binary:logistic'`, `tree_method='hist'`, `device='cuda:5'`). |
| **Evaluation** | Train/validation split (80 % / 20 %) stratified on `target`. Metrics: **Accuracy** & **ROC‑AUC**. |
| **Feature‑selection** | Gain‑based importance from a full‑feature XGBoost model; top 200 features retained. |
| **Robustness test** | Gaussian noise (σ = 0.1 × feature‑wise std) added to the training data. |

---

### 2. Baseline Performance (All 1 865 features)  

| Metric | Value |
|--------|-------|
| Accuracy | **0.621** |
| ROC‑AUC  | **0.671** |
| Top‑20 important features (gain) | `ch76_var`, `corr_ch28_ch29`, `ch111_late_mean`, `ch69_kurt`, `ch85_early_mean`, … |

*Interpretation*: The raw feature set yields modest predictive power (≈ 62 % accuracy, AUC ≈ 0.67). Several channel‑level statistics (variance, kurtosis, mean) dominate importance.

---

### 3. Feature‑Reduction & Impact  

**Procedure** – Extracted the 200 highest‑gain features from the full model, then re‑trained.  

| Metric | All features | Top‑200 features |
|--------|--------------|------------------|
| Accuracy | 0.621 | **0.620** |
| ROC‑AUC  | 0.671 | **0.684** |
| Number of features used | 1 865 | **200** |

*Result*: Keeping only the top 200 attributes **improves AUC** (0.684 vs 0.671) while maintaining accuracy and dramatically reducing dimensionality (≈ 90 % fewer features).  

**Top‑20 features after reduction** (gain):  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `ch124_m170_peak_amp` | 11.18 |
| 2 | `ch96_late_mean` | 7.07 |
| 3 | `ch122_latency` | 6.80 |
| 4 | `ch124_mean_diff` | 6.74 |
| 5 | `ch22_m170_peak_amp` | 6.54 |
| 6 | `overall_mean` | 6.08 |
| 7 | `ch128_latency` | 5.97 |
| 8 | `ch100_m170_peak_amp` | 5.74 |
| 9 | `ch132_latency` | 5.42 |
|10 | `ch74_spec_entropy` | 5.19 |
| … | … | … |

These features are largely **peak‑amplitude, latency, and entropy** measures from specific channels, confirming that temporally‑localized spectral characteristics are most discriminative.

---

### 4. Statistical Relationships Among Retained Features  

* **Average absolute pairwise correlation** (top 200): **0.112** – low overall redundancy.  
* **Highly correlated pairs (|ρ| > 0.8)**: **30** – a small subset of features are strongly linked (mostly latency‑/peak‑related measures from neighboring channels).  

*Conclusion*: The selected set is largely **non‑redundant**, supporting the pruning decision.

---

### 5. Robustness to Noise  

| Condition | Accuracy | ROC‑AUC |
|-----------|----------|---------|
| Clean training data | **0.640** | **0.684** |
| +10 % Gaussian noise (σ = 0.1 × std) | 0.635 | 0.670 |

*Interpretation*: Adding modest Gaussian perturbations to the training features reduces AUC by ~0.015 and accuracy by ~0.005 – the model is **reasonably robust** to moderate noise levels.

---

### 6. Key Take‑aways  

| Aspect | Observation |
|--------|-------------|
| **Predictive power** | Top‑200 features achieve **AUC ≈ 0.68**, a modest but measurable improvement over the full set. |
| **Feature importance** | Gain is concentrated in a handful of **peak‑amplitude, latency, entropy** attributes from a few channels. |
| **Redundancy** | Low average correlation; only ~30 highly correlated pairs, indicating efficient coverage of the signal space. |
| **Impact of pruning** | Removing ~1 665 low‑importance attributes **does not degrade** performance; it simplifies the model and speeds training. |
| **Robustness** | Model tolerates ~10 % added Gaussian noise with < 2 % drop in AUC, suggesting stability for slightly noisy MEG recordings. |

---

### 7. Next Steps for the Team  

* **Scientist Agent** – Investigate the physiological relevance of the top‑ranked channels (e.g., visual‑cortex locations) and whether the identified latency/peak patterns align with known face‑processing dynamics.  
* **Extractor Agent** – Focus future extraction on the high‑importance families (peak‑amplitude, latency, spectral entropy) and consider summarising neighbouring channels to further reduce dimensionality.  

*All findings are documented in the internal notes and the feature set has been pruned to the top 200 attributes for downstream modeling.*