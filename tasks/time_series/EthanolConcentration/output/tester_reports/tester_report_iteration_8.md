**Comprehensive Feature Evaluation Report – EthanolConcentration Classification**

---

### 1.  Experimental Setup
| Step | Details |
|------|---------|
| **Model** | XGBoost (multiclass) – `objective='multi:softprob'`, `eval_metric='mlogloss'`, `device='cuda:5'`, `tree_method='hist'` |
| **Data Split** | 80 % train / 20 % test, stratified by target, `random_state=42` |
| **Target Encoding** | `LabelEncoder` (e35, e38, e40, e45 → 0‑3) |
| **Metrics** | Accuracy, Macro‑averaged F1‑score |
| **Baseline Feature Set** | All 185 non‑target attributes (raw spectra, derived ratios, derivatives, PCA components, etc.) |
| **Reduced Feature Set** | 9 selected, low‑redundancy attributes (see Section 3) |

---

### 2.  Baseline Performance (All Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.679** |
| **Macro‑F1** | **0.656** |

The baseline model already captures a reasonable amount of class‑separating information, but many features appear redundant.

---

### 3.  Feature Importance & Redundancy

**Top‑20 gain importance (XGBoost)**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `intensity_300nm` | 2.375 |
| 2 | `intensity_380nm` | 2.350 |
| 3 | `ratio_320_340` | 1.918 |
| 4 | `pc5` | 1.746 |
| 5 | `deep_uv_variance` | 1.650 |
| 6 | `diff_420_450` | 1.584 |
| 7 | `ratio_420_500` | 1.496 |
| 8 | `intensity_340nm` | 1.475 |
| 9 | `nir_centroid` | 1.447 |
| … | … | … |

**Correlation Findings**  
- Raw intensity columns (`intensity_300nm`, `intensity_380nm`, `intensity_340nm`, `intensity_350nm`, `mean_intensity_250_300`) are **> 0.99** correlated.  
- Ratios `ratio_320_340` ↔ `ratio_300_340` (r = 0.995).  
- `ratio_420_500` ↔ `ratio_380_500` (r = 0.992).  
- Several derivative/mean intensity pairs also exceed 0.96 correlation.

These high‑correlation groups provide little added information beyond a single representative feature.

---

### 4.  Reduced Feature Set Experiment

**Selected attributes (9 total)**  

| Feature | Rationale |
|---------|-----------|
| `intensity_300nm` | Representative raw intensity (captures the whole correlated group). |
| `pc5` | Principal component summarising spectral variance. |
| `deep_uv_variance` | Captures UV‑region variability linked to ethanol content. |
| `ratio_420_500` | Strong discriminative ratio, retained over its highly correlated counterpart. |
| `nir_centroid` | Near‑infrared centroid – sensitive to ethanol concentration. |
| `derivative_mean` | Global shape information of the spectrum. |
| `uv_cosine_similarity_mean` | Global similarity to mean UV spectrum. |
| `second_derivative_mean` | Higher‑order shape descriptor. |
| `ratio_900_1000` | Long‑wave ratio useful for distinguishing higher concentrations. |

**Performance with reduced set**

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.698** |
| **Macro‑F1** | **0.675** |
| **Number of Features** | **9** |

*Result:* **Accuracy improved by ~2 %** and macro‑F1 by ~2 % despite a **≈ 95 % reduction in dimensionality**, confirming that most of the discarded attributes were redundant.

---

### 5.  Robustness Testing

| Noise Level (σ × feature std) | Accuracy | Macro‑F1 |
|------------------------------|----------|----------|
| **0 % (baseline)** | 0.698 | 0.675 |
| **1 %** | **0.717** (slight gain, likely regularisation effect) | 0.699 |
| **5 %** | 0.679 | 0.660 |

*Interpretation:* Small perturbations do not harm performance and may even help (noise‑induced regularisation). Larger noise degrades accuracy modestly, indicating **moderate robustness** of the reduced feature set.

---

### 6.  Feature Pruning Action

Using the **attribute_pruning_tool**, the following 10 highly redundant attributes were removed:

- `intensity_380nm`
- `intensity_340nm`
- `intensity_350nm`
- `mean_intensity_250_300`
- `ratio_320_340`
- `ratio_300_340`
- `ratio_380_500`
- `diff_420_450`
- `diff_420_440`
- `mean_intensity_650_750`

These removals streamline the dataset without sacrificing predictive power.

---

### 7.  Key Take‑aways

1. **Raw intensity wavelengths are extremely collinear**; a single representative (e.g., `intensity_300nm`) suffices.  
2. **Derived ratios and derivatives** provide complementary information, but many are duplicated; careful selection improves model efficiency.  
3. **A compact 9‑feature set achieves higher accuracy** than the full 185‑feature set, simplifying downstream deployment.  
4. **Model is reasonably robust** to minor measurement noise, a realistic scenario for field spectroscopic screening.  
5. **Pruned attributes** reduce storage/computation overhead and mitigate multicollinearity risks.

---

**Prepared by:** Tester Agent (feature‑evaluation loop)  

*All experiments were run with the prescribed XGBoost GPU configuration (`device="cuda:5"`, `tree_method="hist"`). No additional preprocessing or feature engineering beyond the described selections was performed.*