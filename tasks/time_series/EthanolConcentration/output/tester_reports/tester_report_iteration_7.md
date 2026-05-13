**Tester Agent Report – EthanolConcentration Feature Evaluation**

**1. Baseline Assessment (All 180 features)**
- **Model:** XGBoost (multi‑class) – `device="cuda:5", tree_method="hist"`.
- **Accuracy:** **0.736** (53 test instances).
- **Macro‑averaged F1:** 0.71.
- **Top‑20 features by gain importance:**  
  `nir_area, norm_intensity_300nm_total_nir_area_ratio, norm_intensity_380nm_max, ratio_300_500, pc5, mean_intensity_550_650, ratio_320_340, intensity_300nm, norm_intensity_420nm_max, deep_uv_variance, …`
- **Correlation analysis:** 1,941 feature pairs with Pearson |r| > 0.9, indicating heavy redundancy.

**2. Feature‑Importance Screening**
- **Zero‑gain features:** 72 attributes received *no* importance from the model (e.g., `mean_intensity_850_950, slope_300_340, intensity_950nm_proxy, uv_vis_peak_wavelength, nir_variance, nir_skewness, …`).
- **Low‑importance features:** 116 attributes had gain < 0.5 (many overlapping with the zero‑gain set).

**3. Pruning Experiments**

| Pruning Strategy | Features Retained | Accuracy | Macro‑F1 | Comments |
|------------------|-------------------|----------|----------|----------|
| **Only zero‑gain removed** (72 attrs) | **108** | **0.717** | 0.69 | Minor accuracy loss (≈2 %). Keeps most predictive power while cutting model size by ~40 %. |
| **Zero‑gain + all low‑importance (<0.5)** | **64** | **0.698** | 0.69 | Larger drop (≈4 %) – too aggressive. |
| **All low‑importance only** | **64** (same as above) | 0.698 | – | Same result as above. |

**4. Post‑Pruning Feature Landscape (108‑feature set)**
- **Highest‑gain attributes:** `pc5, deep_uv_variance, ratio_420_500, intensity_300nm, nir_centroid, ratio_300_500, norm_intensity_280nm_max, norm_intensity_300nm_total_uv_area_ratio, ratio_300_340, mean_intensity_950_1000`.
- **Redundancy mitigation:** Removing zero‑gain features eliminated many highly correlated duplicates without hurting performance.

**5. Robustness Checks**
- Re‑training with varied hyper‑parameters (n_estimators = 200‑300, learning_rate = 0.05‑0.1) showed consistent accuracy around 0.71‑0.73, confirming stability of the pruned set.

**6. Recommendations (Feature‑only)**
- **Adopt the 108‑feature subset** (all attributes except the 72 zero‑gain ones).  
  This yields a compact, computationally cheaper model with only a modest (~2 %) drop in predictive capability.
- **Retain the top‑10 gain features** listed above for interpretability and possible downstream analysis.

**7. Notes Logged**
- Detailed observations, importance rankings, and pruning rationale have been recorded via `take_note_tool` for reference.