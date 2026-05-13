**Tester Agent – Feature Evaluation Report**  
*Dataset: EthanolConcentration (spectral time‑series, classification of whisky ethanol %)*  

---

### 1. Experimental Setup
- **Model:** XGBoost (multi‑class) – `objective='multi:softprob'`, `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `device='cuda:5'`, `tree_method='hist'`.  
- **Train/Test split:** 80 % / 20 % stratified, random_state = 42.  
- **Target encoding:** `LabelEncoder` (e35, e38, e40, e45 → 0‑3).  
- **Evaluation metrics:** Overall accuracy, per‑class precision/recall/F1, confusion matrix, XGBoost gain‑based feature importance.  

---

### 2. Baseline Performance (All 38 features)  
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.509** |
| Macro‑avg F1 | 0.502 |
| Weighted‑avg F1 | 0.505 |

**Confusion matrix (rows = true, cols = predicted)**  

|   | e35 | e38 | e40 | e45 |
|---|-----|-----|-----|-----|
| **e35** | 4 | 4 | 4 | 1 |
| **e38** | 4 | 5 | 2 | 2 |
| **e40** | 1 | 1 |10 | 2 |
| **e45** | 4 | 1 | 0 | 8 |

*Interpretation:* The model discriminates the highest concentration (e45) reasonably well, but struggles with the lower classes (e35/e38) where mis‑classifications are common.

---

### 3. Feature Importance (Gain) – Top 15  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `ratio_300_500` | 1.586 |
| 2 | `intensity_420nm` | 1.284 |
| 3 | `intensity_340nm` | 1.193 |
| 4 | `intensity_300nm` | 1.010 |
| 5 | `intensity_900nm` | 0.839 |
| 6 | `ratio_900_1000` | 0.829 |
| 7 | `intensity_450nm` | 0.817 |
| 8 | `derivative_max` | 0.757 |
| 9 | `ratio_250_300` | 0.742 |
|10 | `intensity_320nm` | 0.738 |
|11 | `mean_intensity_400_500` | 0.691 |
|12 | `intensity_280nm` | 0.661 |
|13 | `intensity_500nm` | 0.646 |
|14 | `mean_intensity_250_300` | 0.631 |
|15 | `intensity_380nm` | 0.624 |

**Key observations**
- **Ratio features** (`ratio_300_500`, `ratio_250_300`, `ratio_900_1000`) are the strongest predictors, capturing relative spectral changes linked to ethanol concentration.
- **Specific wavelength intensities** around 300‑500 nm and 900‑1000 nm also carry substantial signal.
- **Derivative‑based feature** (`derivative_max`) contributes meaningfully, indicating the shape of the spectrum matters.

---

### 4. Redundancy & Correlation Analysis
- Pairwise absolute Pearson correlations > 0.9 were abundant among raw intensity windows (e.g., `intensity_300nm` ↔ `mean_intensity_250_300`, `intensity_350nm`, `intensity_320nm`, etc.).
- 22 highly correlated or low‑importance attributes were identified and **pruned** (e.g., `intensity_350nm`, `intensity_260nm`, `overall_mean_intensity`, many moving‑window means, secondary derivative stats, and several ratios not in the top‑15).

---

### 5. Post‑Pruning Performance
- **Model re‑trained on the reduced set (16 retained features).**  
- **Accuracy:** **0.509** (unchanged).  
- Feature‑importance ranking remained identical; the pruned attributes contributed negligible predictive power.

*Conclusion:* Pruning eliminated redundancy without harming performance, yielding a more compact, interpretable feature set.

---

### 6. Robustness Checks (quick sanity)
- Adding Gaussian noise (σ = 0.01 of feature std) to the test set reduced accuracy by < 2 pp, indicating modest robustness to minor perturbations.  
- Re‑training with a different random seed produced accuracy variations within ±0.03, confirming stability of the observed performance.

---

### 7. Summary & Recommendations for the Team
1. **Predictive Power:** Current feature set achieves ~51 % accuracy—better than random (25 %) but still limited for reliable screening.  
2. **Core Predictors:** Focus on the 15‑feature subset highlighted above; ratio features and a handful of key intensities are the primary drivers.  
3. **Redundant Features:** The 22 pruned attributes can be safely removed, simplifying downstream modeling and interpretation.  
4. **Further Work:**  
   - Investigate alternative classification algorithms (e.g., SVM with RBF, LightGBM) to see if they exploit the same signal more effectively.  
   - Explore ensemble of the top ratio and intensity features (e.g., logistic regression on the reduced set) to gauge whether a simpler model can close the performance gap.  
   - Conduct cross‑validation (5‑fold) to obtain a more reliable estimate of generalisation performance.

*Prepared by the Tester Agent.*