**Tester Agent – Feature Evaluation Report**  

**Dataset**: 736 eucalyptus observations, 151 engineered attributes + target (utility rating).  

---

### 1. Experimental Design  
1. **Model** – Multi‑class XGBoost (`objective='multi:softprob'`) with GPU (`device="cuda:5", tree_method="hist"`).  
2. **Train/Test split** – 80 % / 20 % stratified by the target, `random_state=42`.  
3. **Metrics** – Overall accuracy, class‑wise precision/recall/F1, macro‑averaged scores.  
4. **Feature importance** – XGBoost gain values.  
5. **Redundancy check** – Pearson correlation matrix; flagged pairs with |r| > 0.9.  

---

### 2. Baseline Results (All 151 features)  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.696** |
| **Macro‑average F1** | 0.677 |
| **Weighted‑average F1** | 0.694 |
| **Best‑class (none) F1** | 0.971 |
| **Worst‑class (average) F1** | 0.417 |

**Top‑20 gain‑based features** (gain > 0.8):  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | Age_Vig_cu | 6.33 |
| 2 | SurvivalVig_DBHFrosts_cu | 4.71 |
| 3 | Age_Vig_DBHFrosts | 4.64 |
| 4 | sq_Health | 3.57 |
| 5 | Age_Vig_Rainfall_sq | 3.44 |
| 6 | Age_Vig_Rainfall | 3.27 |
| 7 | Age_Vig | 3.14 |
| 8 | Age_Vig_sq | 2.96 |
| 9 | Age_Vig_Volume_sq | 2.21 |
| 10 | Age_Vig_Latitude_sq | 1.55 |
| … | … | … |

---

### 3. Redundancy Analysis  

- **819** feature pairs exhibited |r| > 0.9.  
- Example: `DBH_Ht_product` correlated > 0.999 with many DBH‑derived attributes (ratio, rainfall, altitude, etc.).  
- The massive collinearity indicated that most engineered variables are mathematically derived from a few base measurements, inflating dimensionality without adding new information.

---

### 4. Feature Pruning  

**Strategy** – Keep the 30 most informative features (gain‑ranked) and discard the rest.  

**Attributes pruned (121 total, sample):**  
`Surv_raw, FormSum_Ht, Age_Latitude, VigDivDBH_Frosts, Species_DBH, Slenderness_log, TestConstZero, Age_filled_test, Age_Vig_Rainfall_Volume, Age_Frosts_Rainfall_sq, Species_Surv, Slenderness, sq_Volume_per_Age, Ht_cu, FormSum_Ht_Frosts, …`

**Remaining key attributes (30):**  

`Age_Vig_cu, SurvivalVig_DBHFrosts_cu, Age_Vig_DBHFrosts, sq_Health, Age_Vig_Rainfall_sq, Age_Vig_Rainfall, Age_Vig, Age_Vig_sq, Age_Vig_Volume_sq, Age_Vig_Latitude_sq, log_Health, Health, Age_Surv, Age_Surv_Altitude, SurvivalVig_DBHFrosts, Age_Vig_Latitude, FormSum_HtLatitude, Age_Vig_Volume_log, Sp_fr_Volume, log_Frosts_Alt, Age_Frosts, Sp_fr_Health, Age_Surv_Rainfall_sq, Age_Vig_InsRes, Age_Vig_Volume_Species, Altitude_copy, Vig_DBH_cu, Form_Avg, Age_Vig_Rainfall_cu, Vig_DBH`.

---

### 5. Post‑pruning Results  

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.703** |
| **Macro‑average F1** | 0.682 |
| **Weighted‑average F1** | 0.698 |
| **Best‑class (none) F1** | 0.957 |
| **Worst‑class (average) F1** | 0.423 |

*Performance improved slightly despite a 75 % reduction in feature count, confirming that the pruned set retains the predictive signal while removing noise and redundancy.*

---

### 6. Observations & Conclusions  

1. **Predictive Power** – Age‑related interaction terms (`Age_Vig_*`) dominate predictive importance, suggesting that the combination of tree age and vigor is the strongest determinant of utility rating.  
2. **Health Metrics** – Both raw (`Health`) and transformed (`sq_Health`, `log_Health`) features are consistently high‑importance, highlighting the role of tree health in utility.  
3. **Volume & Geometry** – Features linking volume, latitude, and rainfall (`Age_Vig_Volume_*`, `FormSum_HtLatitude`) also contribute meaningfully.  
4. **Redundancy** – Over 800 highly correlated pairs indicate that the original feature engineering produced many near‑duplicate variables; aggressive pruning is essential for model efficiency and interpretability.  
5. **Robustness** – Accuracy remained stable (±0.01) after removing 121 low‑importance attributes, demonstrating that the model’s predictive capability is robust to feature reduction.  

**Next Steps for the Team**  
- Communicate the retained 30‑feature set to the Scientist and Extractor agents for further hypothesis refinement.  
- Consider exploring interaction effects among the top features (e.g., SHAP interaction values) to deepen mechanistic insight, but avoid additional feature engineering per current guidelines.  

*Report compiled by the Tester Agent.*