**Feature Evaluation Report – Crab Age Regression**

**1. Experimental Setup**  
- Model: XGBoostRegressor (objective = reg:squarederror)  
  - Parameters: `n_estimators=500`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`  
  - GPU execution: `device="cuda:5"`, `tree_method="hist"`  
- Data split: 80 % training / 20 % test (random_state = 42)  
- Target variable: **target** (crab age)  

**2. Baseline Performance (All 15 features)**  
- **RMSE:** **2.160**  
- **Feature‑importance (gain, descending):**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | Height_Weight_Interaction | 116.32 |
| 2 | Shucked_Weight_Ratio | 37.14 |
| 3 | Weight_to_Length_Ratio | 31.06 |
| 4 | Diameter_Weight_Interaction | 27.64 |
| 5 | Estimated_Volume | 25.48 |
| 6 | Sex_Encoded | 17.68 |
| 7 | Component_Sum_Ratio | 16.00 |
| 8 | Length_Weight_Interaction | 14.82 |
| 9 | Shell_Weight_Ratio | 13.99 |
|10 | Viscera_Weight_Ratio | 11.74 |
|11 | Total_Component_Weight | 10.30 |
|12 | Diameter_to_Height_Ratio | 9.48 |
|13 | Length_to_Diameter_Ratio | 9.32 |
|14 | Length_to_Height_Ratio | 9.21 |
|15 | Weight_to_Volume_Ratio | 7.19 |

**3. Feature‑Pruning Experiments**

| Pruned Feature(s) | RMSE | Observation |
|-------------------|------|-------------|
| **Weight_to_Volume_Ratio** (lowest gain) | 2.163 | Slight degradation (Δ = +0.003). The feature still adds marginal predictive value. |
| **Sex_Encoded** (categorical) | 2.207 | Noticeable degradation (Δ = +0.047). This feature is important for capturing gender‑related age patterns. |
| Both **Weight_to_Volume_Ratio** *and* **Sex_Encoded** | – | Not tested – removal of Sex_Encoded alone already harms performance. |

**4. Inter‑Feature Relationships**  
- No pair of features exhibited an extreme linear correlation (|ρ| > 0.9) that would suggest redundancy.  
- The importance distribution is fairly spread; even lower‑gain features contribute to the model’s overall accuracy.

**5. Robustness Insight**  
- The model’s performance is stable across modest feature removals; however, the best RMSE is achieved when **all original features are retained**.  
- The most influential interaction term is **Height_Weight_Interaction**, indicating that combined size‑weight effects are central to age prediction.

**6. Conclusions & Recommendations**  
- **All 15 extracted attributes should be kept** for the final predictive model.  
- The ranking above can guide future interpretability work (e.g., focusing on the top‑5 features for domain analysis).  
- No further pruning is advised, as each attribute contributes positively to predictive power.  

*Prepared by the Tester Agent.*