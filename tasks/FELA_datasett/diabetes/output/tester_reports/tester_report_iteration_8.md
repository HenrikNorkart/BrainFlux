**Tester Agent – Feature Evaluation Report**

**1. Experimental Setup**  
- Model: XGBoost (multi:softprob) with GPU (`device="cuda:5"`, `tree_method="hist"`).  
- Data: `df_attributes` (pre‑extracted features, target = diabetes type 0/1/2).  
- Train‑test split: 80 % / 20 %, stratified by target.  
- Metrics: Overall **accuracy** and **macro‑averaged AUC** (multi‑class).  

**2. Baseline Results (All Features)**  
- **Accuracy:** 0.8506  
- **Macro AUC:** 0.7804  

**Top 15 features by XGBoost gain (baseline)**  
| Rank | Feature | Gain |
|------|---------|------|
|1|GenHlth_MetRisk|66.67|
|2|ComorbidityScore_MetRisk|23.83|
|3|HeavyAlcohol|16.22|
|4|CholCheck_MetRisk|14.06|
|5|Age_ComorbidityScore|9.45|
|6|ComorbidityScore|8.57|
|7|ExtendedHolisticRiskScore_HeavyAlcohol|6.46|
|8|Age_Sex|5.01|
|9|BMI|4.85|
|10|HighBP_BMI|4.57|
|…|…|…|

**3. Redundancy & Correlation Analysis**  
- Several feature pairs showed **perfect or near‑perfect correlation (≥0.9)**, indicating redundancy:  
  - `PhysActivity` ↔ `PhysicalActivityScore` (ρ = 1.0)  
  - `FruitVegScore` ↔ `HealthyDietScore` (ρ = 1.0)  
  - `BMI_Sex` ↔ `Sex` (ρ = 0.964)  
  - Numerous interaction features (`*_Sex`, `*_MetRisk`) highly correlated with their base variables.  

**4. Pruning Decision**  
Using the correlation findings, the following clearly redundant attributes were removed:  

- `PhysicalActivityScore`  
- `PhysActivity`  
- `PhysActivity_Sex`  
- `HealthyDietScore`  
- `BMI_Sex`  
- `FruitVegScore`  

**5. Post‑Pruning Results**  
- **Accuracy:** 0.8499 (virtually unchanged)  
- **Macro AUC:** 0.7815 (virtually unchanged)  

**Top 15 features after pruning** (gain values slightly adjusted)  

| Rank | Feature | Gain |
|------|---------|------|
|1|GenHlth_MetRisk|63.59|
|2|ComorbidityScore_MetRisk|34.49|
|3|HeavyAlcohol|18.14|
|4|CholCheck_MetRisk|11.88|
|5|Age_ComorbidityScore|8.49|
|6|MetabolicRiskScore|6.73|
|7|ExtendedHolisticRiskScore_HeavyAlcohol|5.86|
|8|ComorbidityScore|5.59|
|9|HighBP_BMI|4.75|
|10|GenHlth_Sex|4.68|
|11|Age_Sex|4.60|
|12|BMI|4.54|
|13|Age|3.69|
|14|DiffWalk_Sex|3.24|
|15|SocioEconomicScore|3.21|

**6. Interpretation**  
- **Predictive Power:** The feature set reliably predicts diabetes type (≈85 % accuracy, 0.78 macro‑AUC).  
- **Key Predictors:** Interaction‑rich risk scores that combine general health, comorbidity, metabolic risk, and lifestyle (e.g., `GenHlth_MetRisk`, `ComorbidityScore_MetRisk`, `HeavyAlcohol`) dominate importance. Simple demographic variables (Age, BMI) still contribute meaningfully.  
- **Redundancy Impact:** Removing duplicated/near‑duplicate attributes did **not degrade** model performance, confirming that the retained features capture the essential signal.  
- **Robustness:** Consistent performance before and after pruning indicates stability of the predictive signal against minor feature set changes.  

**7. Recommendations for the Team**  
- **Keep** the high‑gain interaction features listed above; they are the strongest drivers of prediction.  
- **Discard** the pruned redundant attributes (already removed).  
- **Future work** could explore further interaction generation or regularisation, but the current feature set is already concise and effective.  

*End of report.*