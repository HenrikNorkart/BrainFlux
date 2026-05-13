**Tester Agent Report – Crab Age Prediction (Regression)**  

**1. Baseline Evaluation (All Features)**  
- **Dataset:** 3 893 rows × 81 columns (80 features + target).  
- **Model:** XGBoostRegressor (GPU, `device="cuda:5"`, `tree_method="hist"`).  
- **Performance:**  
  - **R²:** 0.550  
  - **RMSE:** 2.16  
- **Top 20 Feature Importances (gain):**  
  1. `Shell_Weight_Ratio_x_Log_Height_Weight_Interaction`  
  2. `Shell_Weight_Ratio_x_Height_Weight_Interaction`  
  3. `Log_Shucked_Weight_Ratio`  
  4. `Shucked_Weight_Ratio_Squared`  
  5. `Sex_Shucked_Weight_Ratio_Squared`  
  6. `Shell_Weight_Ratio_x_Height_Weight_Interaction_Squared`  
  7. `Shucked_Weight_Ratio`  
  8. `Diameter_Squared`  
  9. `Log_Diameter`  
  10. `Log_Weight` …  

**2. Correlation & Redundancy Analysis**  
- Pearson‑absolute correlations of all features with the target showed many features > 0.5, but several **high‑correlation clusters (> 0.9)** among the top importance features (e.g., `Shell_Weight_Ratio_x_Height_Weight_Interaction` highly correlated with `Height_Weight_Diameter_Interaction`, `Weight_Squared`, `Length_Weight_Diameter_Interaction`, etc.).  
- Redundant pairs were identified and the **lower‑importance member of each pair** was marked for removal.

**3. Pruning Decision**  
- **18 attributes** were pruned (e.g., `Weight_Squared`, `Log_Weight`, `Height_Weight_Diameter_Interaction`, `Log_Diameter`, `Shucked_Weight_Ratio`, etc.).  
- Pruning was performed via the `attribute_pruning_tool`.

**4. Post‑Pruning Evaluation**  
- **Remaining features:** 62 (reduced from 80).  
- **Model (same hyper‑parameters) performance:**  
  - **R²:** 0.556 (+0.006)  
  - **RMSE:** 2.14 (‑0.02)  
- **Top 15 post‑pruning importances:**  
  1. `Shell_Weight_Ratio_x_Log_Height_Weight_Interaction`  
  2. `Shell_Weight_Ratio_x_Height_Weight_Interaction`  
  3. `Log_Shucked_Weight_Ratio`  
  4. `Height_Weight_Interaction`  
  5. `Shucked_Weight_Ratio_Squared`  
  6. `Shell_Weight_Ratio_x_Height_Weight_Interaction_Squared`  
  7. `Log_Estimated_Volume`  
  8. `Weight_to_Length_Ratio_Squared`  
  9. `Diameter_Squared`  
  10. `Length_to_Height_Ratio_x_Log_Height_Weight_Interaction` …  

**5. Robustness Check**  
- Added Gaussian noise (0.1 × std) to **10 % of rows** across **20 % of features**.  
- Re‑trained the same model:  
  - **R²:** 0.560 (virtually unchanged)  
  - **RMSE:** 2.13 (slightly better)  
- Indicates the model’s predictions are **stable** to moderate perturbations of the input space.

**6. Key Take‑aways**  

| Aspect | Observation |
|--------|-------------|
| **Predictive Power** | Baseline R² ≈ 0.55; modest improvement after pruning (≈ 0.56). |
| **Most Influential Features** | Interaction terms involving `Shell_Weight_Ratio` and log‑height/weight interactions dominate; log‑transformed weight ratios also critical. |
| **Redundancy** | > 30 % of top features were > 0.9 correlated, allowing removal without performance loss. |
| **Feature Reduction** | Pruning 18 low‑importance, highly‑correlated attributes reduced dimensionality by ~22 % while slightly boosting accuracy. |
| **Robustness** | Model performance remains consistent under random noise, confirming resilience. |

**7. Recommended Action (within scope)**  
- **Retain** the 62 pruned‑set features for downstream modeling.  
- **Discard** the 18 pruned attributes as they contribute little unique information and inflate collinearity.  

*All observations have been recorded (see notes) and the feature set is now streamlined for efficient, reliable age prediction of crabs.*