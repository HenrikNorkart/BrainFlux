**Feature Evaluation Report – Plasma Retinol Prediction**

**1. Baseline Model (All Features)**
- **Model:** XGBoost Regressor (500 trees, max_depth=5, learning_rate=0.05, GPU‑enabled)
- **Features:** 60  
- **Performance:** RMSE ≈ 247.14  
- **Top‑5 importance (gain):**  
  1. `SEX_ALCOHOL_BETAPLASMA`  
  2. `QUETELET_ALCOHOL`  
  3. `AGE_BETAPLASMA`  
  4. `FIBER`  
  5. `BETAPLASMA`

**2. Importance‑Based Pruning**
- Retained features with ≥ 0.5 % of total gain (37 features).  
- **RMSE:** ≈ 243.04 (slight improvement).  
- **Observation:** Many low‑gain features contributed little to predictive power.

**3. Redundancy (Correlation) Analysis**
- Identified 16 feature pairs with absolute Pearson > 0.9.  
- **Key redundant groups:**  
  - **Alcohol‑related:** `ALCOHOL` ↔ `FAT_ALCOHOL`, `BMI_CAT_ALCOHOL`, `QUETELET_ALCOHOL`, `SEX_ALCOHOL_BETAPLASMA` (correlations 0.90–0.99).  
  - **Beta‑carotene‑related:** `BETAPLASMA` ↔ `AGE_BETAPLASMA`, `AGE_DECADE_BETAPLASMA`, `QUETELET_BETAPLASMA`, `SEX_AGEDECADE_BETAPLASMA` (correlations 0.91–0.98).  
- **Action:** Removed 9 highly correlated attributes.

**4. Final Compact Model**
- **Features retained:** 28 (selected from importance list, after dropping the redundant 9).  
- **RMSE:** ≈ 238.80 – a **~3.5 % improvement** over the full‑feature baseline while using less than half the original variables.  
- **Remaining top contributors:** `SEX_ALCOHOL_BETAPLASMA`, `ALCOHOL`, `BETAPLASMA`, `FIBER`, `QUETELET`, `CALORIES`, and interaction ratios (e.g., `ALCOHOL_CALORIES_RATIO`).

**5. Conclusions**
- **Predictive Power:** A concise set of 28 well‑chosen features predicts plasma retinol more accurately than the full set.  
- **Feature Importance:** Interaction terms involving alcohol consumption and plasma beta‑carotene dominate predictive relevance.  
- **Redundancy:** High correlations among derived interaction features can be safely eliminated without loss of performance, simplifying the model and reducing over‑fitting risk.  
- **Robustness:** The model’s performance remains stable after pruning, indicating that the retained features capture the essential signal.

**Next Steps (for the Scientist & Extractor Agents)**
- Consider focusing future extraction on the high‑importance domains (alcohol‑related and beta‑carotene‑related metrics).  
- Validate the compact feature set on external or hold‑out data to confirm robustness.  

*All observations have been recorded in the internal notes for reference.*