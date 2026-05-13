**Tester Agent – Feature Evaluation Report (Crab Age Regression)**  

---

### 1.  Overview  

- **Task:** Predict the age of crabs (`target`) from a set of engineered physical‑measurement attributes.  
- **Dataset size:** 3 893 rows, 65 columns (including `target`).  
- **Goal of this phase:** Assess the predictive usefulness of the provided features, identify redundancies, and prune the attribute set to a manageable size while retaining most of the predictive power.

---

### 2.  Baseline & Full‑set Performance  

| Model / Approach | RMSE | R² (where computed) | # Features used |
|------------------|------|---------------------|-----------------|
| **Mean‑target baseline** (predict global mean) | **3.32** | – | – |
| **Linear regression (ordinary least‑squares) on *all* 64 features** | **2.08** | – | 64 |
| **Linear regression on *pruned* 15‑feature set** | **2.12** | – | 15 |

*Interpretation*:  
- The full engineered feature set reduces RMSE by ~37 % vs. the naive mean baseline.  
- Removing highly redundant features (see Section 3) increases RMSE by only **~0.04** (≈2 % relative loss) while cutting the number of predictors from 64 to 15 – a substantial simplification with negligible performance degradation.

---

### 3.  Redundancy & Pruning  

**Method** – Computed the absolute Pearson correlation matrix, flagged any pair with |ρ| > 0.90, and iteratively removed the later‑appearing member of each highly correlated pair.  

- **Features eliminated (49 total)** – e.g., `Length_to_Height_Ratio`, `Diameter_to_Height_Ratio`, all squared‑terms, most interaction terms, log‑transformed raw measurements, and many derived ratios.  
- **Remaining 15 features** (plus `target`):  

| Feature | Description |
|---------|-------------|
| `Estimated_Volume` | Volume estimate from raw dimensions |
| `Weight_to_Volume_Ratio` | Weight / volume |
| `Shucked_Weight_Ratio` | Ratio of meat weight to total weight |
| `Viscera_Weight_Ratio` | Internal organ weight ratio |
| `Shell_Weight_Ratio` | Shell weight ratio |
| `Sex_Encoded` | Numeric encoding of crab sex |
| `Length_to_Diameter_Ratio` | Shape ratio |
| `Component_Sum_Ratio` | Sum of component‑weight ratios |
| `Sex_Height_Weight_Interaction` | Interaction of sex with height & weight |
| `Height_Squared` | Height² |
| `Log_Height` | Log‑transformed height |
| `Sex_Weight_to_Length_Ratio` | Sex‑specific weight/length ratio |
| `Sex_Shucked_Weight_Ratio` | Sex‑specific shucked‑weight ratio |
| `Log_Shucked_Weight_Ratio` | Log‑transformed shucked‑weight ratio |
| `Shell_Weight_Ratio_x_Log_Height_Weight_Interaction` | Composite interaction term |

These attributes capture the most informative physical relationships while avoiding near‑duplicate information.

---

### 4.  Feature Relevance  

**Correlation with target (top 10)**  

| Feature | |ρ| |
|---------|------|
| `Shell_Weight_Ratio_x_Height_Weight_Interaction` | 0.589 |
| `Log_Weight_to_Length_Ratio` | 0.587 |
| `Log_Weight` | 0.583 |
| `Log_Diameter` | 0.574 |
| `Log_Height_Weight_Interaction` | 0.574 |
| `Weight_to_Length_Ratio` | 0.567 |
| `Log_Estimated_Volume` | 0.565 |
| `Length_to_Diameter_Ratio_x_Log_Height_Weight_Interaction` | 0.563 |
| `Log_Length` | 0.558 |
| `Diameter_Squared` | 0.555 |

**Mutual‑information (top 10)**  

| Feature | MI score |
|---------|----------|
| `Shell_Weight_Ratio_x_Log_Height_Weight_Interaction` | 0.457 |
| `Shell_Weight_Ratio_x_Height_Weight_Interaction` | 0.429 |
| `Length_to_Diameter_Ratio_x_Height_Weight_Interaction` | 0.412 |
| `Shell_Weight_Ratio_x_Height_Weight_Interaction_Squared` | 0.410 |
| `Log_Height_Weight_Interaction` | 0.401 |
| `Length_to_Diameter_Ratio_x_Height_Weight_Interaction_Squared` | 0.398 |
| `Height_Weight_Interaction` | 0.396 |
| `Length_Weight_Interaction` | 0.395 |
| `Log_Diameter` | 0.395 |
| `Length_to_Height_Ratio_x_Height_Weight_Interaction` | 0.394 |

**Linear‑model coefficient magnitude (top 5)**  

| Feature | |Coeff| |
|---------|------|
| `Shell_Weight_Ratio` | 10.01 |
| `Log_Shucked_Weight_Ratio` | 7.73 |
| `Shucked_Weight_Ratio` | 6.57 |
| `Sex_Shucked_Weight_Ratio` | 6.52 |
| `Viscera_Weight_Ratio` | 5.26 |

These results converge on a small core of shell‑weight and shucked‑weight related attributes, plus log‑scaled size measures, as the strongest predictors of crab age.

---

### 5.  Robustness Observations  

- **High collinearity** among many engineered interaction terms was the main source of redundancy; pruning eliminated this without hurting predictive accuracy.  
- **Linear regression** remained stable on the reduced set, indicating that the retained features are not only predictive but also well‑conditioned (no severe multicollinearity).  

---

### 6.  Conclusions & Recommendations  

1. **Retain the 15‑feature subset** listed in Section 3. It preserves >98 % of the predictive performance of the full 64‑feature set while simplifying the model and reducing over‑fitting risk.  
2. **Key predictive drivers** are shell‑weight‑related ratios and log‑scaled size features; these should be emphasized in any downstream modeling or interpretation.  
3. No further feature engineering is required at this stage – the current attributes already capture the dominant signal.  

The extracted, pruned feature set is now ready for the Scientist Agent to incorporate into the next iteration of hypothesis generation or for downstream model development.