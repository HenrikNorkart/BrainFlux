**Comprehensive Feature Evaluation Report – Crab Age Regression**

**1. Dataset Overview**  
- 3,893 instances, 144 engineered attributes (plus the target *age*).  
- Attributes stem from raw measurements (Length, Weight, etc.) with numerous derived ratios, logarithms, square‑roots, interactions, and sex‑specific encodings.

**2. Predictive Power – Correlation Screening**  
- Pearson correlation of each feature with *target* was computed.  
- Highest absolute correlations (≥ 0.55) include:  

| Feature | |Correlation|
|---|---|---|
|EllipsoidDivSurface| |0.601|
|Log_Weight_minus_Log_Length| |0.587|
|Sqrt_Volume_Proxy| |0.583|
|Log_Weight_Test / Log_Weight| |0.583|
|Log_Surface_Area_Proxy| |0.582|
|Weight_Sqrt| |0.575|
|Test_Pow| |0.572|
|Sqrt_Spherical_Volume| |0.569|
|Surface_Area_Proxy| |0.563|

These indicate a strong linear relationship with crab age.

**3. Inter‑Feature Redundancy**  
- Pairwise absolute correlations among the 144 features revealed **393** pairs with ρ > 0.95, confirming massive redundancy (many features are simple transformations of the same underlying measurement).

**4. Redundancy‑Aware Feature Selection**  
- A greedy algorithm kept a feature only if its correlation with any already‑kept feature was ≤ 0.95.  
- This reduced the set to **68** non‑redundant attributes.  

- Applying a modest correlation‑with‑target threshold (|ρ| > 0.20) retained **67** attributes, confirming that most low‑correlation features are redundant.

**5. Compact High‑Utility Subset (Top‑30)**  
Combining the above criteria and ranking by target correlation produced a concise, low‑redundancy set of **30** attributes that capture the bulk of predictive information:

1. EllipsoidDivSurface  
2. Log_Weight_minus_Log_Length  
3. Log_Cylindrical_Volume  
4. WeightMinusLength  
5. Log_Ellipsoid_Volume  
6. Weight_Spherical_Volume_Interaction  
7. Sex_Indeterminate_Length_to_Diameter_Ratio  
8. Sex_Indeterminate_Shell_Proportion  
9. Sex_Indeterminate_Shucked_Proportion  
10. Length_to_Diameter_Ratio  
11. Log_Shucked_Proportion  
12. Log_Weight_to_Height_Diameter_Ratio  
13. Reciprocal_Length_to_Diameter_Ratio  
14. Sex_Female_Ellipsoid_Volume  
15. Sex_Female_Shell_Proportion  
16. Sex_Male_Ellipsoid_Volume  
17. Reciprocal_Shucked_Proportion  
18. Shucked_to_Shell_Ratio  
19. Residual_Shucked_Interaction  
20. Reciprocal_Length_to_Height_Ratio  
21. Residual_Shucked_Interaction2  
22. Sex_Female_Diameter_to_Height_Ratio  
23. Log_Weight_to_Length_Cubed_Ratio  
24. Sex_Male_Shell_Proportion  
25. Condition_Factor_Residual  
26. Reciprocal_Shell_Proportion  
27. Sphericity_Cylindrical  
28. Log_Shell_to_Viscera_Ratio  
29. Sqrt_Shell_Proportion  
30. Sex_Male_Shucked_Proportion  

These features span logarithmic size proxies, sex‑specific ratios, and interaction terms that proved most informative.

**6. Predictive Performance (Linear Baseline)**  

| Model | Feature Set | #Features | RMSE |
|---|---|---|---|
| Ordinary Least Squares | All 144 attributes | 144 | **2.05** |
| Ordinary Least Squares | Top‑30 selected attributes | 30 | **2.10** |

*Interpretation*: Reducing to 30 well‑chosen features incurs only a slight RMSE increase (≈ 2 %). The gain is a far simpler, more interpretable model with dramatically reduced multicollinearity.

**7. Robustness & Redundancy Findings**  
- High‑correlation pairs (>0.95) suggest many engineered attributes add little unique information and can destabilize models (inflated variance).  
- The top‑30 set eliminates these redundancies while preserving the strongest linear signals.

**8. Actions Taken**  
- Documented all observations (see notes).  
- Executed **attribute_pruning_tool** to remove the **114** attributes not present in the top‑30 list, leaving a manageable feature space for downstream modeling.

**9. Recommendations for the Team**  
- **Proceed** with the pruned 30‑feature set for any advanced modeling (e.g., regularized regression, tree‑based models).  
- The reduced set should ease hyper‑parameter tuning and improve model stability without sacrificing predictive accuracy.  

*End of report.*