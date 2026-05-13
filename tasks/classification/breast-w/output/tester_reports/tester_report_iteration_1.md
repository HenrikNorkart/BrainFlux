**Feature Evaluation Report – Breast Cancer Diagnosis (Derived Feature Set)**  

**1. Baseline Model (All 9 features)**  
- **Algorithm:** Logistic Regression (max_iter = 1000, n_jobs = 5)  
- **Performance:**  
  - Accuracy: **0.964**  
  - ROC‑AUC: **0.998**  

**2. Feature Importance (Permutation Importance)**  

| Feature | Mean Importance | Std. Dev. |
|---------|----------------|-----------|
| **total_morphology_score** | **0.386** | 0.023 |
| clump_mitoses_sum | 0.011 | 0.003 |
| nuclear_shape_product | 0.006 | 0.005 |
| log_product_all | 0.006 | 0.009 |
| shape_irregularity_score | 0.000 | 0.000 |
| texture_score | 0.000 | 0.000 |
| mitotic_activity | –0.006 | 0.005 |
| nuclear_to_texture_ratio | –0.006 | 0.002 |
| nuclear_size_score | –0.008 | 0.006 |

- **Key Insight:** *total_morphology_score* dominates predictive power.  
- Features with near‑zero or negative importance (shape_irregularity_score, texture_score, mitotic_activity, nuclear_to_texture_ratio, nuclear_size_score) contribute little or act as noise.

**3. Feature Pruning**  
The five low‑impact attributes were removed using the `attribute_pruning_tool`:
- shape_irregularity_score  
- texture_score  
- mitotic_activity  
- nuclear_to_texture_ratio  
- nuclear_size_score  

**4. Model After Pruning (4 retained features)**  
- **Features kept:** total_morphology_score, clump_mitoses_sum, nuclear_shape_product, log_product_all  
- **Performance:**  
  - Accuracy: **0.964** (unchanged)  
  - ROC‑AUC: **0.998** (unchanged)  

**5. Conclusions**  
- The predictive signal for tumor malignancy is almost entirely captured by **total_morphology_score**, with modest additional contributions from the three remaining features.  
- Pruning the five negligible attributes does **not** degrade model performance, simplifying the feature set while preserving predictive power.  
- The final, compact feature set (4 attributes) is recommended for downstream modeling and deployment.  

*All observations are recorded in the internal notes and the unnecessary attributes have been pruned.*