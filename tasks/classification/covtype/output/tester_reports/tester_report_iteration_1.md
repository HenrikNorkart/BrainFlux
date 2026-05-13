**Feature Evaluation Report – Forest Cover Type (7‑class)  
Extractor‑generated attributes: 3 (Soil_Wilderness_Interaction, Wilderness_Area_Code, Soil_Type_Code)**  

---

### 1. Experimental Setup  
- **Model**: RandomForestClassifier (200 trees, `n_jobs=4`, `random_state=42`).  
- **Train‑test split**: 80 % / 20 % stratified on the target.  
- **Metrics**: overall accuracy, weighted F1, per‑class precision/recall/F1.  
- **Feature‑importance**: mean decrease in impurity (RF feature_importances_).  
- **Redundancy check**: Pearson correlation among the top‑gain features.  

---

### 2. Baseline Results (All 3 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.5692** |
| **Weighted F1** | **0.5010** |
| **Macro F1** | 0.3324 |

**Per‑class performance (selected)**  

| Class | Precision | Recall | F1 |
|------|-----------|--------|----|
| 1 | 0.628 | 0.191 | 0.293 |
| 2 | 0.559 | 0.912 | 0.693 |
| 3 | 0.587 | 0.601 | 0.594 |
| 4 | 0.540 | 0.137 | 0.218 |
| 5 | 0.000 | 0.000 | 0.000 |
| 6 | 0.000 | 0.000 | 0.000 |
| 7 | 0.583 | 0.484 | 0.529 |

**Feature‑importance (gain from RF)**  

| Feature | Importance |
|---------|------------|
| Soil_Wilderness_Interaction | **0.461** |
| Wilderness_Area_Code | 0.343 |
| Soil_Type_Code | 0.196 |

**Correlation among top‑20 gain features**  

- **Soil_Wilderness_Interaction ↔ Wilderness_Area_Code**: *r = 0.9996* (near‑perfect collinearity).  

---

### 3. Redundancy Handling  

Because the two features above are almost perfectly correlated, one can be removed without loss of information.  
- **Decision**: prune **Wilderness_Area_Code** (lower importance).  

Pruning performed via `attribute_pruning_tool`.

---

### 4. Post‑pruning Evaluation (2 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.5692** (unchanged) |
| **Weighted F1** | **0.5010** (unchanged) |
| **Macro F1** | 0.3324 (unchanged) |

**Feature‑importance (2 remaining features)**  

| Feature | Importance |
|---------|------------|
| Soil_Wilderness_Interaction | **0.822** |
| Soil_Type_Code | 0.178 |

The predictive performance is identical to the baseline, confirming that **Wilderness_Area_Code** contributed no unique information beyond the interaction term.

---

### 5. Conclusions  

1. **Predictive Power** – The three extracted attributes together achieve modest accuracy (≈ 57 %) and weighted F1 (≈ 0.50) on the 7‑class forest‑cover problem.  
2. **Feature Importance** –  
   - **Soil_Wilderness_Interaction** is the dominant predictor (≈ 82 % of importance after pruning).  
   - **Soil_Type_Code** provides complementary information (≈ 18 %).  
3. **Redundancy** – Wilderness_Area_Code is essentially a linear proxy of the interaction term (correlation ≈ 1.0) and can be safely removed.  
4. **Robustness** – Removing the redundant feature does **not** degrade any evaluated metric, indicating robustness of the remaining feature set.  
5. **Final Feature Set** – **Soil_Wilderness_Interaction** and **Soil_Type_Code** (2 attributes) constitute a compact, non‑redundant set that retains the full predictive capability of the original three‑attribute bundle.  

---

**Next Steps for the Team**  

- Communicate the pruned feature list to the Scientist Agent for documentation.  
- If further performance improvements are required, the Scientist Agent may explore additional interaction terms or higher‑order transformations, but the current set is already minimal and non‑redundant.  

*Report compiled by the Tester Agent.*