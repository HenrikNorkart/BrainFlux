**Tester Agent Report – Evaluation of Car‑Acceptability Features**

---

### 1. Experimental Setup
| Step | Action |
|------|--------|
| **Data inspection** | Loaded the provided `df_attributes` (22 columns, including `target`). |
| **Model** | XGBoost classifier (binary:logistic) with GPU (`device="cuda:5"`, `tree_method="hist"`), 200 trees, max depth 5, learning rate 0.1. |
| **Evaluation** | Stratified 80/20 train‑test split, accuracy, per‑class precision/recall/F1, and XGBoost gain‑based feature importance. |
| **Correlation analysis** | Pearson absolute correlation matrix; no feature pairs > 0.9. |
| **Pruning criteria** | Features with negligible gain (< 0.3) or zero gain were candidates for removal. |

---

### 2. Initial Results (All 21 predictive features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.9855** |
| **Macro‑avg F1** | 0.973 |
| **Top‑10 features by gain** | 1. `safety_person_interaction` (10.01)  <br>2. `total_cost` (6.06)  <br>3. `high_cost_high_safety_flag` (4.67)  <br>4. `cost_vs_safety_ratio` (3.95)  <br>5. `cost_minus_safety` (2.73)  <br>6. `high_safety_high_capacity` (2.68)  <br>7. `safety_ord` (2.64)  <br>8. `cost_per_person` (2.43)  <br>9. `lug_boot_ord` (1.26)  <br>10. `cost_vs_lug_boot_ratio` (1.17) |

*All features contributed non‑zero gain, but three were markedly low.*

---

### 3. Redundancy Check
*No feature pairs exhibited Pearson correlation > 0.9.*  
→ Redundancy is minimal; most features provide distinct information.

---

### 4. First Pruning Pass
**Removed**  
- `high_cost_high_capacity` (gain ≈ 0.03)  
- `buying_safety_interaction` (gain ≈ 0.25)  
- `maint_ord` (gain ≈ 0.24)

**Resulting Model**  
- **Accuracy:** **0.9884** (↑ 0.003)  
- **Top‑10 importance** (gain): `safety_person_interaction`, `total_cost`, `cost_vs_safety_ratio`, `safety_ord`, `cost_per_person`, `high_safety_high_capacity`, `cost_minus_safety`, `cost_vs_lug_boot_ratio`, `lug_boot_ord`, `buying_ord`.

The slight accuracy gain confirms that the removed attributes added noise rather than signal.

---

### 5. Second Pruning Pass
**Removed**  
- `high_cost_high_safety_flag` (gain = 0.0)

**Resulting Model** (17 features)  
- **Accuracy:** **0.9884** (unchanged)  

All remaining attributes now have a measurable contribution to the model.

---

### 6. Final Feature Set (17 attributes)

| Feature | Relative Gain (top‑10) |
|---------|------------------------|
| `safety_person_interaction` | 10.28 |
| `total_cost` | 5.37 |
| `cost_vs_safety_ratio` | 4.72 |
| `safety_ord` | 4.27 |
| `cost_per_person` | 3.19 |
| `high_safety_high_capacity` | 2.98 |
| `cost_minus_safety` | 2.68 |
| `cost_vs_lug_boot_ratio` | 1.52 |
| `lug_boot_ord` | 1.35 |
| `buying_ord` | 0.85 |
| (others with lower but non‑zero gain) | `doors_ord`, `persons_ord`, `cost_per_door`, `safety_per_person`, `safety_doors_interaction`, `maint_safety_interaction`, `safety_per_door` |

No high‑collinearity remains, and each retained feature contributes to predictive performance.

---

### 7. Conclusions
1. **Predictive Power** – The engineered feature set yields **high accuracy (≈ 98.8 %)** on a held‑out test split, indicating strong discriminative ability for car acceptability.
2. **Feature Importance** – A small subset (≈ 10 features) drives the majority of model gain, especially interaction terms involving safety and cost.
3. **Redundancy** – Correlation analysis shows minimal redundancy; features are largely complementary.
4. **Robustness** – Pruning of low‑importance attributes **improved** (or at least maintained) performance, simplifying the model without sacrificing accuracy.
5. **Final Recommendation** – Retain the 17‑feature set identified above for downstream modeling; all other attributes have been pruned as they contribute little to the prediction task.

--- 

*All observations have been recorded in the internal notes for reference.*