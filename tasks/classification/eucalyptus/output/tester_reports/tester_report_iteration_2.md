**Tester Agent – Feature Evaluation Report  
Eucalyptus Utility Rating Classification**

---

### 1. Evaluation Strategy
1. **Baseline Model (All features)**
   * Algorithm: **XGBoost (multiclass)** – `tree_method='hist'`, 150 trees, depth 5.  
   * Train‑test split: 80 % / 20 % (stratified, random 42).  
   * Metrics: Accuracy, macro‑averaged F1‑score, per‑class precision/recall.  
   * Feature importance derived from XGBoost **gain** values.

2. **Redundancy Analysis**
   * Computed absolute Pearson correlation matrix across all 24 attributes.  
   * Identified pairs with **ρ > 0.90** as highly redundant.

3. **Pruning**
   * Removed attributes that were nearly perfectly collinear (ρ ≈ 1.0) and offered no unique information.  
   * Kept a single representative from each correlated group.

4. **Post‑pruning Model**
   * Re‑trained the same XGBoost configuration on the reduced 10‑feature set.  
   * Re‑evaluated the same metrics and importance.

---

### 2. Baseline Results (24 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.682** |
| **Macro F1** | **0.657** |
| Weighted F1 | 0.679 |
| **Top‑10 gain importance** | Vig_DBH_sq, SurvivalVig_DBHFrosts, Vig_DBH_Frosts, Vig_InsRes_product, Vig_DBH, Survival_Vig_InsRes, DBH_Frosts, FormSum_HtLatitude, Form_Avg, Form_Sum |

**Redundancy Findings**  
- **DBH‑derived group** (DBH_Ht_product, DBH_Ht_ratio, DBH_Rainfall, DBH_Altitude, DBH_Frosts, DBH_Latitude, Survival_DBH, Vig_DBH, InsRes_DBH, Vig_DBH_Frosts, Vig_DBH_sq, InsResDBH_HtLatitude, SurvivalVig_DBHFrosts) showed correlations **> 0.999**.  
- **Form_Sum ↔ Form_Avg** correlation = **1.00**.  
- **Ht_Rainfall ↔ Ht_Latitude** correlation = **0.954**.

---

### 3. Pruned Feature Set (10 features)

| Retained Attribute | Reason |
|--------------------|--------|
| DBH_Ht_product | Representative of the entire DBH‑derived cluster. |
| Ht_Rainfall | Retained (only Ht_Latitude removed). |
| Ht_Altitude | Provides distinct height‑related information. |
| Ht_Frosts | Adds climatic nuance. |
| Form_Sum | Keeps form information (Form_Avg removed). |
| FormSum_HtLatitude | Captures interaction between form and latitude. |
| Survival_Vig | Core survival‑vigour metric. |
| Survival_Vig_InsRes | Interaction term, still informative. |
| Vig_InsRes_product | Strong predictive contribution. |
| VigDivDBH_Frosts | Interaction term with distinct signal. |

**Post‑pruning Performance**

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.662** |
| **Macro F1** | **0.642** |
| Weighted F1 | 0.660 |
| **Top‑10 gain importance** | Survival_Vig_InsRes, Vig_InsRes_product, FormSum_HtLatitude, Survival_Vig, Ht_Frosts, Form_Sum, Ht_Altitude, DBH_Ht_product, VigDivDBH_Frosts, Ht_Rainfall |

**Remaining Correlation**  
- Only **Survival_Vig ↔ Survival_Vig_InsRes** with ρ = 0.943 (acceptable; both retain distinct interaction information).

---

### 4. Key Take‑aways

| Aspect | Observation |
|--------|-------------|
| **Predictive Power** | The full feature set achieved modestly higher accuracy (0.682) than the reduced set (0.662). The drop is small (~2 %) given a > 50 % reduction in dimensionality. |
| **Feature Importance** | After pruning, importance concentrates on a few interaction terms (Survival_Vig_InsRes, Vig_InsRes_product) and the original DBH‑height product, indicating they capture most of the discriminative signal. |
| **Redundancy** | Over‑half of the original attributes were virtually duplicate (ρ ≈ 1). Removing them simplifies the model without major loss of performance. |
| **Robustness** | The reduced model, with fewer highly collinear features, is expected to be more stable to noise and easier to interpret. |
| **Statistical Relationships** | Only one high‑correlation pair remains; all other features contribute relatively independent information. |

---

### 5. Recommended Next Steps (for the Scientist & Extractor)

1. **Focus Extraction** on the retained 10 attributes and consider generating additional *orthogonal* interaction terms if scientifically justified.  
2. **Validate Robustness** by adding synthetic noise to the features and re‑evaluating stability (optional for future testing).  
3. **Document** the pruned attributes list (see below) for the Extractor to avoid re‑creating them.

**Pruned Attributes (removed):**  
`DBH_Ht_ratio, DBH_Rainfall, DBH_Altitude, DBH_Frosts, DBH_Latitude, Survival_DBH, Vig_DBH, InsRes_DBH, Vig_DBH_Frosts, Vig_DBH_sq, InsResDBH_HtLatitude, SurvivalVig_DBHFrosts, Form_Avg, Ht_Latitude`

---

*All experiments executed with reproducible code and recorded notes. The feature set is now concise, interpretable, and retains the majority of predictive capability for the eucalyptus utility rating task.*