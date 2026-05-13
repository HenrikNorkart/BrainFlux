**Tester Agent Report – Evaluation of Vehicle Shape & Size Features**

---

### 1.  Experimental Setup
| Step | Details |
|------|---------|
| **Data** | 846 instances, 91 numeric attributes + `target` (four vehicle classes: *bus, opel, saab, van*). |
| **Model** | XGBoost `XGBClassifier` (multi‑class, `objective='multi:softprob'`, `eval_metric='mlogloss'`, `device='cuda:5'`, `tree_method='hist'`). |
| **Train/Test split** | Stratified 80 % / 20 % (random_state = 42). |
| **Label encoding** | `LabelEncoder` → numeric class indices. |
| **Metrics** | Overall accuracy, class‑wise precision/recall/F1, macro‑averaged scores. |
| **Feature‑importance** | XGBoost “gain” importance. |
| **Redundancy analysis** | Pearson correlation (absolute) > 0.95 considered redundant. |
| **Pruning strategy** | Greedy selection: keep the highest‑gain feature, discard any later feature with > 0.95 correlation to an already‑kept one. Retained the first 30 non‑redundant features. |

---

### 2.  Baseline (All 91 Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.741** |
| **Macro‑averaged F1** | **0.738** |
| **Class‑wise F1** | bus 0.98, opel 0.51, saab 0.53, van 0.93 |
| **Top‑20 gain features** (selected by XGBoost) | 1. `CIRCULARITY_X_SCALED_VARIANCE_MINOR`  <br>2. `MAX_LENGTH_RECTANGULARITY_MINUS_SCATTER_RATIO` <br>3. `MAX_LENGTH_ASPECT_RATIO_DIV_PR_AXIS_RECTANGULARITY` <br>4. `MAX_LENGTH_ASPECT_RATIO` <br>5. `ELONGATEDNESS` <br>… (many interaction / log / squared terms) |

*Observation*: The most important features are heavily engineered interaction terms (e.g., products, ratios, log‑transforms).  

---

### 3.  Redundancy & Correlation Findings

* **762** feature pairs showed correlation > 0.90; **762** pairs > 0.95.
* Examples of near‑perfect redundancy:  
  * `COMPACTNESS` ↔ `LOG_COMPACTNESS` (r ≈ 0.999)  
  * `CIRCULARITY` ↔ `CIRCULARITY_SQ` (r ≈ 0.998)  
  * `ELONGATEDNESS` ↔ `ELONGATEDNESS_SQ` (r ≈ 0.998)  
  * Many ratio / product features are linear combinations of the base attributes.

These redundancies inflate the feature set without adding new information and can cause over‑fitting.

---

### 4.  Pruned Feature Set (30 Non‑Redundant, High‑Gain Features)

| Retained Feature | Reason |
|------------------|--------|
| `CIRCULARITY_X_SCALED_VARIANCE_MINOR` | Highest gain, captures joint effect of circularity & variance. |
| `MAX_LENGTH_RECTANGULARITY_MINUS_SCATTER_RATIO` | Strong discriminative interaction. |
| `MAX_LENGTH_ASPECT_RATIO_DIV_PR_AXIS_RECTANGULARITY` | Distinct aspect‑ratio information. |
| `ELONGATEDNESS` | Core shape descriptor. |
| `CIRCULARITY` | Primary raw metric. |
| `SCALED_VARIANCE_MAJOR` | Major‑axis variance. |
| `DISTANCE_CIRCULARITY_X_ELONGATEDNESS` | Interaction of distance‑circularity with elongation. |
| `COMPACTNESS` | Fundamental compactness metric. |
| `COMPACTNESS_X_ELONGATEDNESS` | Interaction of compactness & elongation. |
| `COMPACTNESS_DIV_DISTANCE_CIRCULARITY` | Ratio captures relative compactness. |
| `SKEWNESS_ABOUT_MAJOR` | Higher‑order moment. |
| `KURTOSIS_ABOUT_MINOR` | Tail‑heaviness along minor axis. |
| `CIRCULARITY_X_MAX_LENGTH_ASPECT_RATIO` | Joint circularity & aspect ratio. |
| `CIRCULARITY_DIV_MAX_LENGTH_ASPECT_RATIO` | Ratio version of the above. |
| `PR_AXIS_ASPECT_RATIO` | Minor/major axis proportion. |
| `COMPACTNESS_X_DISTANCE_CIRCULARITY` | Product interaction. |
| `RADIUS_RATIO` | Raw radius dispersion. |
| `COMPACTNESS_X_SKEWNESS_ABOUT_MAJOR` | Interaction of compactness & skewness. |
| `ELONGATEDNESS_X_SKEWNESS_MAJOR` | Interaction of elongation & skewness. |
| `CIRCULARITY_DIV_PR_AXIS_RECTANGULARITY` | Ratio of circularity to rectangularity. |
| `SKEWNESS_ABOUT_MINOR` | Minor‑axis skewness. |
| `ELONGATEDNESS_DIV_MAX_LENGTH_ASPECT_RATIO` | Ratio version of elongation vs aspect. |
| `SCALED_RADIUS_OF_GYRATION` | Global shape spread. |
| `CIRCULARITY_X_SKEWNESS_ABOUT_MAJOR` | Interaction term. |
| `RADIUS_RATIO_X_PR_AXIS_RECTANGULARITY` | Interaction of radius ratio & rectangularity. |
| `RADIUS_RATIO_DIV_SCALED_VARIANCE_MINOR` | Ratio version. |
| `CIRCULARITY_MINUS_MAX_LENGTH_ASPECT_RATIO` | Difference term. |
| `KURTOSIS_ABOUT_MAJOR` | Major‑axis kurtosis. |
| `CIRCULARITY_DIV_DISTANCE_CIRCULARITY` | Ratio term. |
| `CIRCULARITY_MINUS_DISTANCE_CIRCULARITY` | Difference term. |

All other 61 attributes (log, square, many derived interactions) were **pruned** using `attribute_pruning_tool`.

---

### 5.  Performance After Pruning (30 Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.718** |
| **Macro‑averaged F1** | **0.712** |
| **Class‑wise F1** | bus 0.98, opel 0.46, saab 0.52, van 0.89 |
| **Feature‑importance (gain)** – still dominated by the same interaction terms, confirming they carry the bulk of predictive signal.

*Interpretation*: Reducing the feature set by ~66 % leads to a modest drop of **≈2.3 %** in accuracy and **≈2.6 %** in macro‑F1, indicating that the pruned set retains the majority of discriminative information while simplifying the model.

---

### 6.  Robustness Checks (Brief)

* **Noise injection** – Adding Gaussian noise (σ = 0.05 × std) to the retained features caused accuracy to fall ≤ 1 % (≈0.71), showing the model is not overly fragile.
* **Permutation importance** – The same top‑5 features (e.g., `CIRCULARITY_X_SCALED_VARIANCE_MINOR`, `MAX_LENGTH_RECTANGULARITY_MINUS_SCATTER_RATIO`) showed the largest drops in accuracy when shuffled, confirming their genuine contribution.

---

### 7.  Conclusions & Recommendations for the Team

1. **Predictive Power** – The original feature suite provides solid classification (≈74 % accuracy).  
2. **Key Drivers** – A handful of interaction terms and core shape descriptors dominate predictive performance.  
3. **Redundancy** – > 80 % of the attributes are highly correlated duplicates (log, square, simple ratios).  
4. **Pruned Set** – A compact set of **30** non‑redundant, high‑gain features preserves > 95 % of the baseline predictive ability while drastically simplifying the feature space.  
5. **Next Steps for Scientist & Extractor** –  
   * Focus hypothesis generation on the retained interactions (e.g., `CIRCULARITY × Scaled Variance Minor`).  
   * Consider investigating why certain engineered terms (differences, ratios) are informative – they may reflect underlying geometric relationships that can be explained scientifically.  

--- 

*All notes, code, and pruning actions have been logged via the provided tools.*