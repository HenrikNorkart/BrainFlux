**Feature‑Testing Report – Classification Task**  

---

### 1. Dataset & experimental set‑up
* **Rows / Instances:** 846  
* **Features (original):** 110 (plus target) – all derived shape descriptors.  
* **Target classes:** 4 (bus, saab, opel, van).  
* **Model used for evaluation:** `RandomForestClassifier` (300 trees, `n_jobs=-1`, `random_state=0`).  
* **Train‑test split:** stratified 70 % / 30 % (random_state = 42).  
* **Metrics reported:**  
  * Accuracy  
  * Macro‑averaged ROC‑AUC (one‑vs‑rest)  

---

### 2. Baseline performance (all 110 features)

| Metric | Value |
|--------|-------|
| Accuracy | **0.736** |
| Macro‑AUC | **0.917** |

The model already achieves strong discriminative ability (AUC > 0.9), indicating that the feature set is informative for the four‑class problem.

---

### 3. Feature‑level predictive power  

| Rank | Feature (type) | Univariate **Macro‑AUC** | **Mutual Information** |
|------|----------------|--------------------------|------------------------|
| 1 | `RATIO_MAXLEN_ASPECT_TO_SCALED_VAR_MINOR` | 0.506 | – |
| 2 | `INTER_VAR_RATIO_MAXLEN_ASPECT` | 0.506 | – |
| 3 | `INTER_VAR_RATIO_MAJOR_OVER_MINOR_MAXLEN_ASPECT` | 0.506 | – |
| 4 | `INTER_VAR_RATIO_MAJOR_OVER_MINOR_HOLLOWS` | 0.504 | – |
| 5 | `INTER_VAR_RATIO_HOLLOWS_RATIO` | 0.504 | – |
| … | … | … | … |
| Top MI | `VAR_MINOR_DIV_MAXLEN` | – | **0.487** |
| Top MI | `SQRT_VAR_MINOR_DIV_MAXLEN` | – | **0.484** |
| Top MI | `LOG_VAR_MINOR_DIV_MAXLEN` | – | **0.475** |

*Univariate AUC values cluster around 0.50‑0.51, indicating that no single descriptor is highly discriminative on its own, but collectively they provide strong signal.*

---

### 4. Model‑based importance (Permutation)

| Rank | Feature | Mean permutation importance |
|------|---------|-----------------------------|
| 1 | `COMPACTNESS_MEAN` | 0.0177 |
| 2 | `RECIPROCAL_HOLLOWS_RATIO_MEAN` | 0.0122 |
| 3 | `INTER_SRG_VAR_MINOR` | 0.0098 |
| 4 | `INTER_SRG_HOLLOWS` | 0.0091 |
| 5 | `INTER_SKEW_RATIO_HOLLOWS_RATIO` | 0.0079 |
| … | … | … |
| 20 | `SQRT_ELONGATEDNESS_MEAN` | 0.0051 |

Permutation importance highlights a small set of descriptors (compactness‑related, hollows‑related, and several interaction ratios) that most affect the classifier’s predictions.

---

### 5. Correlation & redundancy analysis  

*Pearson correlation matrix revealed **11 pairs** with perfect correlation (|ρ| = 1.0). Example:*

| Perfectly correlated pair |
|----------------------------|
| `INTER_VAR_RATIO_MAXLEN_ASPECT` ↔ `INTER_VAR_RATIO_MAJOR_OVER_MINOR_MAXLEN_ASPECT` |
| `SCALED_VARIANCE_MAJOR_MEAN` ↔ `INTER_VAR_RATIO_VAR_MINOR` |
| `VAR_DIFF_SCALED_VARIANCE` ↔ `VAR_DIFF_MAJOR_MINUS_MINOR` |
| … (total 11 pairs) |

These pairs represent duplicated information (often a raw metric and a derived ratio of the same quantities).

---

### 6. Pruning of redundant features  

Using permutation importance to keep the more influential member of each pair, **11 attributes** were removed:

```
SCALED_VARIANCE_MAJOR_MEAN,
INTER_VAR_RATIO_MAJOR_OVER_MINOR_VAR_MINOR,
PRODUCT_MAXLEN_ASPECT_SCALED_VAR_MINOR,
VAR_DIFF_SCALED_VARIANCE,
VAR_RATIO_SCALED_VARIANCE,
INTER_VAR_DIFF_MAXLEN_ASPECT,
INTER_VAR_RATIO_MAXLEN_ASPECT,
INTER_VAR_DIFF_MAJOR_MINUS_MINOR_VAR_MINOR,
INTER_VAR_RATIO_MAJOR_OVER_MINOR_VAR_MINOR,
INTER_VAR_DIFF_MAJOR_MINUS_MINOR_HOLLOWS,
INTER_VAR_RATIO_MAJOR_OVER_MINOR_HOLLOWS
```

*Resulting feature count:* **99** (down from 110).

---

### 7. Post‑pruning performance  

| Metric | Value (after pruning) |
|--------|-----------------------|
| Accuracy | **0.732** (‑0.4 % vs. baseline) |
| Macro‑AUC | **0.916** (‑0.001 vs. baseline) |

The negligible drop confirms that the removed attributes contributed only redundant information.

---

### 8. Robustness observations  

* **Permutation importance** already measures sensitivity to feature shuffling, effectively testing robustness to noise.  
* The top‑20 important features retained their ranking after pruning, indicating stable importance.  
* Adding Gaussian noise (σ ≈ 0.1 × feature std) to the most important features in a quick sanity check reduced accuracy by < 2 %, confirming that the model’s predictions are not overly fragile to small perturbations.

---

### 9. Key take‑aways  

1. **Predictive power** is distributed across many shape descriptors; no single feature dominates, but a handful (compactness, hollows‑related ratios, and variance‑based interactions) drive most of the model’s performance.  
2. **Feature importance** (permutation) aligns with domain intuition: compactness and hollows capture shape irregularities that differentiate vehicle classes.  
3. **Redundancy** is substantial: 11 perfectly correlated pairs inflated the feature set without adding information. Pruning them reduced dimensionality by ~10 % with virtually no performance loss.  
4. **Stability** of the important features across perturbations and across the original vs. pruned models indicates a robust feature set.  

---

### 10. Recommended next step for the team  

* **Adopt the pruned feature list (99 attributes)** for downstream modelling to simplify training, reduce over‑fitting risk, and speed up computation.  
* **Focus further investigation** (Scientist & Extractor agents) on the top‑20 permutation‑important attributes and the high‑MI univariate features, as they consistently show the strongest relationship with the target.  

*All observations have been recorded via the note‑taking tool for reference.*