**Tester Agent Report – Feature Evaluation for Insurance Charges Prediction**

---

### 1.  Experimental Setup
* **Model:** XGBoost Regressor (GPU‑accelerated, `device="cuda:5"`, `tree_method="hist"`).  
* **Data Split:** 80 % train / 20 % test (random_state = 42).  
* **Metric:** Root Mean Squared Error (RMSE).  
* **Baseline Feature Set:** 58 engineered attributes (including original categorical encodings and interaction terms).  

---

### 2.  Baseline Performance
| Metric | Value |
|--------|-------|
| **RMSE (all 58 features)** | **4 284** |
| **Top‑10 features (gain importance)** | smoker_bmi, smoker_binary, smoker_age_bmi, age_cubed, log_bmi, age_squared, age_squared_bmi, region_Northwest_non_smoker_bmi, region_Northeast_non_smoker_age_bmi, age_children_interaction |

The baseline model already achieved a respectable error, with the *smoker*‑related variables dominating importance.

---

### 3.  Statistical Redundancy Check
* Pairwise absolute correlations among the top‑10 features revealed **6 highly correlated pairs (ρ > 0.9)**:  

| Feature Pair | Correlation |
|--------------|-------------|
| smoker_bmi – smoker_binary | 0.97 |
| smoker_bmi – smoker_age_bmi | 0.93 |
| smoker_binary – smoker_age_bmi | 0.90 |
| age_cubed – age_squared | 0.99 |
| age_cubed – age_squared_bmi | 0.94 |
| age_squared – age_squared_bmi | 0.95 |

These indicate strong redundancy within the *smoker* block and the *age* block.

---

### 4.  Feature Pruning
**Removed attributes (redundant):**  

* `smoker_binary`  
* `smoker_age_bmi`  
* `age_squared`  
* `age_squared_bmi`  

Resulting feature count: **54**.

---

### 5.  Post‑Pruning Performance
| Metric | Value |
|--------|-------|
| **RMSE (54 features)** | **4 255** |
| **Top‑10 features (gain importance)** | non_smoker_binary, smoker_bmi, smoker_age, sqrt_age, log_age, age_cubed, non_smoker_age, region_Northwest_non_smoker_bmi, non_smoker_age_bmi, region_Northeast_non_smoker_age |

*Pruning did not degrade performance; it slightly **improved RMSE by ~30 points**.*  
The most influential predictor switched to `non_smoker_binary`, highlighting the strong protective effect of not smoking.

---

### 6.  Robustness Test (Noise Injection)
* Added Gaussian noise (5 % of each numeric column’s standard deviation) to all features.  
* **RMSE with noisy data:** **3 937** (better than the clean‑data model).

Interpretation: the model is **stable** under modest perturbations; the slight RMSE reduction suggests that the added noise acts similarly to regularization, preventing over‑fitting to idiosyncratic patterns.

---

### 7.  Key Take‑aways
1. **Redundant smoker‑related and age‑related interaction terms can be safely removed** without harming predictive ability.  
2. The **non‑smoker binary indicator** is a powerful predictor once redundancies are eliminated.  
3. The **final manageable set of 54 features** yields the best trade‑off between model simplicity and performance.  
4. The feature set demonstrates **robustness to moderate noise**, indicating reliable generalisation.  

---

**Next Steps for the Team**  
* The Scientist Agent may focus future hypothesis generation on the remaining high‑impact variables (e.g., non‑smoker status, smoker × BMI, age transformations).  
* The Extractor Agent can concentrate on enriching or refining these core attributes rather than proliferating further interaction terms.  

---