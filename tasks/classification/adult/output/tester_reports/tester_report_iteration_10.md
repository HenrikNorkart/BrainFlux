**Comprehensive Feature‑Evaluation Report**

---

### 1. Objective  
Assess the predictive utility of the 86 engineered attributes supplied in **df_attributes** for the binary classification target *‘target’* (yes / no).

---

### 2. Methodology  

| Step | Description |
|------|-------------|
| **Data split** | Stratified 80/20 train‑test split (random_state = 42). |
| **Model** | XGBoost (binary:logistic) – `device="cuda:5"`, `tree_method="hist"`, 200 trees, max_depth = 6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8. |
| **Primary metrics** | ROC‑AUC (primary) and Accuracy (secondary). |
| **Feature importance** | XGBoost *gain* scores (the contribution of each split to the objective). |
| **Correlation analysis** | Pearson |r| on the top‑20 gain features; pairs with |r| > 0.9 flagged as redundant. |
| **Pruning rule** | For each highly correlated pair, the feature with the **lower gain** was marked for removal. |
| **Robustness test** | Added Gaussian noise (σ = 0.1 × feature std) to all numeric attributes and re‑evaluated AUC. |

All code was executed via the provided `generic_python_executor_tool`; notes were captured with `take_note_tool`; redundant attributes were removed using `attribute_pruning_tool`.

---

### 3. Results  

| Metric | Using **all 86** features | After **pruning 8** redundant features (77 remain) |
|--------|---------------------------|---------------------------------------------------|
| ROC‑AUC | **0.9307** | **0.9306** (Δ = ‑0.0001) |
| Accuracy | **0.8747** | **0.8755** (Δ = +0.0008) |
| Feature count | 86 | **77** |

**Top‑10 gain features (pre‑pruning)**  

| Rank | Feature | Gain |
|------|---------|------|
| 1 | relationship_edu_hours_age_interaction | 145.56 |
| 2 | high_education_binary | 140.27 |
| 3 | cap_gain_to_loss_ratio | 56.39 |
| 4 | log_capital_gain | 53.88 |
| 5 | net_capital | 50.64 |
| 6 | high_occupation_binary | 40.43 |
| 7 | race_occ_interaction | 38.73 |
| 8 | occ_cap_gain_interaction | 28.64 |
| 9 | is_married | 24.88 |
|10 | wealth_idx_adj_squared | 24.04 |

**High‑correlation pairs (|r| > 0.9) among the top‑20**  

| Feature A | Feature B | Correlation | Higher‑gain kept |
|-----------|-----------|-------------|-------------------|
| cap_gain_to_loss_ratio | net_capital | 0.999 | **cap_gain_to_loss_ratio** |
| cap_gain_to_loss_ratio | occ_cap_gain_interaction | 0.903 | **cap_gain_to_loss_ratio** |
| cap_gain_to_loss_ratio | occ_capgain_interaction | 0.946 | **cap_gain_to_loss_ratio** |
| log_capital_gain | wealth_idx_adj_squared | 0.987 | **log_capital_gain** |
| log_capital_gain | native_country_wealth_adj_interaction | 0.921 | **log_capital_gain** |
| log_capital_gain | wealth_idx_adj | 0.965 | **log_capital_gain** |
| net_capital | occ_cap_gain_interaction | 0.902 | **cap_gain_to_loss_ratio** (kept) |
| net_capital | occ_capgain_interaction | 0.945 | **cap_gain_to_loss_ratio** |
| high_occupation_binary | race_occ_interaction | 0.922 | **high_occupation_binary** |
| is_married | relationship_husband_wife | 0.960 | **is_married** |
| wealth_idx_adj_squared | wealth_idx_adj | 0.927 | **log_capital_gain** (kept) |
| native_country_wealth_adj_interaction | wealth_idx_adj | 0.954 | **log_capital_gain** |

**Features pruned (8 total)**  

```
race_occ_interaction,
relationship_husband_wife,
native_country_wealth_adj_interaction,
occ_capgain_interaction,
net_capital,
wealth_idx_adj_squared,
occ_cap_gain_interaction,
wealth_idx_adj
```

These were the lower‑gain members of each highly correlated pair.

**Robustness test (Gaussian noise, σ = 0.1 × std)**  

| Condition | ROC‑AUC |
|-----------|----------|
| Original (77 features) | 0.9306 |
| With added noise | **0.9140** |

A modest drop (≈ 1.6 % absolute) indicates the model remains reasonably robust to moderate perturbations of the input features.

---

### 4. Interpretation  

* **Predictive power** – The engineered feature set yields strong discrimination (AUC ≈ 0.93).  
* **Redundancy** – Several features are near‑perfect linear combinations; removing the lower‑gain members does **not** degrade performance, confirming that the information they carry is already captured by higher‑gain counterparts.  
* **Feature importance** – The most influential attributes combine demographic information (e.g., `high_education_binary`, `is_married`) with engineered financial ratios (`cap_gain_to_loss_ratio`, `log_capital_gain`). Interaction terms involving age, education, and relationship status also rank highly.  
* **Robustness** – Adding realistic noise reduces AUC only slightly, suggesting the model’s decisions are not overly sensitive to small measurement errors.  

---

### 5. Outcome  

* **Final feature set:** 77 attributes (the original 86 minus the 8 redundant ones).  
* **Model performance:** AUC = 0.9306, Accuracy = 0.8755 – essentially identical to the full‑feature baseline.  
* **Recommendation:** Use the pruned 77‑feature set for downstream modeling; it is more parsimonious without sacrificing predictive quality.

--- 

*All observations were recorded via `take_note_tool`; redundant attributes were removed with `attribute_pruning_tool` as described.*