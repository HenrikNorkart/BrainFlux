**Feature‑Testing Report – Classification Target “target”**

---

### 1. Experimental Setup
| Step | Details |
|------|---------|
| **Model** | `GradientBoostingClassifier` (200 trees, learning‑rate 0.1, max‑depth 3) – chosen to avoid XGBoost GPU‑dependency while still providing robust non‑linear learning. |
| **Train/Test Split** | 80 % / 20 % stratified split, random_state 42. |
| **Target Encoding** | `LabelEncoder` → *yes* = 1, *no* = 0. |
| **Evaluation Metrics** | Accuracy, ROC‑AUC. |
| **Feature‑Importance** | `model.feature_importances_` (gain‑like importance). |
| **Pruning Criterion** | Features with importance < 0.001 were deemed non‑contributory. |

---

### 2. Baseline Performance (All 69 engineered attributes)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.875** |
| **ROC‑AUC** | **0.928** |
| **Number of Features** | 69 |

**Top‑10 predictive attributes (gain importance)**  

| Rank | Feature | Relative Importance |
|------|---------|----------------------|
| 1 | `relationship_edu_hours_age_interaction` | 0.505 |
| 2 | `net_capital` | 0.059 |
| 3 | `cap_gain_to_loss_ratio` | 0.038 |
| 4 | `education_squared` | 0.036 |
| 5 | `race_occ_interaction` | 0.032 |
| 6 | `log_capital_loss` | 0.028 |
| 7 | `wealth_idx_squared` | 0.027 |
| 8 | `log_wealth_idx` | 0.024 |
| 9 | `wealth_idx` | 0.022 |
|10 | `fnlwgt_high_education_interaction` | 0.022 |

**Low‑importance attributes (importance < 0.001)** – 36 features (e.g., `gender_marital_code`, `occupation_code`, `test_constant`, `high_education_binary`, `wealth_idx_bin`, many interaction terms).  

These were recorded in the internal notes.

---

### 3. Feature Pruning
All 36 low‑importance attributes were removed via **`attribute_pruning_tool`**, reducing the dataset to **34 features**.

---

### 4. Post‑Pruning Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.874** (Δ − 0.001) |
| **ROC‑AUC** | **0.928** (Δ ≈ 0.000) |
| **Number of Features** | 34 |

**Top‑10 after pruning** (very similar ranking):

| Rank | Feature | Relative Importance |
|------|---------|----------------------|
| 1 | `relationship_edu_hours_age_interaction` | 0.507 |
| 2 | `race_occ_interaction` | 0.052 |
| 3 | `net_capital` | 0.047 |
| 4 | `cap_gain_to_loss_ratio` | 0.039 |
| 5 | `fnlwgt_high_education_interaction` | 0.032 |
| 6 | `log_capital_loss` | 0.030 |
| 7 | `log_capital_gain` | 0.029 |
| 8 | `education_squared` | 0.026 |
| 9 | `log_wealth_idx` | 0.025 |
|10 | `wealth_idx_squared` | 0.023 |

*Performance impact*: pruning caused **negligible loss** (‑0.1 % accuracy, unchanged AUC) while halving the feature count, simplifying the model and reducing computational load.

---

### 5. Statistical Observations
* **Dominant predictor** – `relationship_edu_hours_age_interaction` alone accounts for ~50 % of the model’s importance, indicating a strong interaction between marital relationship, education‑hours, and age.
* **Economic signals** – `net_capital`, `cap_gain_to_loss_ratio`, and `wealth_idx`‑related features collectively contribute ~15 % importance, confirming that capital‑gain/loss dynamics are highly discriminative.
* **Demographic encodings** – `race_occ_interaction` and `education_squared` also provide meaningful signal, whereas many one‑hot or frequency‑encoded categorical variables (e.g., `race_freq_enc`, `native_country_freq_enc`) showed near‑zero importance and were safely removed.

---

### 6. Recommendations for the Team
1. **Keep** the 34 retained attributes – they deliver strong predictive power with a compact model.
2. **Discard** the 36 pruned attributes – they add no measurable benefit and increase dimensionality.
3. **Future work** (outside the scope of this testing) could explore whether any of the removed interaction terms become useful after alternative model families or regularisation, but current evidence suggests they are redundant.

---

*Report compiled by the Tester Agent after systematic evaluation, feature‑importance analysis, and robust pruning.*