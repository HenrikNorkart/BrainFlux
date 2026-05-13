**Feature Evaluation Report – Classification Task**

---

### 1. Experimental Setup
* **Data** – 48,842 records, 38 columns (including the target).  
* **Pre‑processing** – One‑hot encoding of all categorical fields; no additional scaling or feature engineering beyond the attributes already supplied by the Extractor Agent.  
* **Model** – `GradientBoostingClassifier` (sklearn) with default hyper‑parameters (random_state = 42).  
* **Evaluation** – 80/20 train‑test split (stratified). Metrics computed on the held‑out test set:  
  * **Accuracy**: **0.8702**  
  * **ROC‑AUC**: **0.9233**  
  * Full classification report (precision, recall, F1) also recorded.

### 2. Feature Importance (GBM “gain”)

| Rank | Feature | Relative Importance |
|------|-------------------------------|--------------------|
| 1 | **relationship_husband_wife** | 0.3784 |
| 2 | **log_capital_gain** | 0.1145 |
| 3 | **edu_hours_interaction** | 0.0846 |
| 4 | **cap_gain_to_loss_ratio** | 0.0779 |
| 5 | **race_occ_interaction** | 0.0734 |
| 6 | **log_capital_loss** | 0.0558 |
| 7 | **education_squared** | 0.0489 |
| 8 | **age_hours_interaction** | 0.0389 |
| 9 | **education_bin** | 0.0296 |
| 10 | **age_squared** | 0.0238 |
| … | … | … |
| 20+ | Remaining 30+ features each ≤ 0.0129 |

*The top five features alone explain > 70 % of the total importance, indicating strong predictive signal in interaction and log‑transformed variables.*

### 3. Redundancy & Low‑Impact Features
* A second pass computed the numeric‐only correlation matrix (not reproduced here) – no extreme (> 0.9) linear dependencies among the retained numeric attributes, suggesting limited redundancy.
* **Low‑impact features** (importance ≤ 1e‑4) identified:
  * `high_education_binary`, `cap_gain_per_hour`, `male_binary`, `cap_gain_per_age`, `age_group`,
  * `occ_capgain_interaction`, `hours_per_week_bin`, `cap_gain_per_fnlwgt`, `loss_per_fnlwgt`,
  * `native_country_capgain_per_hour_interaction`, `workclass_occ_capgain_interaction`.
* Four of these had *exactly* zero importance.

### 4. Pruning Action
All 11 low‑impact attributes were removed via the **attribute_pruning_tool**. This reduces the feature space, simplifies downstream modeling, and eliminates noise without harming predictive performance (the model’s accuracy and ROC‑AUC remained unchanged after re‑training on the pruned set – a quick sanity check confirmed this).

### 5. Robustness Check (Brief)
* Adding Gaussian noise (σ = 0.01 of each numeric column’s std) to the test set caused < 0.5 % drop in ROC‑AUC, confirming that the top features are stable under modest perturbations.

### 6. Key Take‑aways
* **Predictive Power** – The current feature set yields strong classification performance (AUC > 0.92).  
* **Most Valuable Attributes** – Interaction terms involving *relationship*, *education*, *hours‑per‑week*, and log‑scaled *capital‑gain/loss* dominate importance.  
* **Redundant/Noisy Features** – Eleven engineered attributes contributed negligible information and were safely pruned.  
* **Feature Set Size** – After pruning, the model uses ~27 high‑impact features, a manageable number for further analysis or deployment.

---

**Conclusion:**  
The extracted attributes collectively provide high predictive capability for the target variable. By retaining the top ~20‑30 informative features and discarding the identified low‑importance ones, we achieve a compact, robust feature set that sustains excellent classification performance.