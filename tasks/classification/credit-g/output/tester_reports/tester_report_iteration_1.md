**Comprehensive Feature Evaluation Report – Credit‑G Dataset**

---

### 1. Experimental Setup
- **Model:** `GradientBoostingClassifier` (sklearn) – chosen for reliable performance without GPU dependencies.  
- **Target Encoding:** `yes` → 1, `no` → 0.  
- **Train/Test Split:** 80 % train / 20 % test, stratified, `random_state=42`.  
- **Metrics Reported:** Accuracy, ROC‑AUC.  
- **Tools Used:** Python executor, note‑taking, attribute pruning.

---

### 2. Baseline Performance (All 9 Features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.75** |
| **ROC‑AUC** | **0.7204** |

**Feature Importance (gain)**
| Rank | Feature | Relative Importance |
|------|---------|----------------------|
| 1 | `total_credit_exposure` | 33.8 % |
| 2 | `credit_amount_per_month` | 22.6 % |
| 3 | `checking_status_score` | 22.3 % |
| 4 | `property_magnitude_score` | 6.1 % |
| 5 | `credit_history_score` | 5.5 % |
| 6 | `employment_years_score` | 4.0 % |
| 7 | `residence_years` | 3.3 % |
| 8 | `savings_status_score` | 2.2 % |
| 9 | `high_installment_flag` | **0.4 %** (negligible) |

**Inter‑Feature Correlation**
- No pair of features exhibited absolute Pearson correlation ≥ 0.5.  
- Features are largely independent, reducing redundancy concerns.

---

### 3. Impact of Pruning the Least‑Important Feature
- **Action:** Removed `high_installment_flag` (importance < 0.5 %).  
- **Result:**  
  - Accuracy **↑ 0.755** (↑ 0.5 % points)  
  - ROC‑AUC **↑ 0.728** (↑ 0.8 % points)  

**Interpretation:** The flag contributes virtually nothing and its removal slightly improves model generalisation.

---

### 4. Robustness Check – Noise Injection
- **Procedure:** Added Gaussian noise (σ = 0.1 × std) to the most important feature `total_credit_exposure`.  
- **Outcome:**  
  - Accuracy **≈ 0.75** (unchanged)  
  - ROC‑AUC **≈ 0.725** (minor drop)  

**Interpretation:** Model performance is reasonably stable to modest perturbations of the dominant feature, indicating robustness.

---

### 5. Summary & Recommendations for Feature Set
- **Effective Predictors:** `total_credit_exposure`, `credit_amount_per_month`, `checking_status_score` (together explain ~78 % of importance).  
- **Redundant/Irrelevant:** `high_installment_flag` – **prune**.  
- **Correlation:** Low inter‑feature correlation; no need for further dimensionality reduction due to multicollinearity.  
- **Robustness:** Model tolerates realistic noise on key features, suggesting reliable deployment.

---

### 6. Action Taken
- **Pruned Feature:** `high_installment_flag` (using `attribute_pruning_tool`).  

---  

**Final Note:** The retained nine‑minus‑one feature set provides solid predictive power (≈ 75 % accuracy, 0.73 AUC) with a compact, non‑redundant representation suitable for downstream modeling pipelines.