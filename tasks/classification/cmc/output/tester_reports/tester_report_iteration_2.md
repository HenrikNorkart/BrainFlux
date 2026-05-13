**Tester Agent Report – Feature Evaluation for Contraceptive_method_used**

---

### 1. Experimental Setup  

| Step | Methodology | Rationale |
|------|-------------|-----------|
| **Data split** | `train_test_split(test_size=0.20, stratify=target, random_state=42)` | Guarantees a representative hold‑out set for a multi‑class problem. |
| **Model** | **RandomForestClassifier** (n_estimators = 300, n_jobs = ‑1, random_state = 42) | Tree‑based ensemble provides built‑in feature‑importance (Gini impurity) and works well with mixed categorical/ordinal features without heavy preprocessing. |
| **Metrics** | Accuracy, per‑class classification report, feature‑importance (Gini) | Accuracy is the primary task metric; the report gives insight into class‑wise performance. |
| **Robustness test** | Added Gaussian noise (μ=0, σ=1) to the most important feature (`Parity_category`) and re‑trained the model. | Checks sensitivity of the model to perturbations of a key predictor. |
| **Pruning** | Removed four lowest‑importance attributes (`Husband_high_Wife_low`, `Wife_high_Husband_low`, `Both_High_Education`, `Both_midhigh_or_higher`). | To keep the feature set manageable while preserving predictive power. |

All code was executed via the `generic_python_executor_tool`; notes were captured with `take_note_tool`.

---

### 2. Baseline Model Performance (All 14 features)

```
Accuracy: 0.4983
Top 5 features by Gini importance:
  Parity_category                0.2154
  Wifes_age_group                0.1748
  Socioeco_detailed_score        0.1056
  Socioeconomic_score            0.0905
  Religion_Work_interaction      0.0586
```

*The baseline accuracy (~49.8 %) is well above random guessing (≈33 % for three classes) and demonstrates that the engineered features carry useful signal.*

---

### 3. Full Feature‑Importance Ranking  

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **Parity_category** | 0.2152 |
| 2 | **Wifes_age_group** | 0.1768 |
| 3 | **Socioeco_detailed_score** | 0.1054 |
| 4 | **Socioeconomic_score** | 0.0900 |
| 5 | **Religion_Work_interaction** | 0.0585 |
| 6 | Occupation_HusbandEdu_interaction | 0.0562 |
| 7 | Religion_Edu_Work_score | 0.0535 |
| 8 | Occupation_WifeEdu_interaction | 0.0527 |
| 9 | Combined_Edu_Occupation_sum | 0.0485 |
| 10 | Education_product | 0.0356 |
| 11 | Education_sum | 0.0312 |
| 12 | Education_media_interaction | 0.0281 |
| 13 | Education_disparity | 0.0219 |
| 14 | Both_High_Education | 0.0105 |
| 15 | Both_midhigh_or_higher | 0.0102 |
| 16 | Husband_high_Wife_low | 0.0045 |
| 17 | Wife_high_Husband_low | 0.0012 |

*The list shows a clear concentration of predictive power in the first 8–10 features; the last four contribute negligible variance.*

---

### 4. Feature Explanations (Top 5)

| Feature | Description (from `attribute_lookup_tool`) |
|---------|--------------------------------------------|
| **Parity_category** | “Parity influences contraceptive choice; higher parity often leads to greater contraceptive use.” |
| **Wifes_age_group** | “Age groups capture non‑linear relationship between a woman’s age and method choice.” |
| **Socioeconomic_score** | “A composite socioeconomic score aggregating education, occupation, and living‑standard indices.” |
| **Socioeco_detailed_score** | “Detailed socioeconomic indicator that combines education, occupation, and standard‑of‑living into a richer metric.” |
| **Religion_Work_interaction** | “Interaction captures combined effect of religious affiliation and employment status on contraceptive decisions.” |

These explanations align with known sociological drivers of contraceptive behavior.

---

### 5. Pruning Low‑Impact Features  

The following attributes were pruned (importance < 0.01):  

- `Husband_high_Wife_low`  
- `Wife_high_Husband_low`  
- `Both_High_Education`  
- `Both_midhigh_or_higher`

**Result after pruning:**  

```
Accuracy: 0.4949
```

*The drop of ~0.3 % points is negligible, confirming that the removed features add little predictive value.*

---

### 6. Robustness Check – Noise Injection  

Gaussian noise added to the most important feature (`Parity_category`) reduced accuracy to **0.4237** (≈7.5 % absolute loss).  

*Interpretation:* The model is fairly sensitive to perturbations of `Parity_category`, underscoring its central role. Other features (e.g., `Wifes_age_group`) still retain predictive contribution, but the experiment highlights the importance of data quality for the top predictor.

---

### 7. Conclusions  

| Aspect | Finding |
|--------|----------|
| **Predictive Power** | The engineered feature set yields ~50 % accuracy, substantially better than random. |
| **Key Drivers** | `Parity_category`, `Wifes_age_group`, and socioeconomic scores dominate model performance. |
| **Redundant Features** | Four low‑importance attributes can be safely removed without harming accuracy. |
| **Feature Interactions** | Interaction terms (e.g., `Religion_Work_interaction`, `Occupation_HusbandEdu_interaction`) contribute meaningfully, confirming the value of engineered cross‑features. |
| **Robustness** | Model performance degrades when the strongest predictor is corrupted, indicating a reliance on high‑quality parity data. |

**Recommendation for the team:** Focus future investigative efforts on refining parity‑related measurements, exploring richer age‑group encodings, and possibly augmenting socioeconomic composites. The current feature set is already concise and effective after pruning.