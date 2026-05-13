**Tester Agent – Feature Evaluation Report**

---

### 1. Experimental Setup  

| Item | Detail |
|------|--------|
| **Target** | `target` (three‑class: *low*, *medium*, *high*) |
| **Data** | 1,994 instances, 132 numeric attributes (all columns except `target`) |
| **Model** | XGBoost (multiclass, `objective=multi:softprob`) |
| **Training‑test split** | Stratified 80 % / 20 % (random_state = 42) |
| **XGBoost parameters** | `tree_method='hist'`, `device='cuda:5'`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping = 20 rounds |
| **Evaluation metrics** | Overall accuracy, macro‑averaged F1‑score |
| **Feature‑importance** | XGBoost gain importance (`bst.get_score(importance_type='gain')`) |

---

### 2. Baseline Results (All 132 features)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.7143** |
| **Macro F1** | **0.7145** |

The model already provides a respectable baseline for the three‑class crime‑rate prediction task.

---

### 3. Feature‑Importance Findings  

The top‑gain features (first 10) were:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `PctIlleg_raw` | 18.97 |
| 2 | `PctKids2Par_raw` | 16.58 |
| 3 | `Illegitimacy_racepctblack_interaction_raw` | 10.17 |
| 4 | `racePctWhite_raw` | 5.62 |
| 5 | `racepctblack_PctNotSpeakEnglWell_interaction` | 4.89 |
| 6 | `NumIlleg_raw` | 4.85 |
| 7 | `PctHousNoPhone_racepctblack_interaction` | 4.07 |
| 8 | `racePctHisp_raw` | 3.41 |
| 9 | `TotalDivorce_poverty_interaction_raw` | 3.39 |
|10 | `racepctblack_NumStreet_interaction` | 3.36 |

**Key patterns**

* **Family structure & illegitimacy** (`PctIlleg_raw`, `PctKids2Par_raw`, `NumIlleg_raw`) dominate importance.
* **Race‑related interactions** (e.g., `Illegitimacy_racepctblack_interaction_raw`, `racepctblack_*` interactions) consistently rank high.
* **Economic‑poverty interactions** (`TotalDivorce_poverty_interaction_raw`, `rentBurden_pctPov_interaction`) also contribute substantially.
* Pure demographic percentages (e.g., `racePctWhite_raw`) still matter but are less dominant than interaction terms.

---

### 4. Pruning Experiment  

**Strategy** – Keep the **30 highest‑gain features** (cumulative ≈ 95 % of total gain) and discard the remaining 102 attributes.

**Top‑30 retained features** (full list in the notebook):  

```
PctIlleg_raw, PctKids2Par_raw, Illegitimacy_racepctblack_interaction_raw,
racePctWhite_raw, racepctblack_PctNotSpeakEnglWell_interaction,
NumIlleg_raw, PctHousNoPhone_racepctblack_interaction,
racePctHisp_raw, TotalDivorce_poverty_interaction_raw,
racepctblack_NumStreet_interaction, rentBurden_pctPov_interaction,
race_poverty_interaction_raw, WorkMom_poverty_interaction_raw,
TotalPctDiv_raw, RentMedian_pctPov_interaction,
FemalePctDiv_raw, numStreet_pctPov_interaction,
MedRent_racepctblack_interaction, TotalDivorce_racepctblack_interaction_raw,
MedRent_pctPov_interaction, noPhone_racepctblack_interaction_raw,
WorkMomYoungKids_poverty_interaction_raw, NumStreet_raw,
PctWOFullPlumb_racepctblack_interaction, PctLess9thGrade_raw,
medIncome_raw, poverty_urban_interaction_raw,
racepctblack_PctImmigRec5_interaction, less3BR_racePctWhite_interaction,
PctVacantBoarded_racepctblack_interaction
```

**Post‑pruning performance**

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.7143** (unchanged) |
| **Macro F1** | **0.7176** (slight improvement) |

The reduction from 132 to 30 features **did not degrade predictive power**; macro‑F1 even improved marginally, indicating that the pruned attributes were largely redundant or noisy.

---

### 5. Robustness Check (Noise Injection)  

A quick sanity test added Gaussian noise (σ = 0.01 of each feature’s std) to the test set. The model’s accuracy dropped by **< 0.5 %**, confirming stability against minor perturbations.

---

### 6. Conclusions  

| Observation | Implication |
|-------------|-------------|
| Baseline XGBoost with all features: 0.714 accuracy, 0.714 macro‑F1 | Reasonable starting point. |
| Top‑gain features are heavily centered on **family structure, illegitimacy, race‑poverty interactions**. | These attributes capture the strongest signals for community crime level. |
| Pruning to the **30 most important features** retains (and slightly improves) performance while cutting dimensionality by **≈ 77 %**. | A leaner model is faster, easier to interpret, and less prone to over‑fitting. |
| Model is **robust** to small random noise. | Feature set is stable. |

**Recommended next step for the team:**  
- The Scientist Agent can focus hypothesis generation on the retained 30 attributes (especially the interaction terms) to deepen domain understanding.  
- The Extractor Agent may concentrate future extraction efforts on similar interaction‑style features, as they prove highly predictive.

--- 

*All notes captured via `take_note_tool` and the attribute list was pruned using `attribute_pruning_tool` as instructed.*