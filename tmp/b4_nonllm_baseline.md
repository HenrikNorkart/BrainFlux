# B4 — Non-LLM feature-engineering baseline

Nightshift session, 2026-09-11/12. HQ to-do #6. Branch `fix/threshold-train-optimization-v2`.
CPU only, no GPU, no LLM calls.

## Verdict (one sentence)

**The LLM loop wins decisively in the high-precision regime it is actually sold on
(pooled test recall 0.512 vs 0.197 for every non-LLM baseline at a 0.98 precision
floor) and loses in the relaxed regime (0.617 vs 0.688–0.702 at a 0.95 floor), so
the honest claim is "better where precision matters, not better in general".**

## 1. What was compared, and how the comparison was made fair

Reviewer 1's point 6 asks whether ordinary automated feature selection would have
found the same thing. To answer that, every arm must be fit against the same target,
on the same rows, and be scored by the same downstream procedure. That is what this
script enforces.

**Target.** All baselines are fit against the *phantom* target, not survival: the
`InvSqDistTroublemaker` indicator — 1 for the `top_p` fraction of non-target-class
patients lying closest to the fitted suppression-ratio decision boundary, 0 for
everyone else, including patients with no SR recording (which is what the
`.fillna(0)` in `phantom_menace/agents/tester.py` does). The boundary itself comes
from re-fitting `SingleLinearClassifier1D` on the same 1/1000 threshold grid at
`min_precision = 0.97`.

**Split.** `auth_split/train.csv` (1087) → `auth_split/test.csv` (725). This is the
split the published G1–G4 tree was fitted on, so both the LLM arm and the baselines
see exactly the same training patients and neither touches the test rows. It also
sidesteps the unresolved question in #106 — nothing here uses `data/train.csv` /
`data/test.csv`, so there is no cross-cutting partition to worry about.
Of those patients, 452 train / 313 test have a suppression-ratio recording and
therefore enter the downstream stage.

**Candidate inputs.** 42 raw EHR measurement means, one per measurement `name` in
the EHR extract that parses as numeric for ≥80% of its records and is present for
≥50% of the cohort, plus a forced carry-through of `Pain Score`, `Glasgow Coma
Score`, `Sedation Score`, `Oxygen % (FiO2)`, `FiO2 - vent`, `SpO2 - Ventilator RC`,
`O2 Saturation` and `Temperature`. The forced list matters: the baselines must be
*able* to rediscover the LLM's features, otherwise the comparison is rigged. No
interactions, trends or autocorrelations are handed to the baselines — those are
precisely what the LLM loop is claimed to have invented. Missing values are
median-imputed using train statistics only.

**Partitioning.** Identical for all three baselines: `DecisionTreeClassifier(
max_leaf_nodes=4)` on the selected feature subset, giving a 4-way partition
comparable to the published trimmed 4-leaf tree. Only the *feature selection* step
differs between arms.

**Downstream.** The pooled clean-split procedure from A3: per group, pick
`(threshold, direction)` on TRAIN maximising train recall subject to train precision
≥ floor; apply unchanged to TEST; union the predicted-positive sets; compute
precision/recall once over the union. A group whose floor is unreachable on train
contributes nothing (`groups_used` records how many contributed).

## 2. Pooled recall at matched precision floors

Primary table, `tm_frac = 0.3` (see §4 for the sweep). Non-survivor target
(label 0), test n = 313 (305 for the LLM arm, which drops 8 `Unassigned` patients).

| floor | method | groups used | test precision | **test recall** | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 1.00 | LLM-discovered (G1–G4) | 4 | 1.000 | **0.1244** | 25 | 0 | 176 |
| 1.00 | (a) shallow decision tree | 2 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 1.00 | (b) L1 logistic regression | 3 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 1.00 | (c) gradient boosting + SHAP | 2 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 1.00 | undivided cohort | 1 | 1.000 | 0.0385 | 8 | 0 | 200 |
| 0.99 | LLM-discovered (G1–G4) | 4 | 0.981 | **0.5124** | 103 | 2 | 98 |
| 0.99 | (a) shallow decision tree | 2 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 0.99 | (b) L1 logistic regression | 3 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 0.99 | (c) gradient boosting + SHAP | 2 | 1.000 | 0.0433 | 9 | 0 | 199 |
| 0.99 | undivided cohort | 1 | 1.000 | 0.0385 | 8 | 0 | 200 |
| **0.98** | **LLM-discovered (G1–G4)** | 4 | 0.981 | **0.5124** | 103 | 2 | 98 |
| **0.98** | (a) shallow decision tree | 2 | 1.000 | 0.1971 | 41 | 0 | 167 |
| **0.98** | (b) L1 logistic regression | 3 | 1.000 | 0.1971 | 41 | 0 | 167 |
| **0.98** | (c) gradient boosting + SHAP | 2 | 1.000 | 0.1971 | 41 | 0 | 167 |
| **0.98** | undivided cohort | 1 | 1.000 | 0.0385 | 8 | 0 | 200 |
| 0.95 | LLM-discovered (G1–G4) | 4 | 0.969 | 0.6169 | 124 | 4 | 77 |
| 0.95 | (a) shallow decision tree | 2 | 0.966 | **0.6875** | 143 | 5 | 65 |
| 0.95 | (b) L1 logistic regression | 3 | 0.966 | **0.6875** | 143 | 5 | 65 |
| 0.95 | (c) gradient boosting + SHAP | 2 | 0.966 | **0.6875** | 143 | 5 | 65 |
| 0.95 | undivided cohort | 1 | 0.963 | 0.5048 | 105 | 4 | 103 |

Read floor-by-floor, not row-by-row:

- **At floors 0.98–1.00 the LLM partition is 2.6–11.8× the recall of every baseline.**
  The gap at the 0.98 floor the manuscript operates at is 103 TP vs 41 TP — 62
  patients, far too large to be sampling noise at n = 313.
- **At the 0.95 floor the ordering reverses.** All three baselines reach 0.688 against
  the LLM's 0.617 at essentially the same realised precision (0.966 vs 0.969). That
  gap is 19 patients and is *not* clearly outside noise, but it is certainly not a
  win for the LLM.
- **Every arm beats the undivided cohort at the 0.98 floor** (0.197–0.512 vs 0.038),
  so "partitioning the cohort at all helps" is robust and is not an LLM-specific
  claim.
- Caveat on realised vs nominal precision: at the 0.95 floor nothing reaches 0.98
  realised precision, so the 0.95 rows are not a like-for-like substitute for the
  0.98 rows. The three baselines happen to produce the identical predicted-positive
  union at `tm_frac = 0.3` — their partitions differ but the pooled operating points
  coincide.

## 3. Which features each non-LLM method selected

Features that actually appear as splits in each method's 4-leaf partition:

| tm_frac | (a) shallow tree | (b) L1 logistic | (c) GBM + SHAP |
|---|---|---|---|
| 0.1 | Pain Score, RRT Peak Inspiratory Pressure, Total respiratory rate | Arterial Systolic Pressure, Basic vital signs comments | O2 Saturation, Pain Score |
| 0.3 | Glasgow Coma Score, Systolic BP, Total respiratory rate | Glasgow Coma Score, RRT Tidal Volume Exhaled | Glasgow Coma Score, Total respiratory rate, RRT Ventilator Type |
| 0.5 | Glasgow Coma Score, RRT Ventilator Type, Systolic BP | Glasgow Coma Score, RRT Ventilator Type, RRT Tidal Volume Exhaled | Glasgow Coma Score, RRT Ventilator Type, O2 Saturation |
| 0.7 | Glasgow Coma Score, O2 Saturation, RRT Ventilator Type | Glasgow Coma Score, RRT Ventilator Type, O2 Saturation | Glasgow Coma Score, RRT Ventilator Type, O2 Saturation |

**Does any baseline recover pain-score × GCS, sedation, or FiO2?** Stated plainly:

- **pain-score × GCS — no.** `Glasgow Coma Score` is selected by all three methods at
  `tm_frac` ≥ 0.3, and `Pain Score` is selected by two of them at `tm_frac = 0.1`, but
  **no baseline ever selects both simultaneously, and none can form their product** —
  the interaction is not in the candidate set and a 4-leaf axis-aligned tree cannot
  synthesise it. The two ingredients are individually discoverable; the interaction is
  not. That is the narrowest defensible statement of what the LLM added.
- **Sedation Score — no.** Never selected by any method at any `tm_frac`, despite
  being in the candidate set with 99% coverage.
- **FiO2 — no.** Neither `Oxygen % (FiO2)` nor `FiO2 - vent` is ever selected, again
  despite being in the candidate set with 98–99% coverage.

The baselines instead lean on ventilator plumbing (`RRT Ventilator Type`,
`RRT Tidal Volume Exhaled`, `Total respiratory rate`) and blood pressure, which is
weak evidence that the LLM's pick is the clinically more plausible one — but it is
evidence about *interpretability*, not about performance, and the two should not be
conflated in the write-up.

## 4. `tm_frac` sensitivity, and the one assumption this work rests on

The published tree run (`output/feature_tester_runs/20260224_090507/RESULTS_SUMMARY.md`)
records the precision floor (97%) but **not** the `tm_frac` the troublemaker set was
drawn at, and the code that produced that run is not in the repo. `is_troublemaker`
in `auth_split/*_patient_scores.csv` is the *tree's own prediction* at proba ≥ 0.40,
not the fitting target, so it cannot be used to back out `tm_frac` either.

The whole sweep from `phantom_menace_main.py` (`tm_frac ∈ {0.1, 0.3, 0.5, 0.7}`) was
therefore run. The LLM arm is the fixed published partition and does not move with
`tm_frac`; only the baselines refit.

Pooled test recall at the 0.98 floor:

| tm_frac | phantom positives (of 452 train) | LLM | (a) tree | (b) L1 | (c) GBM+SHAP | undivided |
|---|---|---|---|---|---|---|
| 0.1 | 15 | **0.5124** | 0.1875 | 0.1827 *(prec 0.826)* | 0.2788 | 0.0385 |
| 0.3 | 45 | **0.5124** | 0.1971 | 0.1971 | 0.1971 | 0.0385 |
| 0.5 | 75 | **0.5124** | 0.0385 | 0.0385 | 0.0433 *(prec 0.692)* | 0.0385 |
| 0.7 | 105 | **0.5124** | 0.0385 | 0.0385 | 0.0385 | 0.0385 |

**The LLM arm beats all three baselines at the 0.98 floor at every `tm_frac`**, so the
headline comparison does not depend on the unknown setting. At the 0.95 floor the
baselines beat the LLM arm at `tm_frac` ≥ 0.3 (0.688–0.702 vs 0.617) and lose at
`tm_frac = 0.1` (tree 0.567, GBM+SHAP 0.538, L1 0.183, vs LLM 0.617). The reversal at
the relaxed floor is therefore
`tm_frac`-dependent; the win at the strict floor is not.

Note also that at `tm_frac` ≥ 0.5 the baselines collapse to roughly the undivided
cohort: a phantom target that labels half the survivors as troublemakers carries too
little signal for a 4-leaf tree over raw columns to exploit.

## 5. Limitations that should go in the paper, not be papered over

1. **The arms are not perfectly symmetric.** The LLM arm is the *published* partition,
   fitted (by the original run, in a container, with code not in this repo) on 9
   engineered features including `temperature_trend`, `pain_score_autocorr1`, `age`
   and `peep_mean`, against a troublemaker target whose `tm_frac` is unknown. The
   baselines are refitted here on 42 raw columns against a reconstructed target. Both
   see the same 1087 training patients and neither sees the test rows, so the
   comparison is sound as an *end-to-end system* comparison; it is not a controlled
   ablation of "LLM vs no LLM" holding the feature vocabulary fixed.
2. **The LLM arm's advantage may be partly a feature-vocabulary advantage rather than
   an LLM advantage.** Handing the baselines `pain × gcs` as a candidate column would
   settle this. That is a distinct experiment and is filed as follow-up rather than
   silently folded in here.
3. **No confidence intervals.** n = 313 test, 208 non-survivors. The 0.98-floor gap
   (62 patients) is large; the 0.95-floor reversal (19 patients) is not obviously
   outside noise. B6 (#12) should attach intervals before either is asserted in text.
4. **Denominator mismatch.** The LLM arm scores 305 test patients, the baselines 313,
   because the published tree leaves 8 test patients `Unassigned`. Recall denominators
   differ by 7 non-survivors; this slightly flatters the LLM arm and should be stated.
5. **`age` is not in the EHR extract** under any of the 275 measurement names, so no
   baseline could use it even though the published 9-feature tree did.

## 6. What this means for the manuscript

- A claim of the form *"the LLM loop finds features standard AutoFE does not"* is
  **supported at the operating point the paper uses (P ≥ 0.98)** and should be stated
  with that qualifier attached, not as a general superiority claim.
- A claim of the form *"the discovered stratification is simply better"* is
  **contradicted** at the 0.95 floor, where all three baselines beat it. If the paper
  contains an unqualified version of that claim, it needs the floor condition added.
- The interpretability argument (sedation, FiO2 and the pain × GCS interaction are
  never recovered by any baseline) is the cleanest thing this experiment produces and
  is worth a sentence of its own.

## 7. Reproducing

```bash
cd filter-repository-python
python b4_build_features.py      # ~3 min, writes b4_cache/patient_table.csv
python b4_nonllm_baseline.py     # ~6 min
```

Needs `pandas`, `numpy`, `scikit-learn`, `shap`. Outputs `b4_cache/b4_results.csv`
(all 80 arm × floor × tm_frac cells) and `b4_cache/b4_selected_features.csv`
(selected features plus the printed tree rules per arm). CPU only; no GPU, no
network, no LLM calls. Seeded (`random_state=0`) and deterministic.
