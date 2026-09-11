# A3 — Clean train/test split for the reported case-study numbers

Nightshift session, 2026-09-11. HQ to-do #4. Branch `fix/threshold-train-optimization-v2`.

## Summary

The defect is real and is now fixed. Every case-study number in the ICTAI submission
was produced by sweeping thresholds over the same rows it then scored, so all of them
were in-sample. `compute_fdp_and_recall_clean.py` now selects the operating point on
TRAIN rows only and applies it unchanged to TEST rows, with a hard leak guard.

**The to-do expected the number to drop. It mostly does not, and that is the honest
result.** The headline group-level figure survives the clean split and survives a
stricter doubly-clean split as well:

| quantity | in-sample (published) | clean split | doubly clean |
|---|---|---|---|
| G2 recall @ P≥0.98 | **0.8340** | **0.8689** (prec 0.991, n=125) | **0.8750** (prec 1.000, n=48) |
| pooled recall @ P≥0.98 | **0.6835** @ prec 0.9835 | **0.6595** @ prec 0.9839 (n=265) | **0.6301** @ prec 1.000 (n=107) |

The pooled in-sample computation reproduces the figure the to-do cites (0.683 recall at
0.983 precision) **exactly**, which validates the pooling convention used throughout.

What *does* break is a different claim: **the precision floor is not honoured
out-of-sample.** See §4.

## 1. What was wrong

`compute_fdp_and_recall.py::main` writes the unsuffixed `fdp_and_recall.csv` — the file
the manuscript reads — from the `"all"` split, and `best_recall()` sweeps candidate
thresholds over exactly the rows it then scores. Selection and evaluation share every
patient.

The `train`/`test` suffixed outputs that the same function already writes do **not**
fix this. They re-run the *same* in-sample sweep independently within each split, so
`fdp_and_recall_test.csv` is in-sample on the test rows. The split files were read but
never used to separate selection from evaluation.

## 2. What was built

`filter-repository-python/compute_fdp_and_recall_clean.py`:

- `select_on_train()` — picks `(threshold, direction)` maximising TRAIN recall subject
  to TRAIN precision ≥ floor, and returns nan if the floor is unreachable on train
  (rather than silently falling back to an in-sample choice).
- `assert_no_leak()` — raises `LeakageError(AssertionError)` if the selection set and
  the evaluation set share any patient id. It raises; it never warns. It is called on
  the raw split files, on every group subset, and inside the pooled path.
- Outputs `data/fdp_and_recall_clean.csv` with, per group and floor, side by side:
  train-selected threshold and direction, train precision, train recall, test recall,
  test precision, test TP/FP/FN, the in-sample ceiling on the test rows, and a
  `test_precision_meets_floor` flag.
- Also `data/fdp_and_recall_clean_pooled.csv` and
  `data/fdp_and_recall_clean_doubly_clean.csv`.

Tests: `filter-repository-python/tests/test_clean_split_leak_guard.py`, 12 tests. The
key ones deliberately feed an overlapping split to `clean_split_rows()` and
`pooled_clean()` and assert the run aborts. Full suite: **52 passed** (12 new, 40
pre-existing from A1).

> **Note on output location.** The to-do says `data/fdp_and_recall_clean.csv`. It is
> written to `filter-repository-python/data/`, not `paper/*/data/`, because writing
> under `paper/` is off-limits for this session. The original script writes its outputs
> into the paper data directory; if the manuscript build expects the clean file there,
> that copy is Henrik's to make.

## 3. Split provenance

`data/train.csv` (730) and `data/test.csv` (500) are disjoint. Of the 671 patients in
`per_patient_scores.csv`, 403 fall in train and 268 in test; none fall outside.

The 1230 patients in the split files far exceed the 671 with filter scores — 559 have
no score. That mismatch is A4's subject (#5) and is not resolved here.

## 4. Per-group results: in-sample vs clean split

Non-survivor target (label 0). "in-sample" is the published
`paper/ictai2026/data/fdp_and_recall.csv`. "clean" selects on train, applies to test.
`meets` = did test precision actually reach the floor.

| group | floor | in-sample recall (n=671/658/…) | clean test recall | clean test precision | TP | FP | meets floor |
|---|---|---|---|---|---|---|---|
| Full cohort | 1.00 | 0.0714 | 0.1915 | 0.9474 | 36 | 2 | **no** |
| Full cohort | 0.99 | 0.0714 | 0.1915 | 0.9474 | 36 | 2 | **no** |
| Full cohort | 0.98 | 0.2210 | 0.2872 | 0.9643 | 54 | 2 | **no** |
| Full cohort | 0.95 | 0.5402 | 0.4521 | 0.9659 | 85 | 3 | yes |
| Assigned | 1.00 | 0.0734 | 0.1892 | 0.9459 | 35 | 2 | **no** |
| Assigned | 0.99 | 0.0734 | 0.1892 | 0.9459 | 35 | 2 | **no** |
| Assigned | 0.98 | 0.1353 | 0.2811 | 0.9630 | 52 | 2 | **no** |
| Assigned | 0.95 | 0.5367 | 0.4486 | 0.9651 | 83 | 3 | yes |
| G1 | 1.00 | 0.3667 | 0.6000 | 1.0000 | 6 | 0 | yes |
| G1 | 0.99 | 0.3667 | 0.6000 | 1.0000 | 6 | 0 | yes |
| G1 | 0.98 | 0.3667 | 0.6000 | 1.0000 | 6 | 0 | yes |
| G1 | 0.95 | 0.3667 | 0.6000 | 1.0000 | 6 | 0 | yes |
| **G2** | 1.00 | 0.1245 | 0.2623 | 0.9697 | 32 | 1 | **no** |
| **G2** | 0.99 | 0.1245 | 0.2623 | 0.9697 | 32 | 1 | **no** |
| **G2** | **0.98** | **0.8340** | **0.8689** | 0.9907 | 106 | 1 | yes |
| **G2** | 0.95 | 0.8340 | 0.8770 | 0.9907 | 107 | 1 | yes |
| G3 | 1.00 | 0.2900 | 0.1579 | 1.0000 | 6 | 0 | yes |
| G3 | 0.99 | 0.2900 | 0.1579 | 1.0000 | 6 | 0 | yes |
| G3 | 0.98 | 0.6200 | 0.1579 | 1.0000 | 6 | 0 | yes |
| G3 | 0.95 | 0.7900 | 0.6842 | 0.9630 | 26 | 1 | yes |
| G4 | 1.00 | 0.0976 | 0.2667 | 0.8000 | 4 | 1 | **no** |
| G4 | 0.99 | 0.0976 | 0.2667 | 0.8000 | 4 | 1 | **no** |
| G4 | 0.98 | 0.0976 | 0.2667 | 0.8000 | 4 | 1 | **no** |
| G4 | 0.95 | 0.0976 | 0.2667 | 0.8000 | 4 | 1 | **no** |

Test-set group sizes: G1 68, G2 125, G3 43, G4 29.

### 4a. The finding that actually hurts

**The precision floor is an in-sample artifact.** In 12 of the 24 group×floor cells
above, the operating point selected on train fails to reach its own floor on test.
Most damaging, at floor = 1.00 — the manuscript's "strong filter, precision = 1.0"
condition — the realised test precision is:

- Full cohort **0.9474**, Assigned **0.9459**, G2 **0.9697**, G4 **0.8000**.

Only G1 and G3 hold precision 1.0 on held-out data, and both do so on 6 predictions.
A claim of perfect precision cannot be supported from the held-out data at cohort level.

The mechanism is visible in the thresholds: on the full cohort, precision 1.0 in-sample
needs τ = 0.9302 (recall 0.0714), but on the smaller 403-row train set the floor is
reachable at τ = 0.689 (recall 0.2423). The smaller the selection set, the easier the
floor is to satisfy by luck, and the more the threshold overfits. Two false positives
on test are enough to break it.

**G4 should not be reported as a working stratum.** Test precision 0.80 at every floor
including 1.00, off 5 predictions in 29 patients.

## 5. Pooled results

Pooling convention: each assigned group G1–G4 contributes its own group-specific
operating point, predicted-positive sets are unioned, precision/recall computed once
over the union. In-sample pooling reproduces the published figure exactly.

| floor | in-sample recall | in-sample precision | clean test recall | clean test precision | clean TP | clean FP |
|---|---|---|---|---|---|---|
| 1.00 | 0.1766 | 1.0000 | 0.2595 | 0.9600 | 48 | 2 |
| 0.99 | 0.1766 | 1.0000 | 0.2595 | 0.9600 | 48 | 2 |
| **0.98** | **0.6835** | **0.9835** | **0.6595** | **0.9839** | 122 | 2 |
| 0.95 | 0.7225 | 0.9752 | 0.7730 | 0.9795 | 143 | 3 |

In-sample n = 658; clean test n = 265.

At the operating floor the manuscript uses (0.98), pooled recall falls from 0.6835 to
**0.6595**, a 2.4-point drop, at essentially unchanged precision. That is a small,
honest correction — not a collapse.

## 6. Residual leakage this fix does NOT remove

The G1–G4 grouping comes from a trimmed decision tree fitted on `auth_split/train.csv`
(1087 patients). **`data/train.csv` / `data/test.csv` are a different partition that
cross-cuts `auth_split/`**: of the 268 scored test patients, **160 (60%) sit inside the
tree's own training set.** Fixing threshold selection therefore does not by itself
produce a fully out-of-sample estimate — the group *assignment* stage remains in-sample
for most of the test set.

Sensitivity analysis on the 108 test patients held out of *both* stages
(`fdp_and_recall_clean_doubly_clean.csv`, 107 with scores, 73 non-survivors):

| floor | recall | precision | TP | FP |
|---|---|---|---|---|
| 1.00 | 0.2603 | 1.0000 | 19 | 0 |
| 0.99 | 0.2603 | 1.0000 | 19 | 0 |
| **0.98** | **0.6301** | **1.0000** | 46 | 0 |
| 0.95 | 0.7123 | 0.9811 | 52 | 1 |

G2 alone on this subset (n = 48, 48 non-survivors): recall **0.8750** at precision
**1.0000** at floors 0.98 and 0.95.

The doubly-clean estimate is *better behaved* than the merely clean one — precision 1.0
with zero false positives at the 0.98 floor. Pooled recall steps down monotonically
across the three regimes (0.6835 → 0.6595 → 0.6301), which is the expected direction
and a small effect.

Caveat: n = 107 pooled and n = 48 for G2. These intervals will be wide; B6 (#12) should
attach confidence intervals before any of these numbers go in the manuscript.

## 7. What this means for the manuscript

1. **The headline 83.4% survives.** G2 recall at P≥0.98 is 0.8340 in-sample, 0.8689 on
   the clean split, 0.8750 doubly clean. It should be re-reported as a held-out number,
   which makes it *stronger* evidence, not weaker.
2. **The "13.5% → 83.4%" contrast weakens and should be restated.** Under the clean
   split the cohort-level (Assigned) comparator at P≥0.98 rises from 0.1353 to 0.2811,
   so the contrast becomes roughly 28% → 87% rather than 13.5% → 83.4%. The direction
   holds; the magnitude roughly halves. Note also that the clean Assigned cell does not
   meet its floor (test precision 0.9630), so 0.2811 is not a like-for-like comparator —
   the honest cohort-level statement needs B6's intervals first.
3. **Drop or heavily qualify any precision = 1.0 claim.** It does not survive on the
   268-patient test set at cohort level (§4a).
4. **G4 should not be presented as a functioning stratum** (n=29, precision 0.80).
5. **The pooled number moves from 0.683 to 0.660** at the 0.98 floor.

Item 2 and item 3 contradict text currently in the manuscript. Item 1 does not — it
strengthens it.

## 8. Reproducing

```bash
cd filter-repository-python
python compute_fdp_and_recall_clean.py
python -m pytest tests/ -q
```

Outputs: `data/fdp_and_recall_clean.csv`, `data/fdp_and_recall_clean_pooled.csv`,
`data/fdp_and_recall_clean_doubly_clean.csv`. CPU only, a few seconds, no GPU.
