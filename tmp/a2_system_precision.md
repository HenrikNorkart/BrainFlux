# A2 — Enforcing the precision floor at system level

Session 2026-09-15. HQ to-do #8. CPU only, no GPU.

```bash
cd /c/code/PHD/BrainFlux
python -m pytest filter-repository-python/tests -q          # 69 passed
cd filter-repository-python && python ../tmp/a2_system_precision.py
```

---

## Summary

**The defect is real and the mechanism is now proven by test.** A union of per-cell rules
each at precision ≥ P can have system precision < P, and
`tests/test_system_precision_floor.py` builds the case explicitly: two cells agreeing on
which targets to catch but misfiring on different non-targets, each at exactly 0.900,
union 0.818.

**On the actual four-range suppression-ratio configuration the empirical damage is much
smaller than the to-do anticipated — it bites at one floor out of four, by 0.0008.** That
is the honest number and it is reported as such below rather than dressed up.

**Two things the to-do expected that did NOT happen**, both worth recording so nobody
re-litigates them:

- A2 is **not** the explanation for a non-monotone Table 1. Recall falls monotonically as
  the floor rises (0.5563 → 0.1192 → 0.1192 → 0.1192). The `0.309-at-1.00 /
  0.000-at-0.99` pattern the earlier comment hoped to reproduce does not appear in this
  configuration.
- The fix does **not** cost recall. At the one floor where the modes differ, the system
  mode gets *more* recall, not less.

The manuscript sentence still needs changing, but for a different reason than A2 —
see §4.

## 1. What was built (criterion 1)

`SingleLinearClassifier1D` gains `selection_mode`:

- **`"per_cell"`** — the historical behaviour, and **still the default** so every existing
  result reproduces without a flag. Each `(i, j)` cell independently satisfies the floor;
  `predict()` then ORs the cells.
- **`"system"`** — greedy forward selection against the OR-aggregated prediction. Each
  round adds at most one cell, and a candidate is admissible only if the *union* with the
  already-selected cells still satisfies the floor. It stops when no admissible addition
  improves system recall, so the fitted model satisfies the floor on the fitting data by
  construction.

Both modes share `_threshold_grid()`, so they search identical candidates and any
difference is attributable to the selection rule alone rather than to the grid.

## 2. The test (criterion 2)

`filter-repository-python/tests/test_system_precision_floor.py`, 10 tests. The fixture:

| rule | catches | TP | FP | precision |
|---|---|---|---|---|
| cell A | t1–t9, **f1** | 9 | 1 | 0.900 |
| cell B | t1–t9, **f2** | 9 | 1 | 0.900 |
| **A OR B** | t1–t9, f1, f2 | 9 | 2 | **0.818** |

`test_per_cell_mode_violates_the_system_floor` asserts the old mode emits 9/11 = 0.818
against a 0.90 floor. `test_system_mode_satisfies_the_system_floor` asserts the new mode
reaches the floor — **and** that it does so with non-zero recall, because satisfying a
floor by predicting nothing is not a fix. A parametrised test repeats the invariant at
0.95 / 0.98 / 0.99 / 1.00, allowing abstention at an unreachable floor.

## 3. Achieved system precision, four-range SR grid (criterion 3)

`GeneralRangeFilter(suppression_ratio, num_ranges=4, num_time_divisions=1)` — the Table 1
grid, **not** the single-cell case study, which is unaffected by this defect. Fit on
`auth_split/train.csv` (452 scored), evaluated on `auth_split/test.csv` (313 scored), the
split #106 settled on. Target class 0.

| floor | mode | cells | train precision | train recall | **meets floor at fit?** | test precision | test recall |
|---|---|---|---|---|---|---|---|
| **0.95** | per_cell | 4 | **0.9492** | 0.5563 | **NO** | 0.9590 | 0.5625 |
| **0.95** | **system** | 3 | **0.9508** | **0.5762** | **yes** | 0.9593 | **0.5673** |
| 0.98 | per_cell | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |
| 0.98 | system | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |
| 0.99 | per_cell | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |
| 0.99 | system | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |
| 1.00 | per_cell | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |
| 1.00 | system | 3 | 1.0000 | 0.1192 | yes | 0.9474 | 0.0865 |

**How badly was the published guarantee violated? At one floor, by 0.0008.** At floor 0.95
the per-cell fit selects a fourth cell whose false positives drag system precision to
0.9492, just under its own floor. At floors 0.98 and above every admissible cell happens
to be perfectly precise on the training rows, so their union is too, and the two modes
coincide exactly.

That is a much narrower failure than "the four-range filter carries no system-level
guarantee" implies. The guarantee was *unsound in principle* — nothing prevented a larger
violation, and the unit test shows how — but on this data it held everywhere except the
0.95 row.

### 3a. Data notes

Two suppression-ratio files are zero-length and fail to load
(`PUH-2015-102_4_ar.npy`, `PUH-2016-175_1_ar.npy`); the aggregator skips them. 635 train
and 412 test labels have no signal file, consistent with A4's finding that only 671 of
1231 labelled patients have an SR recording.

## 4. Recommendation (criterion 4)

**Report `selection_mode="system"`, and change the manuscript sentence anyway.**

Switching modes is close to free. At floors 0.98–1.00 the two modes produce *identical*
models, so nothing in the high-precision regime the paper operates in changes at all. At
0.95 the system mode is strictly better on both axes — it meets the floor the per-cell
fit misses, and its recall is *higher* (0.5762 vs 0.5563 on train, 0.5673 vs 0.5625 on
test), because dropping the fourth cell freed the greedy search to place the remaining
three more aggressively. **Table 1 changes in the 0.95 row only, and moves up.** There is
no argument for keeping the per-cell mode as the reported configuration.

But switching modes does **not** rescue this sentence in `sections/discussion.tex`:

> "every positive prediction continues to satisfy the same minimum-precision floor as the
> undivided-cohort filter"

Even in system mode the floor is guaranteed **on the fitting data only**. On held-out
rows, test precision at the 0.98, 0.99 and 1.00 floors is **0.9474** in both modes — the
floor is not met, and no fitting procedure can make it so. This is the same
generalisation failure A3 §4a found from the threshold-selection side, and it is the
larger of the two problems: A2 was a soundness bug worth 0.0008 here, whereas the
held-out shortfall is 0.03–0.05 and structural.

The honest formulation is that the floor is a **selection constraint applied during
fitting**, not a guarantee carried by deployed predictions. Suggested replacement:

> Each positive prediction is produced by a rule selected under a minimum-precision
> constraint on the training partition. We stress that this is a constraint on model
> selection, not a guarantee on held-out predictions: on the evaluation partition the
> realised precision of the four-range filter is 0.947 at nominal floors of 0.98 and
> above, so the floor should be read as the criterion that chose the operating point
> rather than as a bound the deployed filter satisfies.

## 5. Status against acceptance criteria

1. System-level selection mode added, old mode kept and named — **done**.
2. Multi-cell test asserting old violates / new satisfies, one file — **done** (10 tests).
3. Achieved system precision at all four floors — **done**, §3.
4. Recommendation and Table 1 impact — **done**, §4.
5. `python -m pytest filter-repository-python/tests -q` — **69 passed**.

## 6. What this unblocks

#14 (B3) and #16 (A5) list #8 as a prerequisite. Note for A5 in particular: the pooled
headline numbers it produces should use `selection_mode="system"`, and should carry the
held-out precision caveat from §4 rather than quoting the nominal floor.
