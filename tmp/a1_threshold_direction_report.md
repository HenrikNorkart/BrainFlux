# A1 — Threshold direction inversion: fix, tests, and what it does *not* explain

Nightshift session, 2026-09-11. HQ to-do #3. Branch `fix/threshold-train-optimization-v2`.

## Summary

The inversion is real and is now fixed, with a test suite that fails on the pre-fix
code. **But the to-do's diagnosis is wrong on one point, and it matters:** this defect
is *not* the cause of the non-monotone Table 1 in the ICTAI submission. On the
production setting (`target_class=0`) the old code was behaviourally equivalent to the
fixed code, and reproduces the paper's numbers exactly. The non-monotonicity has to
come from somewhere else — most plausibly A2 (per-cell precision floor + OR
aggregation across cells).

## What was actually broken

`filter-repository-python/brainflux/classifiers/classifiers/single_linear_classifier_1d.py`,
`_evaluate_thresholds`, carried **two** compounding defects, not one:

1. **Swapped direction branches** (the one the to-do describes):
   `"under" -> (x >= t)` and `"over" -> (x <= t)`, both inverted relative to their
   names, while `fit` stores sign 0 for "under" and `predict` applies sign 0 as
   `x <= t`.

2. **Prediction mask compared against the class label** (not in the to-do):
   ```python
   TP = np.sum((labels == self.target_class) & (predictions == self.target_class))
   ```
   `predictions` is a 0/1 *membership mask* for the predicted-positive set, but it was
   compared against `self.target_class` as if it were a vector of class labels.

### The two defects cancel at `target_class = 0`

With `target_class = 0`, `predictions == self.target_class` means `predictions == 0`,
i.e. **predicted negative** — which re-inverts defect (1). Composing them:

| direction | pre-fix predicted-positive set, `target_class=0` | correct set |
|---|---|---|
| `"under"` | `{x < t}`  | `{x <= t}` |
| `"over"`  | `{x > t}`  | `{x >= t}` |

The old code was therefore a *correct* classifier with a **strict** inequality. Only
points sitting exactly on the threshold differ.

With `target_class = 1` the defects do **not** cancel and the deployed rule is the
exact complement of the fitted one. Every call site in the repo
(`external_connectors/brainflux_filter_pipeline.py:168`, `gui_tkinter/filter_gui.py:634,696,707`)
passes `target_class=0`, so production never hit the non-cancelling case.

`predict()`, `brainflux/evaluators/linear1d/classification_1D.py` and
`compute_fdp_and_recall.py::precision_recall` all already agreed on the canonical
convention; only `_evaluate_thresholds` disagreed.

## The fix

`_evaluate_thresholds` now builds a boolean `predicted_positive` mask with the
reference convention (`"under" -> x <= t`, `"over" -> x >= t`) and counts TP/FP/FN
against `labels == self.target_class`. It is now correct for any `target_class`, not
just `{0, 1}` — previously any `target_class` outside `{0, 1}` produced meaningless
precision and recall, because no `predictions` value could ever equal it.

Diff is confined to `_evaluate_thresholds`. No change to `fit`, `predict`,
`save_model`, `load_model`, or the sign encoding, so saved `.npz` models remain valid.

## Empirical check: the fix changes no reported number

`tmp/a1_impact_check.py` replays old vs. new `_evaluate_thresholds` through the same
threshold sweep `fit()` uses, over the real per-patient filter scores in
`filter-repository-python/oos_per_patient.csv` (n=1812, 1227 non-survivors),
`target_class=0`:

```
=== score_top  (exact 0.001-grid ties=744 of 1812) ===
 floor | recall_old recall_new |  tau_old  tau_new | dir_old dir_new
  0.90 |     0.6797     0.6797 |    0.009    0.009 |    over    over
  0.95 |     0.5705     0.5705 |    0.056    0.056 |    over    over
  0.98 |     0.3350     0.3350 |    0.386    0.386 |    over    over
  0.99 |     0.2046     0.2046 |    0.703    0.703 |    over    over
  1.00 |     0.0693     0.0693 |    0.929    0.929 |    over    over

=== score_bot  (exact 0.001-grid ties=299 of 1812) ===
 floor | recall_old recall_new |  tau_old  tau_new | dir_old dir_new
  0.90 |     0.7058     0.7058 |    0.703    0.703 |   under   under
  0.95 |     0.5460     0.5460 |    0.306    0.306 |   under   under
  0.98 |     0.2934     0.2934 |    0.025    0.025 |   under   under
  0.99 |     0.0000     0.0000 |     None     None |    None    None
  1.00 |     0.0000     0.0000 |     None     None |    None    None
```

Identical recall, identical `tau`, identical direction at every precision floor,
despite hundreds of exact grid ties. **Re-running A6 will not move the headline
numbers on account of A1.**

## What this means for the manuscript

- The to-do body states the inversion "is the root cause of the non-monotone Table 1
  (baseline recall 0.309 at P=1.00 but 0.000 at P=0.99)". **That claim does not
  survive.** A single-threshold sweep is monotone in the precision floor both before
  and after the fix (see the `score_bot` column above: 0.7058 → 0.5460 → 0.2934 → 0 →
  0, non-increasing). If the same explanation has been written into the SAC 2027
  rebuttal or response-to-reviewers draft, it needs to be removed.
- The remaining candidate mechanism is the one A2 targets: `fit` enforces the precision
  floor **per (feature, channel) cell**, while `predict` ORs the cells together with
  `np.max(predicts, axis=(1, 2))`. A union of cells each at precision ≥ P can have
  global precision < P, and the set of winning cells changes discontinuously with the
  floor — so the *global* recall reported in Table 1 need not be monotone even though
  each cell's is. This is a system-level precision-floor bug, and it can produce
  exactly the 0.309-at-1.00 / 0.000-at-0.99 pattern. A2 (#8) should be treated as the
  live explanation, and A1 should not be cited as the cause.
- Decision left for Henrik: whether the non-monotonicity claim appears anywhere in
  `paper/` or in the response letter. Per the hard boundary, nothing under `paper/`
  was read or edited by this session.

## Tests

`filter-repository-python/tests/test_single_linear_classifier_1d.py`, 42 tests, all
passing on the fixed code; **23 fail on the pre-fix code**.

| test | covers |
|---|---|
| `test_evaluate_thresholds_matches_reference_implementation` | agreement with `compute_fdp_and_recall.py::precision_recall`, both directions × `target_class ∈ {0,1}` |
| `test_evaluate_thresholds_is_inclusive_at_exact_ties` | `x == t` belongs to the positive side (the only pre/post difference at `target_class=0`) |
| `test_evaluate_thresholds_under_selects_low_values` | direction semantics, literal |
| `test_fit_predict_round_trip_random` (12 seeds × 2 classes) | **AC2** — fit-time P/R == P/R recomputed from `predict()` output, within 1e-9 |
| `test_fit_predict_round_trip_separable_both_directions` | **AC4** — both directions incl. target class *below* the threshold; also asserts the recovered sign |
| `test_recall_is_non_increasing_in_precision_floor` | **AC3** — recall non-increasing across [0.90, 0.95, 0.98, 0.99, 1.00] |

Pre-fix failures are concentrated on `target_class=1` (all 12 round-trip seeds, both
separable directions, both monotonicity directions) plus all four tie parametrisations
at both classes — which is precisely the signature predicted by the cancellation
analysis above.

Run with:

```bash
python -m pytest filter-repository-python/tests -q
```

(needs `numpy` + `pytest`; runs on the Windows host, no devcontainer or GPU.)

## Note on repository layout

`filter-repository-python/` is a **nested, separate git repository**
(`github.com/BrainFlux-Lab/filter-repository-python`), untracked by the outer
`BrainFlux` repo. The classifier fix and tests were therefore committed inside the
nested repo, on a branch named `fix/threshold-train-optimization-v2` to match the
outer repo's branch. The `git stash` verification in the to-do's VERIFY block has to
be run from inside `filter-repository-python/`, not from the BrainFlux root, or it
stashes nothing.

## Follow-ups filed

- Nothing new beyond A2 (#8), which already covers the system-level precision floor and
  is now the sole live explanation for the Table 1 non-monotonicity.
