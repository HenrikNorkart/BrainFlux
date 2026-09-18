# B6 — Confusion counts and confidence intervals

Session 2026-09-15. HQ to-do #12. CPU only, no GPU.

```bash
cd /c/code/PHD/BrainFlux
python -m pytest filter-repository-python/tests/test_intervals.py -q   # 44 tests
python tmp/b6_intervals.py
```

---

## Summary

**Every one of the 56 cells in A3's tables is fragile** — each rests on fewer than ten of
whichever event is rarer. Not most of them; all of them.

The consequence is blunt: **no precision figure in this paper should appear without its
interval.** The "precision = 1.000" cells have Wilson lower bounds between 0.51 and 0.84.
A reader currently sees 1.000 and reasonably concludes the filter does not make mistakes;
the data supports no such conclusion.

Separately, the bootstrap in §4 shows the paper's *own* comparison — routed versus
undivided — is statistically solid at every floor. Read alongside B1 (#10), that produces
the session's sharpest single sentence: **the comparison the manuscript makes is sound;
it is simply the wrong comparison.**

## 1. What was built (criteria 1 and 2)

`filter-repository-python/brainflux/utils/intervals.py`:

- `wilson_interval` — the display default; behaves at p near 0 and 1, where the normal
  approximation leaves the unit range.
- `clopper_pearson_interval` — conservative exact alternative, for a reviewer who asks
  for guaranteed coverage.
- `paired_diff_bootstrap` — CI for the difference between two **dependent** proportions
  measured on the same patients. Resamples patients, not arms, so the dependence between
  routed and undivided is preserved; an independent two-sample interval would be wrong
  here.
- `is_fragile` — counts the **rarer** of successes/failures, because a precision of 0.983
  from 298 TP and 5 FP is decided by the 5.

`tests/test_intervals.py`, **44 tests**, validated against
`statsmodels.stats.proportion.proportion_confint` as an independent oracle at `abs=1e-6`
over 16 count pairs spanning the regimes this project reports in, at three alpha levels.
The test does not re-implement the formula it checks.

Two things the tests caught and that are worth recording:

- Wilson's upper bound at `k == n` is analytically exactly 1 but lands a few ulps below,
  which left the point estimate outside its own interval. Pinned in the implementation.
- "Clopper-Pearson is always wider than Wilson" is **false at the degenerate ends** — at
  48/48, Wilson spans 0.92590–1 and CP 0.92597–1. That is a real property of the two
  formulas, so the test was corrected to assert conservatism only in the interior rather
  than asserting the convenient thing away.

## 2. Reissued cells (criterion 3)

Both splits are produced: `auth_split` is what #106 settled on and is primary; `data` is
the split A3 originally used, so that document can be checked cell for cell. **The `data`
column reproduces A3 exactly** (G2 at 0.98: 0.8689, precision 0.9907, and pooled 0.6595 at
0.9839), which validates the scoring path.

### `data` split — A3's own numbers, now with counts and Wilson 95 % CIs

| stratum | floor | TP | FP | FN | TN | precision [95 % CI] | recall [95 % CI] |
|---|---|---|---|---|---|---|---|
| Full cohort | 1.00 | 36 | 2 | 152 | 78 | 0.9474 [0.8271, 0.9854] | 0.1915 [0.1416, 0.2537] |
| G1 | 1.00 | 6 | 0 | 4 | 58 | **1.0000 [0.6097, 1.0000]** | 0.6000 [0.3127, 0.8318] |
| G3 | 1.00 | 6 | 0 | 32 | 5 | **1.0000 [0.6097, 1.0000]** | 0.1579 [0.0744, 0.3042] |
| G4 | 1.00 | 4 | 1 | 11 | 13 | 0.8000 [0.3755, 0.9638] | 0.2667 [0.1090, 0.5195] |
| **G2** | **0.98** | **106** | **1** | 16 | 2 | **0.9907 [0.9490, 0.9983]** | **0.8689 [0.7975, 0.9176]** |
| **Pooled G1–G4** | **0.98** | **122** | **2** | 63 | 78 | **0.9839 [0.9431, 0.9956]** | **0.6595 [0.5885, 0.7239]** |
| Pooled G1–G4 | 0.95 | 143 | 3 | 42 | 77 | 0.9795 [0.9413, 0.9930] | 0.7730 [0.7074, 0.8274] |

### `auth_split` — the reporting split

| stratum | floor | TP | FP | FN | TN | precision [95 % CI] | recall [95 % CI] | meets floor |
|---|---|---|---|---|---|---|---|---|
| Full cohort | 1.00 | 7 | 0 | 172 | 94 | 1.0000 [0.6457, 1.0000] | 0.0391 [0.0191, 0.0785] | yes |
| G1 | 1.00 | 4 | 0 | 8 | 61 | **1.0000 [0.5101, 1.0000]** | 0.3333 [0.1381, 0.6094] | yes |
| G4 | any | 0 | 0 | 21 | 18 | — | 0.0000 [0.0000, 0.1546] | no |
| **G2** | **0.98** | **80** | **3** | 18 | 3 | **0.9639 [0.8990, 0.9876]** | **0.8163 [0.7283, 0.8805]** | **no** |
| **Pooled G1–G4** | **0.98** | **94** | **3** | 79 | 90 | **0.9691 [0.9130, 0.9894]** | **0.5434 [0.4690, 0.6158]** | **no** |
| Pooled G1–G4 | 0.95 | 114 | 5 | 59 | 88 | 0.9580 [0.9054, 0.9819] | 0.6590 [0.5856, 0.7255] | yes |

Full 56-row table with Clopper-Pearson bounds in `tmp/b6_intervals.csv`.

## 3. Fragility (criterion 5)

**56 of 56 cells are flagged.** The specific cases that most need the flag:

1. **Every "precision = 1.000" claim.** G1 at 1.00 rests on 4 predictions: the interval is
   [0.5101, 1.0000]. G3 on 6: [0.6097, 1.0000]. Full cohort on 7: [0.6457, 1.0000]. These
   are consistent with a true precision near one-half. The manuscript's perfect-precision
   language cannot be supported by four to seven events.
2. **The pooled 0.98 precision.** 0.9839 on the `data` split rests on **2** false
   positives; the interval reaches down to 0.9431. Even where `meets_floor` is True, the
   lower bound sits below the nominal floor — the floor is met by the point estimate only.
3. **G4 is empty on `auth_split`.** Zero predictions at every floor: recall
   0.0000 [0.0000, 0.1546]. It should not be presented as a working stratum, consistent
   with A3 §4a.
4. **G2 is the sturdiest cell in the paper and still rests on 1–3 false positives.**
   Its recall interval, ±7 points, is the honest width for the headline.

## 4. Routed versus undivided, paired bootstrap (criterion 4)

10 000 resamples, patients resampled jointly so the two arms stay paired. Units are the
non-survivors present in the assigned test rows.

| split | floor | n | routed recall | undivided recall | difference [95 % CI] | excludes 0 |
|---|---|---|---|---|---|---|
| auth_split | 1.00 | 173 | 0.1156 | 0.0405 | **+0.0751 [+0.0347, +0.1156]** | yes |
| auth_split | 0.99 | 173 | 0.5434 | 0.0405 | **+0.5029 [+0.4220, +0.5780]** | yes |
| auth_split | 0.98 | 173 | 0.5434 | 0.0405 | **+0.5029 [+0.4220, +0.5780]** | yes |
| auth_split | 0.95 | 173 | 0.6590 | 0.4046 | **+0.2543 [+0.1792, +0.3295]** | yes |
| data | 0.98 | 185 | 0.6595 | 0.2811 | **+0.3784 [+0.3027, +0.4541]** | yes |
| data | 0.95 | 185 | 0.7730 | 0.4486 | **+0.3243 [+0.2541, +0.3946]** | yes |

**The routed system beats the undivided baseline at every floor on both splits, and every
interval excludes zero by a wide margin.** This is a real, statistically well-supported
effect, and the manuscript is not wrong to report it.

## 5. How this sits with B1

B6 and B1 (#10) are not in conflict, and the combination is the thing to understand:

- **Against the undivided cohort** the routed system's advantage is large and its interval
  excludes zero comfortably (§4).
- **Against a partition matched on size and class balance** it has no advantage at all —
  pooled percentiles 2.4 / 45.6 / 24.8 / 16.7, p = 0.98 / 0.54 / 0.75 / 0.83.

So the effect the paper measures is real; the question is what it is an effect *of*. The
undivided baseline does not control for class balance, and B2 (#11) shows the discovered
features predict survival directly. The honest reading is that the routed system's gain
over the undivided filter is largely the gain from sorting patients by outcome risk, which
a random balance-matched partition achieves equally well.

## 6. What to do in the manuscript

1. **Attach an interval to every precision and recall.** Use Wilson; note Clopper-Pearson
   is available if a reviewer prefers exact coverage.
2. **Report raw TP/FP/FN/TN alongside every rate.** A reader who sees "FP = 2" will
   calibrate correctly without needing the interval explained.
3. **Delete perfect-precision claims** or state them as "1.000 (95 % CI 0.51–1.00, 4
   predictions)", which makes the point honestly and is self-limiting.
4. **Keep the routed-versus-undivided bootstrap** — it is a genuine result — but present
   it next to B1's sham control rather than alone, and say plainly which comparator
   supports which claim.
5. **Drop G4** from any stratum-level result.

## 7. Status against acceptance criteria

1. Reusable Wilson / Clopper-Pearson / paired-bootstrap utility — **done**,
   `brainflux/utils/intervals.py`.
2. Unit test against an independent oracle at 1e-6, not re-implementing the formula —
   **done**, 44 tests vs `statsmodels`, including three alpha levels and degenerate ends.
3. Every A3 precision and recall reissued with TP/FP/FN/TN and 95 % CI — **done**, §2,
   both splits, 56 cells; the `data` column reproduces A3 exactly.
4. Bootstrap CI on the improvement over the undivided baseline, resamples stated —
   **done**, §4, 10 000 resamples, paired.
5. Fragile cells flagged — **done**, §3; all 56 qualify.

Full suite after this work: **115 passed**.
