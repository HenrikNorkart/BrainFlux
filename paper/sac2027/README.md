# SAC 2027 submission

ACM SAC 2027, **ASH** track (Applications and Systems for Healthcare).
Deadline **2 October 2026**. Double-blind, ≥3 reviewers.
6–8 pages, +2 at extra cost, 10 maximum — current budget is **9**, see
`tmp/w1_claim.md` §3.

## Build

```bash
cd paper/sac2027
latexmk -pdf main.tex
grep PAGECOUNT build.log     # page budget, printed on every run
```

Requires `acmart.cls` (present in the local MiKTeX install).

## Boundaries

- **Never edit `paper/ictai2026/`.** That is the ICTAI submission of record.
- **Never push to the Overleaf remote.** This directory is local + GitHub only.

## Framing

Fixed in `tmp/w1_claim.md` (HQ #132) — the claim, five contributions, section arc,
figure plan, and the answer to the strongest objection. **Do not drift from it
without updating that document first.** Three questions are open for Henrik there:
whether to buy the 9th page, whether to foreground the expert-plausibility finding,
and whether the title changes.

## Where the numbers live

Nothing in this paper needs recomputing. Every result is already written up:

| report | what it carries |
|---|---|
| `tmp/b1_sham_control.md` | sham control, 2000 exact-matched partitions |
| `tmp/b2_ehr_only.md` | features predict survival, AUROC + CIs |
| `tmp/c1_temporal_cutoff.md` | coverage table, base-rate collapse |
| `tmp/c117_fixed_window.md` | G2 recall 0.8163 → 0.3048 under 24 h |
| `tmp/a5_headline_numbers.md` | pooled clean-split numbers, floor failures |
| `tmp/b6_intervals.md` | fragility of all 56 cells |
| `tmp/t4_24h_sham_precheck.md` | fresh 24 h run, no usable partition |
| `tmp/b4_nonllm_baseline.md`, `tmp/b4b_interaction_baseline.md` | non-LLM baselines |
| `tmp/b5_topp_sweep.md` | construct survives at top-*p* = 0.1 |
| `tmp/a4_cohort_reconciliation.md`, `tmp/c2_wlst.md` | ready-to-paste prose |

Reference implementation of the gate: `filter-repository-python/sham_gate.py`.

## Known gap

`data/` and `*.pdf` are gitignored repo-wide, so a clean checkout cannot build the
figures until the `compute_*` scripts in `filter-repository-python/` are re-run.
That matches how `paper/ictai2026/` already works; changing it is a separate call.

## Drafting tasks

Each section file opens with a comment naming its HQ to-do, page budget and sources.

| task | HQ | section |
|---|---|---|
| W3 | #134 | `results.tex`, `survives.tex` |
| W4 | #135 | `pipeline.tex`, `controls.tex` |
| W5 | #136 | `abstract.tex`, `introduction.tex`, `background.tex`, conclusion |
| W6 | #137 | `discussion.tex`, limitations |
| W7 | #138 | final pass — page budget, consistency, submittable PDF |
