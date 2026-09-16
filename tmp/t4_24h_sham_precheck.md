# T4 — Sham-gate pre-check on the completed 24 h run

Session 2026-09-16. HQ to-do #127. CPU only, no GPU.

```bash
cd /c/code/PHD/BrainFlux
python tmp/t4_24h_sham_precheck.py --n 2000
```

---

## Verdict (criterion 4, one sentence)

**FAIL at every floor — but the more informative result is that the 24 h run's features do
not produce a usable partition at all: the 4-leaf tree puts 661 of 671 patients (98.5 %)
into a single leaf, so what the gate actually measured was a partition that barely
partitions.**

And the caveat the to-do requires repeated: **this run was retrieval-free on both channels**
(`knowledge.py:167` hardcodes a `/workspaces/` path that does not exist on CAIR, so Chroma
opened an empty store; no Serper key at launch). The published runs had the medical-
knowledge store live. A failure here is therefore a **lower bound** and does **not** settle
the question — it says the handicapped run does not clear the bar, not that the loop
cannot. T3 (#126) removes the handicap; T5 (#128) is the real test.

## 1. The partition is degenerate

`grp_24disc` comes from `tmp/b24h_rescore.py`: the phantom target built on TRAIN only
exactly as B4 builds it, then a 4-leaf `DecisionTreeClassifier` fitted on TRAIN only over
the ten discovered features, applied to all patients. (The published pipeline used a tree
*regressor* pruned to four leaves; this is the classifier equivalent on the thresholded
target. Close enough to compare, and noted here because it is not literally identical.)

| 24 h discovered | n | survivors | survivor fraction |
|---|---|---|---|
| D1 | **661** | 220 | 0.3328 |
| D3 | 5 | 0 | 0.0000 |
| D5 | 4 | 3 | 0.7500 |
| D6 | 1 | 0 | 0.0000 |

| published G1–G4 | n | survivors | survivor fraction |
|---|---|---|---|
| G1 | 164 | 134 | 0.8171 |
| G2 | 280 | 15 | 0.0536 |
| G3 | 116 | 16 | 0.1379 |
| G4 | 98 | 57 | 0.5816 |

D1's survivor fraction (0.3328) is the cohort base rate. The tree found three slivers of
5, 4 and 1 patients and left everything else undivided. That is what "no day-one structure"
looks like mechanically, and it matches what C1 (#13) predicted: within 24 h, pain score,
GCS, sedation and FiO₂ are too sparse to support the published vocabulary, and the run fell
back on raw vital-sign aggregates that do not separate the cohort.

## 2. Gate results (criteria 2 and 3)

2000 shams, exact size and balance match, seed 20260915. Both partitions scored in the same
run so nothing is compared across sessions.

**24 h discovered**

| floor | recall | precision | TP | FP | meets floor | sham mean | sham SD | percentile | p |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.0447 | 1.0000 | 8 | 0 | yes | 0.0469 | 0.0073 | 22.10 | 0.779 |
| 0.99 | 0.0447 | 1.0000 | 8 | 0 | yes | 0.0469 | 0.0073 | 22.10 | 0.779 |
| 0.98 | 0.0447 | 1.0000 | 8 | 0 | yes | 0.0508 | 0.0265 | 21.55 | 0.785 |
| 0.95 | 0.5084 | 0.9579 | 91 | 4 | yes | 0.4150 | 0.0335 | **93.15** | **0.069** |

**Published G1–G4** (reproduces B1 exactly, as T1 established)

| floor | recall | precision | TP | FP | meets floor | sham mean | sham SD | percentile | p |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.1156 | 1.0000 | 20 | 0 | yes | 0.4157 | 0.1578 | 2.40 | 0.976 |
| 0.99 | 0.5434 | 0.9691 | 94 | 3 | **no** | 0.5147 | 0.1436 | 45.65 | 0.544 |
| 0.98 | 0.5434 | 0.9691 | 94 | 3 | **no** | 0.5916 | 0.0816 | 24.80 | 0.752 |
| 0.95 | 0.6590 | 0.9580 | 114 | 5 | yes | 0.7249 | 0.0740 | 16.70 | 0.833 |

Against the bar T2 (#125) proposes — percentile ≥ 95 and p ≤ 0.05 — **both partitions fail
at every floor.** The 24 h partition's 0.95 row is the closest anything has come (93.15,
p = 0.069), and it is still short.

## 3. A methodological problem this exposed, which T2 should fix

The 24 h partition scores **percentile 93.15 at floor 0.95 while the published partition
scores 16.70** — even though the 24 h partition's actual recall is *lower* (0.5084 vs
0.6590).

That is not the 24 h partition being better. It is an artefact of how the gate works. The
sham distribution is matched to *the partition under test*, so a degenerate partition gets
compared against degenerate shams: shuffling patients between a group of 661 and groups of
5, 4 and 1 changes almost nothing, which collapses the sham spread (SD 0.0335 at floor 0.95,
against 0.0740 for the published partition; 0.0073 vs 0.1578 at floor 1.00). Against a
distribution that tight, a mediocre real value lands at a high percentile. The heavy ties
are visible in the numbers — the real value 0.508380 equals the sham p95 to six decimals,
and 137 shams tie or exceed it.

**Percentiles are therefore not comparable across partitions of different shapes.** The gate
is sound as a within-partition test — "is this partition better than chance *given its own
shape*" — but it cannot rank two differently-shaped partitions, and a sufficiently
degenerate partition could in principle clear the bar while being useless.

**Recommendation for T2, before it is signed off:** add a non-degeneracy precondition, so a
partition must be a real partition before the gate result means anything. Suggested: every
group holds at least 20 patients and at least 5 of the rarer outcome class, or the run is
recorded as "no usable partition found" rather than as a gate FAIL. On this cohort the
published tree passes that precondition and the 24 h run does not — which is the honest
description of what happened.

## 4. What this does and does not tell us

**Does:** the handicapped 24 h run does not clear the bar, and did not produce a usable
partition to begin with. There was no cheap win waiting — the GPU work in T5 is still
required.

**Does not:** it does not show the loop cannot find day-one structure. Two live alternative
explanations remain, and they are exactly the ones T5 is designed to remove — the retrieval
handicap (T3/#126), and the single stochastic draw (T2 pre-registers three seeds).

## 5. Files

- `tmp/t4_24h_sham_precheck.py` — the run
- `tmp/t4_24h_sham_precheck.csv` — both partitions, all four floors
- `filter-repository-python/sham_gate.py` — the gate (T1/#124)
