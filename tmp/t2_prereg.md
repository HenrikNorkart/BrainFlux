# T2 — Pre-registration for the option-4 discovery run

Session 2026-09-16. HQ to-do #125. **Status: awaiting Henrik's sign-off.**

Nothing in this document may be changed after the run starts. If something here turns out
to be wrong, the honest move is to say so in T6 and re-register, not to quietly adjust.

---

## Why this exists

The ICTAI result was assembled from choices made after seeing the numbers: an operating
point (P ≥ 0.98) that was never in the grid, the best two of four subgroups quoted as the
headline, and a metric that could not tell a real partition from a shuffled one. Every one
of those was individually defensible and collectively fatal.

The question this run answers — *does the fixed pipeline find a partition that beats
matched chance?* — is worth asking exactly once. So the answer is defined first.

---

## 1. Observation window

**24 hours, anchored on each patient's own earliest event** across their EHR and MED files.

Not a free choice. C1 (#13) established that an EEG-relative cutoff is impossible: no EEG
timestamp exists anywhere in the extract — the suppression-ratio files are bare arrays with
neither a start time nor a documented epoch length. 24 h is the shortest admission-relative
window that keeps most patients assignable (606/671).

The full-record setting is **not** run as a primary arm. #117 showed roughly half the
pooled recall, and two thirds of G2's, is observation-window artefact (G2 recall
0.8163 → 0.3048). A full-record result cannot support a prospective claim, so spending
~5 GPU-hours to produce one is not justified up front.

*Secondary, conditional:* if the 24 h arm PASSES, one full-record arm is worth running to
quantify what the deployable window costs. Not otherwise.

## 2. Split

**`auth_split`**, per #106. Already settled, no discretion here.

## 3. Precision floor

**Primary floor for the gate decision: 0.95.** All four of 1.00 / 0.99 / 0.98 / 0.95 are
evaluated and reported; only 0.95 decides PASS/FAIL.

Two reasons, both independent of the new result — which is the point:

- It is the strictest floor the current routed system demonstrably **meets** on held-out
  data. A5 (#16) found the system fails its own floor at 0.99 and 0.98 (94 TP, 3 FP gives
  0.9691 against a 0.99 requirement). A gate decided at a floor the system cannot reach
  would be measuring the wrong thing.
- It has the most statistical power. The sham distribution at 0.95 is the tightest of the
  four (SD 0.074 vs 0.158 at floor 1.00), so a real effect of a given size is most likely
  to be detected there.

> **Tension Henrik should rule on.** 0.95 precision means roughly 1 in 20 patients flagged
> as non-survivor is wrong. For a signal that may inform withdrawal of life-sustaining
> therapy, that may be clinically indefensible regardless of what it does for statistical
> power, and the paper's whole framing is "high-precision". The alternative is a primary
> floor of **1.00**, which the system does meet (20 TP, 0 FP) at a recall of 0.1156 — but
> its sham distribution is by far the widest, so the test there is the weakest of the four.
> **Recommendation: 0.95 for the gate, with 1.00 reported prominently alongside.** Say if
> you want 1.00 instead.

## 4. top-p

**0.1.** B5 (#15) found the boundary-adjacent phantom framing survives at top-*p* = 0.1 and
dissolves at 0.7, where 70 % of survivors get labelled phantom. Running the primary arm at
the setting that keeps the paper's own construct coherent is the only defensible choice.

## 5. The gate

At the primary floor, using `filter-repository-python/sham_gate.py` (T1/#124):

| parameter | value |
|---|---|
| shams | 2000 |
| seed | 20260915 |
| matching | exact on group count, per-group size, per-group survivor fraction |
| **PASS requires** | **real percentile ≥ 95.0 AND one-sided empirical p ≤ 0.05** |

The partition must **also meet its own precision floor** on held-out data. A partition that
beats the shams but does not achieve 0.95 precision is not a pass — it is a different
finding and must be reported as one.

## 6. What we do in each outcome

**PASS** — the loop finds real structure inside a deployable window. Proceed to B3 (#14) to
test whether objective substitution is what produced it, and write the paper around the
result. Venue per T7 (#130).

**FAIL** — the substituted objective does not produce a partition beating matched chance
under a 24 h window. Close B3 as not-needed. The paper becomes the cautionary methods
paper: the sham control as the contribution, with B1, B2, C1/#117 and A5 as the evidence
that subgroup analyses in clinical ML pass every conventional check and still fail this
one. Target PAKDD.

Neither branch is renegotiated after the fact. In particular, on a FAIL we do **not** go
looking for a floor, a top-*p*, or a subgroup where it passes.

## 7. Number of runs

**Three seeds: 20260916, 20260917, 20260918.** The gate is applied to each independently.

**PASS requires at least 2 of 3.** A 1-of-3 result is reported as a failure to replicate,
not as a win — the LLM loop is stochastic, and a single passing run cannot be
distinguished from a lucky draw.

At ~5 h wall-clock per run (measured: the 24 h run `869q3b4z` took 290 min on 2 GPUs), three
runs is ~15 GPU-hours, spread over two or three nights under the 2-of-8 share.

> **This is the input to the venue decision.** Three runs plus scoring plus writing does not
> fit before **2 October**. It fits comfortably before **PAKDD on 15 November**. If you want
> SAC, the honest options are two seeds instead of three — which weakens the replication
> claim — or accepting that the paper goes to PAKDD. **Recommendation: three seeds, target
> PAKDD.**

---

## Sign-off

Reply on to-do #125 with either **"accept"** to take all seven as written, or the specific
items you want changed. T5 (#128) stays blocked until this is answered.
