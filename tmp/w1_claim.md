# W1 — What the SAC paper claims

Session 2026-09-17. HQ to-do #132. **Framing is Henrik's call — flag anything wrong here
before W3–W6 build on it.**

---

## 1. The claim

> An LLM-driven automated feature-engineering loop discovered a patient stratification that
> passed every conventional validation we could apply — a held-out split, a precision floor,
> executable interpretable features, and a blinded expert survey — and was nonetheless
> indistinguishable from chance once compared against partitions matched on group size and
> class balance.

Everything else in the paper is the evidence for that sentence, or the two things that
survive it.

## 2. Contributions

1. **A complete worked case study.** One cohort (658 post-cardiac-arrest ICU patients)
   carried through conventional validation and then through four independent controls,
   each of which breaks the result. Most negative results report one failed check on one
   axis; the value here is that the same cohort fails on four, and that each failure has a
   named, generalisable mechanism.

2. **Four failure modes, each quantified.**
   - *Class-balance confounding* — the partition does not beat exactly-matched shams
     (percentiles 2.4 / 45.6 / 24.8 / 16.7; p = .98 / .54 / .75 / .83). [B1 #10]
   - *Supervision leakage* — the features claimed to be EEG confounders are strong survival
     predictors in their own right (AUROC .821 / .715 / .709 against the EEG signal's .914),
     so the stratification is partly outcome enrichment. [B2 #11]
   - *Observation-window artefact* — the features average over the whole record. Restricted
     to the first 24 h, the headline subgroup's recall falls .8163 → .3048 and every
     subgroup lands within 3 points of the 66.8 % base rate. Only 12.6–17.3 % of the
     defining measurements exist by then, and a full week still discards ~45 %. There is no
     window both prospective and sufficient. [C1 #13, #117]
   - *Finite-sample fragility* — every reported cell rests on fewer than ten of the rarer
     event; the headline precision rests on three false positives. The system also fails its
     own precision floor on held-out data at 0.99 and 0.98. [B6 #12, A5 #16]

3. **A reusable gate, with the precondition a real failure taught us.** Open
   implementation, reproduces its reference result exactly, and carries a non-degeneracy
   requirement discovered the hard way: a partition that puts 661 of 671 patients in one
   leaf gets compared against equally degenerate shams, collapsing their spread, and can
   score a high percentile while being useless. Percentiles are not comparable across
   partition shapes. [T1 #124, T4 #127]

4. **Expert plausibility did not predict statistical validity.** Five ICU specialists,
   blinded, rated three of four subgroups above matched decoys on interpretability and
   clinical sensibility — for a partition later shown to be chance-equivalent. This is
   probably the paper's most quotable finding and it costs nothing to report, because the
   survey already happened.

5. **One positive result that survives.** No conventional feature-selection baseline —
   shallow tree, L1, gradient boosting with SHAP — selects the pain × GCS interaction, even
   when handed the interaction term directly. The loop finds combinations standard
   selection does not. [B4/B4b #6]

### On novelty — state this plainly rather than let a reviewer say it

Permutation and label-shuffling controls are **standard**, and the paper must say so in the
introduction rather than in a rebuttal. The contribution is not the test. It is (a) the
completeness of the case, (b) the specific observation that for *precision-constrained*
subgroup discovery the control that matters is matching on **class balance**, not just
shuffling labels — because recall at a fixed precision floor gets mechanically easier as
the rarer class is depleted, and (c) the artefact plus its precondition.

**W5 must check whether this control already has an established name** in the subgroup-
discovery literature. If it does, use it and credit it.

## 3. Arc and page budget

ACM `sigconf`, two columns. SAC allows 6–8 pages, +2 at cost, 10 maximum.

| § | Section | pp |
|---|---|---|
| — | Abstract — negative result stated in the first three sentences | .2 |
| 1 | Introduction — setup, the result that looked good, the pivot, contributions | 1.0 |
| 2 | Background and related work — post-arrest EEG prognostication, LLM AutoFE, validation of subgroup discovery, permutation controls in clinical ML | .8 |
| 3 | Pipeline and the original result — self-contained, no reliance on the two under-review references | 1.2 |
| 4 | The controls — the gate specified precisely enough to reimplement, plus the non-degeneracy precondition | .8 |
| 5 | **Results — what each control found. The centrepiece.** | 2.3 |
| 6 | What survives | .5 |
| 7 | Discussion — what would have caught this earlier, and what to adopt | .7 |
| 8 | Limitations and conclusion | .5 |
| — | References | .8 |
| | **total** | **≈ 8.8** |

**DECIDED 2026-09-17 (Henrik): force 8 pages. No paid page.**

So ~0.8pp has to come out, and the decision of *where* is made now rather than at the final
pass — otherwise §5 gets gutted by default, which would cut the evidence the paper exists to
present. The budget above is revised to:

| § | revised | change |
|---|---|---|
| 3 Pipeline | .8 | **−.4** — compress to what a reader needs to follow the controls. Push the full reproduction detail (prompts, generation parameters, retrieval config, exact feature definitions) to the anonymous supplement and cite it. |
| 2 Background | .6 | **−.2** — related work stays, the clinical primer on post-arrest prognostication shrinks to a paragraph. |
| 7 Discussion | .5 | **−.2** — one clear recommendation, not three. |
| 5 Results | 2.3 | unchanged. Protected. |

**Known risk, accepted:** trimming §3 is exactly what Reviewer 2 complained about last time
("relies on two unpublished, under-review papers ... hinders independent verification"). The
mitigation is that the supplement must exist and be referenced explicitly, and that §3 must
still stand alone for *the controls* even if not for the full pipeline. If the supplement
does not materialise, this trim is the first thing to revisit.

## 4. Figures

**Keep**
- Two per-patient SR traces (`sr_signal_example`) — the cheapest way to explain the setup.
- Recall vs precision floor by subgroup (`recall_vs_precision`) — but relabelled as *the
  result being examined*, not as a result. Caption must say in-sample.
- Survey ratings (`survey_scores`) — repositioned per contribution 4.

**Build new — these carry the argument**
- **The sham distribution.** Histogram of the 2000 matched-sham pooled recalls with the real
  value marked, one panel per floor. This is the paper's key figure and does not exist yet.
- **The window collapse.** Subgroup non-survivor rate at full record vs 24 h against the
  base rate — C1 §3c as a figure. Makes the artefact visible in one look.

**Cut** — `cohort_distribution`, `subgroup_distribution`, `filter_score_dist`. Space is the
binding constraint and none of them carry an argument the tables do not.

**Probably keep** — the pruned decision tree, since the paper needs to show the partition it
is taking apart.

## 5. The strongest objection, and the answer

> *"This is one negative result, on one cohort, from a pipeline that had a bug in it. Maybe
> the method is fine and the implementation was not."*

Concede the part that is true: there were two real defects — a fit/predict direction
inversion and a per-cell precision floor applied to OR-ed predictions — and both are
reported in §3 rather than hidden. But the controls that break the result do not depend on
them:

- **B1** runs on the corrected scoring path; the sham comparison is unaffected by the bug.
- **C1** is about *when measurements exist in the record*. No implementation detail changes
  that 12.6–17.3 % of them fall inside 24 h.
- **B2** is a property of the features themselves — they predict survival regardless of what
  the classifier does with them.

And the two headline mechanisms — class-balance confounding and observation-window artefact
— are properties of the *data and the evaluation design*, not of this pipeline. Any method
producing a subgroup result on this kind of cohort inherits both.

The honest residual: this is a single institution and a single filter framework, and the
paper must say so in the contribution statement rather than only in Limitations.

---

## Open questions for Henrik

1. **Buy the 9th page?** My recommendation is yes — see §3.
2. **Is contribution 4 (expert plausibility vs validity) safe to foreground?** It is the
   most interesting finding, but it does implicitly say five named colleagues' clinical
   judgement did not track validity. It is presented as a property of the *method of
   validation*, never of the clinicians — confirm you are comfortable with that framing.
3. **Does the pipeline keep the "Phantom Menace" title?** The phantom construct survives
   only at top-*p* = 0.1 (B5), and the paper is no longer about phantoms.
