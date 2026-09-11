# A4 — Cohort partition reconciliation

Nightshift session, 2026-09-11. HQ to-do #5. Branch `fix/threshold-train-optimization-v2`.
All counts below are reproduced by `tmp/a4_cohort_counts.py`:

```bash
cd /c/code/PHD/BrainFlux
uv run --with pandas python tmp/a4_cohort_counts.py
```

CPU only, a few seconds. No number appears in this document that the script does not print.

---

## 0. Summary — the discrepancy is not an arithmetic error, it is a mislabelling

Every number the to-do lists is individually correct except one (`data/train.csv` has
**731** rows, not 730 — see §1). The counts do reconcile. What does not hold up is the
*description* of what those counts are.

The manuscript presents the cohort as one train/test split:

> "The held-out validation cohort consists of $658$ patients […]; the remaining $573$
> patients form the training partition used by the Rogue One discovery loop and for
> fitting the phantom predictor."
> — `sections/experiment.tex` line 7

There are in fact **two orthogonal axes** in this dataset, and the manuscript has fused
them into one:

| axis | what it is | sizes |
|---|---|---|
| **Split axis** | the actual train/test partition of the 1231 labelled patients | 731 train / 500 test |
| **Exclusion axis** | which patients have a usable suppression-ratio recording and complete EHR features | 658 usable / 573 excluded |

`658` is not a held-out partition. It is the analysable subset, and it **straddles both
sides of the real split**: 393 of the 658 are in `data/train.csv` and 265 are in
`data/test.csv`. `573` is not a training partition either — it is exactly
`1231 − 658`, i.e. the set of patients who were **dropped for missing data** (560 with
no suppression-ratio recording + 13 with incomplete EHR features). Not one of those 573
patients trained anything; they could not have, because they have no filter score.

This matters beyond wording. It is the same defect A3 (#4) found from the other end: if
658 is described as held out, then every number computed on those 658 reads as an
out-of-sample result, and none of them are. See §6.

---

## 1. The `730` in the to-do is an off-by-one

`data/train.csv` has **731** data rows, not 730. The file has no trailing newline, so
`wc -l` reports 731 (header + 731 rows − 1 missing final newline) and a `wc -l` minus
header gives 730.

```console
$ tail -c 1 filter-repository-python/data/train.csv | od -c | head -1
0000000   0                       # last byte is '0', not a newline
$ tail -c 1 filter-repository-python/data/test.csv | od -c | head -1
0000000  \n
```

With the correct count the headline arithmetic closes exactly:

```
731 (train) + 500 (test) = 1231   ==   labels_combined.csv   ==   the manuscript's 1,231
```

`train ∩ test = ∅` (verified), and `labels_combined.csv` is literally
`pd.concat([train, test]).drop_duplicates("patient")` — see
`compute_per_patient_scores.py` §1. So there is **no missing patient**. The
"730 + 500 = 1230 ≠ 1231" problem does not exist.

(For the record, `tmp/a3_clean_split_results.md` §3 says "the 1230 patients in the split
files" and "559 have no score". Correct values are **1231** and **560**. Same off-by-one;
it changes none of A3's conclusions.)

---

## 2. Counts at every stage, with provenance

### 2.1 The split axis

| stage | n | provenance |
|---|---|---|
| labelled patients, training partition | **731** | `data/train.csv`, `len(pd.read_csv(...))` |
| labelled patients, evaluation partition | **500** | `data/test.csv` |
| overlap | **0** | `len(set(train) & set(test))` |
| total labelled cohort | **1231** | union; equals `labels_combined.csv`, built by `compute_per_patient_scores.py` lines 40–45 |

Split proportions: **59.4 % train / 40.6 % test**.

### 2.2 The exclusion axis (applied to all 1231)

| stage | n | provenance |
|---|---|---|
| distinct patient ids with any file under `data/` | 2038 | union of stems in `data/aEEG`, `data/Suppression Ratio`, `data/EHR` |
| — of the 1231 labelled: no record file of any kind | **247** | `labelled − (sr ∪ aeeg ∪ ehr)` |
| labelled with an EHR file | **984** | `labelled ∩ {f[:-8] for f in data/EHR if f.endswith("_EHR.csv")}` |
| labelled with an aEEG recording | 979 | `labelled ∩ aeeg` |
| **labelled with a suppression-ratio recording** | **671** | `labelled ∩ sr` |
| labelled with SR **and** EHR | **671** | `labelled ∩ sr ∩ ehr` — identical set |
| labelled with EHR but no SR | **313** | the modality that actually binds |
| scored by the range filter | **671** | `paper/ictai2026/data/per_patient_scores.csv`, verified `== labelled ∩ sr` |
| assigned to G1–G4 | **658** | `group != "Unassigned"` |
| Unassigned | **13** | `group == "Unassigned"` |

Two things worth stating plainly:

1. **EHR availability is not a filter.** Every one of the 671 patients with an SR
   recording also has an EHR file, and all 984 labelled patients that appear in `data/`
   at all have an EHR file. The manuscript's phrase "complete EEG and EHR records" is
   therefore doing no work on the EHR side — the binding constraint is the
   suppression-ratio recording alone (671 of 1231 = 54.5 %).
2. **It is SR specifically, not EEG generally.** 979 labelled patients have an aEEG
   recording but only 671 have suppression ratio; `sr ⊂ aeeg` strictly. The 2478 SR
   files on disk cover only 765 distinct patients (multiple recordings per patient).

### 2.3 Where the manuscript's `573` comes from

```
1231 − 658 = 573  =  560 (labelled, no SR recording)  +  13 (Unassigned)
```

This is the **exclusion set**, reached by subtraction. There is no file on disk
containing 573 patients, and no model was fitted on them.

### 2.4 The cohort split across the real partition

| | scored | assigned | survivors (label 1) | non-survivors (label 0) |
|---|---|---|---|---|
| `data/train.csv` (731) | 403 | **393** | 142 | 251 |
| `data/test.csv` (500) | 268 | **265** | 80 | 185 |
| **total (1231)** | **671** | **658** | **222** | **436** |

The 222 / 436 totals match `paper/ictai2026/data/fdp_and_recall.csv` row
"Assigned (G1+G2+G3+G4)" (`n=658, n_survivor=222, n_nonsurvivor=436`) exactly, which
confirms the manuscript's class counts are right even though its partition label is not.

Note the label polarity, since it is easy to get backwards: **label 1 = survivor,
label 0 = non-survivor**, and the reported recall targets label 0.

### 2.5 The Rogue One discovery datasets

`train_datasett_rogue_one.csv` has **403** rows and is exactly `data/train ∩ scored`.
`test_datasett_rogue_one.csv` has **268** rows and is exactly `data/test ∩ scored`.
(Both verified by set equality.)

So the discovery loop's training set is **403 patients**, not 573 and not 731. This is
the number that belongs in the sentence about "the training partition used by the Rogue
One discovery loop".

---

## 3. The 13 Unassigned patients — fully explained

`assign_group()` in `compute_per_patient_scores.py` (lines ~172–186) is the trimmed
4-leaf tree and opens with:

```python
p = row["pain_score_mean_x_gcs_score"]
s = row["sedation_score_mean"]
f = row["fio2_mean"]
if any(pd.isna(v) for v in (p, s, f)):
    return "Unassigned"
```

It is a **missing-data guard, not a model outcome**. A patient is Unassigned iff at
least one of the three tree split features is NaN. Verified by set equality: the 13
Unassigned rows are precisely the 13 rows of `per_patient_scores.csv` with a NaN in any
of `pain_score_mean_x_gcs_score`, `sedation_score_mean`, `fio2_mean`.

The missingness, per feature (counts over all 671 scored patients — every NaN in the
file falls in these 13 rows):

| feature | NaN | cause |
|---|---|---|
| `pain_score_mean_x_gcs_score` | 7 | no `Glasgow Coma Score` row in the patient's EHR file (`pain_score_mean` is never NaN — it is imputed to 0.0 for sedated patients, `compute_per_patient_scores.py` lines ~135–139) |
| `sedation_score_mean` | 6 | no `Sedation Score` row |
| `fio2_mean` | 1 | no `Oxygen % (FiO2)` / `FiO2 - vent` row |

(7 + 6 + 1 = 14 > 13 because `PUH-2015-271` is missing both GCS and FiO2.)

The 13, for the record: PUH-2011-007, PUH-2011-198, PUH-2012-031, PUH-2012-058,
PUH-2012-164, PUH-2012-182, PUH-2013-147, PUH-2014-043, PUH-2014-137, PUH-2015-055,
PUH-2015-182, PUH-2015-271, PUH-2016-240.

Split location: **10 of the 13 are in `data/train.csv`, 3 in `data/test.csv`.**
Class balance: **12 of 13 are label 0 (non-survivors)**; only `PUH-2015-271` is a survivor.

### Recommendation on the denominator

**Report `n = 671` as the case-study cohort and carry the 13 Unassigned as an explicit
row, rather than silently reporting 658.**

Reasons:

- The exclusion is a **data-completeness artifact of the EHR extraction, not a property
  of the patient**. Dropping them without saying so removes patients from the
  denominator for a reason unrelated to the phenomenon being measured. A reviewer who
  notices that 658 ≠ 671 will read it as a dropped-hard-cases problem.
- The 13 are **92 % non-survivors versus 66 % in the assigned cohort**, so they are not
  missing at random with respect to the outcome. Their mean filter score (0.344, from
  `subgroup_summary.csv`) is also the highest of any row in that table. Silently
  dropping 12 non-survivors with high filter scores is exactly the kind of exclusion a
  reviewer will flag.
- `fdp_and_recall.csv` already reports both rows ("Full cohort" n=671 and "Assigned"
  n=658), so nothing needs recomputing — the paper simply needs to quote both.
- The difference is small in effect (in-sample recall at P≥0.98: 0.221 on Full cohort vs
  0.135 on Assigned), so there is no cost to honesty here.

Concretely: state the cohort as 671, state that 13 could not be routed by the tree
because of missing EHR fields, report per-group results over the 658 assigned, and keep
the Full-cohort row in every table as the un-excluded comparator.

---

## 4. `658` vs `500` — what each file represents

They are **not comparable quantities** and should never have been put in the same
sentence.

| | `500` | `658` |
|---|---|---|
| file | `data/test.csv` | derived: `per_patient_scores.csv` where `group != "Unassigned"` |
| population | the **evaluation half of the train/test split**, over all labelled patients | the **analysable cohort**, over both halves of the split |
| axis | split | exclusion |
| relationship | 265 of the 658 are in this file | 393 in train + 265 in test |

So `658 ≠ 500` because 658 is not a partition at all. The quantity that *is* comparable
to 500 is **731** (the training partition), and the quantity comparable to 658 within the
test partition is **265**.

---

## 5. Is the evaluation partition larger than the training partition?

**No.** Reviewer 1's objection is based on the manuscript's own wording, and the wording
is wrong.

The real partition is **731 train / 500 test — 59.4 % / 40.6 %**, a conventional and
unremarkable split with the training side larger. Restricted to the analysable cohort it
is **403 / 268** scored, or **393 / 265** assigned — the same ≈60/40 proportion. At no
point in the pipeline is an evaluation set larger than its training set.

The appearance of an inverted split comes entirely from comparing 658 (an exclusion-axis
quantity spanning both partitions) against 573 (its arithmetic complement, i.e. the
discarded patients). Once the two axes are separated the anomaly disappears. This should
be stated explicitly in the response to reviewers, because as written the manuscript
does claim a 658/573 evaluation-heavy split and the reviewer was right to query it.

I found no evidence anywhere in the repository that an evaluation-larger-than-training
split was ever deliberately chosen. It is a reporting error, not a design decision.

---

## 6. What this does *not* fix — carry-over to A3 (#4) and #106

Correcting the wording removes the arithmetic contradiction but makes a different
problem explicit: **once 658 is correctly described as spanning the split, it can no
longer be called "held-out", and the numbers computed on it are in-sample.** That is
precisely A3's finding, arrived at independently.

Two further facts belong in any partition discussion and are reproduced by §6 of the
script:

- `auth_split/` — the partition the G1–G4 tree was actually fitted on — contains
  **1087 train / 725 test = 1812 patients**, a *different and larger* population than
  `data/`'s 1231. **828** auth_split patients are absent from `labels_combined.csv`, and
  the **247** labelled patients absent from auth_split are exactly the 247 that have no
  record file under `data/` at all (verified by set equality).
- Of the 268 scored `data/test.csv` patients, **160 sit inside `auth_split/train.csv`**
  and only 108 inside `auth_split/test.csv`. So the two partitions cross-cut, and group
  assignment remains in-sample for 60 % of the test set.

**This is open question #106 and it is Henrik's to decide, not mine.** Nothing in §7
below presumes an answer: the proposed prose describes the partition that exists today
and stands whichever way #106 is resolved.

---

## 7. Proposed replacement prose

Replacement for the cohort sentence, `paper/ictai2026/sections/experiment.tex` line 7.
**Not applied — `paper/` is off-limits for this session.** Quoted ready to paste:

```latex
We validate our approach on a dataset of $1{,}231$ patients admitted following cardiac
arrest, partitioned once into a training set of $731$ patients and a held-out evaluation
set of $500$. Of these, $671$ ($54.5\%$) have a suppression-ratio EEG recording and are
therefore scorable by the BrainFlux range filter; the remaining $560$ have no such
recording and are excluded. A further $13$ of the $671$ cannot be routed by the sub-group
tree because at least one of its three EHR split features (Glasgow Coma Score, sedation
score, FiO$_2$) is absent from their record, leaving $658$ assigned patients ($436$
non-survivors, $222$ survivors). We report the full scorable cohort of $671$ alongside
the $658$ assigned throughout, so that this exclusion remains visible. The scorable
cohort spans both sides of the partition --- $403$ patients fall in the training set and
$268$ in the evaluation set --- and the Rogue~One discovery loop is run on the $403$
training-side patients only. Complementary EHR data include patient demographics,
clinical descriptors (GCS score, sedation scale, pain response), and medication records.
```

Companion sentence for the first use of $n=658$ in §IV, so no table has to re-explain the
exclusion. Line 31 and the figure/table captions at lines 36 and 65 currently say
"held-out cohort of $658$" and "full case-study cohort ($n=658$)", both of which become
inaccurate under the corrected framing and need "held-out" / "full" struck:

```latex
Here $n=658$ is the assigned sub-cohort, i.e. the $671$ scorable patients minus the $13$
whose EHR record lacks a tree split feature; it is not a held-out partition and spans
both sides of the $731/500$ split.
```

Suggested reviewer-response line:

> Reviewer 1 correctly observed that our stated evaluation partition ($658$) exceeded our
> stated training partition ($573$). This was a reporting error. The $658$ figure is the
> sub-cohort with usable EEG and EHR data, not a partition; $573$ was its arithmetic
> complement, i.e. the excluded patients. The actual split is $731$ training / $500$
> evaluation. Section IV has been rewritten to separate data availability from the
> train/test partition, and all reported numbers now state which partition they were
> computed on.

### Things this prose deliberately does not say

- It does not call the $658$-patient results held-out, because they are not (§6).
- It does not claim the $573$ were used for training, because they were not.
- It leaves the choice of reporting split (#106) open.

---

## 8. Follow-up found while doing this

`train_datasett_rogue_one.csv` and `test_datasett_rogue_one.csv` encode **label 0 as an
empty field**, so `pd.read_csv` yields `NaN` for all 260 non-survivors in the training
file (143 of 403 rows carry `1.0`, the other 260 are `NaN`, none carry `0.0`). If the
discovery loop reads that column without an explicit `fillna(0)`, it either dropped or
mis-labelled every non-survivor it was shown. Filed separately as its own to-do; not part
of A4.

```console
$ uv run --with pandas python -c "import pandas as pd; \
    print(pd.read_csv('train_datasett_rogue_one.csv').Label.value_counts(dropna=False).to_dict())"
{nan: 260, 1.0: 143}
```
