# C1 — Observation cutoff for the EHR features

Session 2026-09-15. HQ to-do #13. CPU only, no GPU, no network.

```bash
cd /c/code/PHD/BrainFlux
python tmp/c1_anchor_probe.py        # criterion 1     -> tmp/c1_anchor_probe.csv
python tmp/c1_cutoff_viability.py    # criteria 2/3/4  -> tmp/c1_cutoff_viability.csv
```

---

## Summary

Criterion 1 splits two ways, and the answer is different on each side:

- **Per-measurement timestamps DO exist** for all four feature inputs. Every one of the
  23 277 pain, 32 838 GCS, 25 616 sedation and 99 283 FiO2 rows carries a populated
  `relative_sec`. Coverage is 100.00 %, not merely high.
- **EEG recording start time does NOT exist**, in any form, on any clock. There is
  nothing to cut *at*.

So criterion 2 is **not executable as written** — "restricted to measurements at or
before EEG start" has no computable right-hand side — and criterion 3 applies. But
because the EHR timestamps are complete, an *admission-relative* cutoff is well defined,
and that substitute is worth far more than documenting an absence. §3 runs it.

**The result is devastating, and it is worse than C2 (#7) indicated.** C2 showed the
partition moves under a 24 h window. The stronger finding here is that the partition
stops *discriminating*: at a 24 h cutoff every subgroup's non-survivor rate collapses to
within a few points of the cohort base rate. The stratification the paper reports is an
artifact of full-record observation.

**The paper's performance claims do not survive as prospective claims** (criterion 4,
one sentence, §5).

---

## 1. Criterion 1 — the evidence, both sides

### 1a. EHR timestamps: present, complete

`data/EHR/*_EHR.csv` is long-format `id, name, result, unit, normalized_id, relative_sec`.
`relative_sec` is per measurement, not per patient:

| feature input | rows (over the 671 scored patients) | with a timestamp | observed span |
|---|---|---|---|
| `Pain Score` | 23 277 | **23 277 (100.00 %)** | 0 – 2174 h |
| `Glasgow Coma Score` | 32 838 | **32 838 (100.00 %)** | 0 – 2198 h |
| `Sedation Score` | 25 616 | **25 616 (100.00 %)** | 0 – 2094 h |
| `Oxygen % (FiO2)` + `FiO2 - vent` | 99 283 | **99 283 (100.00 %)** | 0 – 2042 h |

The clock origin is per patient and consistent: **662 of 671 records begin at exactly
`relative_sec = 0`**, and `all_merged.csv` carries `age` at `relative_sec = 0`, so the
origin is the admission/registration anchor rather than an arbitrary offset.

There is therefore **no technical obstacle to an observation cutoff**. The published
features simply do not apply one.

### 1b. EEG start time: absent, and not recoverable

| where a timestamp could live | what is actually there |
|---|---|
| the SR files themselves | bare 1-D `float32` arrays. `PUH-2010-061_1_ar.npy` is `shape=(116000,)`. Checked 300 files: **0** have a structured or object dtype, so no embedded metadata |
| a sidecar or index file | none. `data/Suppression Ratio/` and `data/aEEG/` contain `.npy` files and nothing else |
| the filename | `<patient>_<n>_ar.npy`. The index `n` gives recording *order* (1–30 per patient), never a time |
| the loader | `NumpyLoader._load` (`brainflux/dataloaders/eeg/numpy_loader.py:16`) constructs `EEGData(data=data)` and sets nothing else |
| the dataclass | `EEGData` (`brainflux/dataclasses/eeg.py:8`) has no start-time field at all — only `data`, `channel_names`, `subject_id`, `session_id`, `sampling_rate`, `duration`, `label` |
| `sampling_rate` | never set for real data. The only assignment in the repo is `sampling_rate=256` in `dummy_loader.py:55`. `duration` is derived from it, so `duration` is always `None` too |

The consequence is stronger than "start time is missing": **recording duration is not
recoverable either**, because the SR epoch length is not recorded anywhere. Assuming the
10 s epoch of the sibling `spike_detection (10 sec)` directory yields SR spans a median
of **2.97×** the patient's entire EHR record (IQR 1.04–5.64, max 9.66) — so 10 s is the
wrong epoch for SR, and the right one is not written down. We can locate the SR signal
neither in absolute time nor relative to admission.

**Nothing in this dataset can align the EEG to the EHR clock.** That is not a processing
gap to be closed; it needs a new extract from the source system.

---

## 2. What that forecloses

The specific question Reviewer 1 asked — do the EHR features contain information
recorded *after* the EEG interval being classified — **cannot be answered**, and the
fraction of post-EEG measurements per feature (criterion 2, second half) cannot be
computed. Given §1a, the honest expectation is that the fraction is large: the features
are means over records whose median length is 78 h for non-survivors and 408 h for
survivors, and the SR recordings sit somewhere inside that, unlocatable.

---

## 3. The substitute: an admission-relative cutoff

Since the EHR clock is complete, the three tree features were recomputed inside windows
measured from record start, and the published thresholds re-applied
(`tmp/c1_cutoff_viability.py`).

### 3a. Measurement retention — the features are overwhelmingly late-record

Share of each feature's measurements falling inside the window:

| cutoff | pain | GCS | sedation | FiO2 |
|---|---|---|---|---|
| 12 h | 7.3 % | 6.0 % | 7.6 % | 9.4 % |
| **24 h** | **12.8 %** | **12.7 %** | **12.6 %** | **17.3 %** |
| 48 h | 23.0 % | 24.6 % | 21.2 % | 30.0 % |
| 72 h | 31.6 % | 34.4 % | 28.6 % | 40.0 % |
| 96 h | 38.4 % | 42.2 % | 35.0 % | 47.7 % |
| 168 h (one week) | 53.0 % | 57.3 % | 50.1 % | 61.7 % |

A full week of observation still discards ~45 % of the data the published means are
built from. There is no window short enough to be prospective and long enough to
reproduce the published features.

### 3b. Assignability and stability

| cutoff | assignable | Unassigned | same group as published | G1 retained | G2 retained |
|---|---|---|---|---|---|
| 12 h | 527 | 144 | 40.5 % | 6.1 % | 62.5 % |
| **24 h** | **606** | **65** | **53.5 %** | **12.8 %** | **82.1 %** |
| 48 h | 632 | 39 | 63.0 % | 24.4 % | 91.1 % |
| 72 h | 649 | 22 | 69.3 % | 34.8 % | 93.6 % |
| 96 h | 652 | 19 | 75.6 % | 45.1 % | 96.4 % |
| 168 h | 653 | 18 | 83.3 % | 57.9 % | 97.9 % |

(The 24 h row reproduces C2's independent figure of 53.5 % exactly.)

### 3c. The finding that settles it — discrimination collapses

Stability is the lesser question. What matters is whether the subgroups still separate
the outcome. Cohort base rate: **448 / 671 = 66.8 % non-survivors**.

| group | published: n | published: % non-survivor | at 24 h: n | at 24 h: % non-survivor |
|---|---|---|---|---|
| G1 | 164 | **18.3 %** | 21 | 38.1 % |
| G2 | 280 | **94.6 %** | 398 | **67.6 %** |
| G3 | 116 | **86.2 %** | 138 | **69.6 %** |
| G4 | 98 | **41.8 %** | 49 | 65.3 % |
| Unassigned | 13 | 92.3 % | 65 | 66.2 % |

Published, the groups span 18.3 %–94.6 % — a 76-point spread, which is the entire
stratification result. At a 24 h cutoff they span 65.3 %–69.6 % excluding G1, i.e.
**every group sits within 3 points of the 66.8 % base rate**. G1 retains some signal
(38.1 %) on 21 patients.

The partition does not merely shift under a deployable observation window. It stops
carrying prognostic information.

---

## 4. Ready-to-paste text (criterion 3)

> **Temporal validity.** The engineered EHR features are unweighted means over each
> patient's complete clinical record. The extract timestamps every measurement relative
> to admission, but it contains no recording time for the EEG signals — the suppression-
> ratio series are stored as unlabelled arrays with neither a start time nor a documented
> epoch length — so we cannot determine which measurements precede the EEG interval being
> classified, and cannot report the proportion of each feature that is post-EEG. We can
> bound the concern from the other direction, using an admission-relative window. Only
> 12.6–17.3 % of the measurements contributing to these features fall within 24 hours of
> admission, and a full week still excludes roughly 45 % of them. Recomputed within a
> 24-hour window, the discovered subgroups retain 53.5 % of their assignments and, more
> importantly, cease to separate the outcome: non-survivor rates across the subgroups
> compress from a range of 18.3–94.6 % to 65.3–69.6 %, against a cohort base rate of
> 66.8 %. We therefore present this case study as retrospective stratification and cohort
> audit — a demonstration that the discovery framework recovers structure a clinician
> finds interpretable in a completed record — and explicitly not as evidence of
> prospective prognostic performance, which this data cannot establish.

---

## 5. Criterion 4 — one sentence

**No: the paper's performance claims do not survive as prospective claims, because the
subgroups that produce them lose essentially all outcome discrimination (76-point spread
→ 4-point spread, against a 66.8 % base rate) once the features are restricted to a
deployable observation window.**

---

## 6. What follows

1. **#117 (re-score under a fixed window) should use an admission-relative cutoff, not an
   EEG-relative one** — the latter is impossible. That item's criterion 1 said "use the
   cutoff C1 settles on": it is **24 h from record start**, on clinical grounds (the
   window a post-arrest prognostication decision actually occupies) and because §3b shows
   it is the shortest window that keeps most patients assignable (606/671).
2. **Re-scoring the existing tree is no longer sufficient.** §3c shows the published
   thresholds applied to 24 h features produce groups that do not discriminate. The tree
   was *fitted* on full-record features; it cannot be expected to transfer. The
   discovery loop needs to be re-run with the cutoff applied at feature-extraction time,
   so it can find whatever structure exists in the first 24 h — if any.
3. **#109 gates that re-run.** If the loop's label encoding was corrupt, re-running now
   repeats the defect at GPU cost. Settle it first — it is a CPU-only trace.
4. The B4 (#6) baseline comparison would need re-running against 24 h features too,
   since it currently compares LLM and non-LLM arms on full-record data.
