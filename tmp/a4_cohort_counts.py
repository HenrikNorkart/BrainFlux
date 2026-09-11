"""A4 - cohort reconciliation: recompute every cohort count from source.

Run from the repository root:

    uv run --with pandas python tmp/a4_cohort_counts.py

Every number printed here is quoted in tmp/a4_cohort_reconciliation.md.
CPU only, a few seconds, no GPU, no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "filter-repository-python"
DATA = FR / "data"
PAPER = ROOT / "paper" / "ictai2026" / "data"

TREE_FEATURES = ("pain_score_mean_x_gcs_score", "sedation_score_mean", "fio2_mean")


def ids(path: Path, col: str) -> set[str]:
    return set(pd.read_csv(path)[col])


def stem_ids(directory: Path, suffix: str = "", split_on: str | None = None) -> set[str]:
    out = set()
    for name in os.listdir(directory):
        if suffix and not name.endswith(suffix):
            continue
        out.add(name.split(split_on)[0] if split_on else name[: -len(suffix)])
    return out


def line(label: str, value) -> None:
    print(f"{label:<58} {value}")


def main() -> None:
    train = ids(DATA / "train.csv", "patient")
    test = ids(DATA / "test.csv", "patient")
    labelled = ids(FR / "labels_combined.csv", "patient")

    sr = stem_ids(DATA / "Suppression Ratio", split_on="_")
    aeeg = stem_ids(DATA / "aEEG", split_on="_")
    ehr = stem_ids(DATA / "EHR", suffix="_EHR.csv")
    any_record = sr | aeeg | ehr

    scores = pd.read_csv(PAPER / "per_patient_scores.csv")
    scored = set(scores.patient_id)
    assigned = scores[scores.group != "Unassigned"]
    unassigned = scores[scores.group == "Unassigned"]

    print("\n== 1. label files (the split axis) ==")
    line("data/train.csv patients", len(train))
    line("data/test.csv patients", len(test))
    line("train n test (must be 0)", len(train & test))
    line("train u test", len(train | test))
    line("labels_combined.csv patients", len(labelled))
    line("labels_combined == train u test", labelled == (train | test))
    line("train share of cohort", f"{len(train)/len(labelled):.1%}")
    line("test share of cohort", f"{len(test)/len(labelled):.1%}")

    print("\n== 2. record availability (the exclusion axis) ==")
    line("distinct patient ids anywhere under data/", len(any_record))
    line("labelled with NO record file of any kind", len(labelled - any_record))
    line("labelled with an EHR file", len(labelled & ehr))
    line("labelled with an aEEG recording", len(labelled & aeeg))
    line("labelled with a Suppression Ratio recording", len(labelled & sr))
    line("labelled with SR AND EHR", len(labelled & sr & ehr))
    line("labelled with EHR but no SR", len((labelled & ehr) - sr))
    line("SR recordings on disk (files, not patients)", len(os.listdir(DATA / "Suppression Ratio")))

    print("\n== 3. scoring and group assignment ==")
    line("patients in per_patient_scores.csv", len(scored))
    line("scored set == labelled n SR", scored == (labelled & sr))
    line("assigned to G1-G4", len(assigned))
    line("Unassigned", len(unassigned))
    has_nan = scores[list(TREE_FEATURES)].isna().any(axis=1)
    line("rows with NaN in any tree split feature", int(has_nan.sum()))
    line("Unassigned set == NaN-in-tree-feature set",
         set(scores.loc[has_nan, "patient_id"]) == set(unassigned.patient_id))
    for f in TREE_FEATURES:
        line(f"  NaN in {f}", int(scores[f].isna().sum()))
    line("1231 - 658 (the manuscript's '573')", len(labelled) - len(assigned))
    line("  = (labelled without SR) + (Unassigned)",
         f"{len(labelled - sr)} + {len(unassigned)} = {len(labelled - sr) + len(unassigned)}")

    print("\n== 4. how the cohort sits across the split ==")
    for name, part in (("data/train", train), ("data/test", test)):
        sub = scores[scores.patient_id.isin(part)]
        sub_a = sub[sub.group != "Unassigned"]
        line(f"{name}: scored", len(sub))
        line(f"{name}: assigned", len(sub_a))
        line(f"{name}: assigned survivors (label 1)", int((sub_a.label == 1).sum()))
        line(f"{name}: assigned non-survivors (label 0)", int((sub_a.label == 0).sum()))
    line("assigned survivors overall (label 1)", int((assigned.label == 1).sum()))
    line("assigned non-survivors overall (label 0)", int((assigned.label == 0).sum()))

    print("\n== 5. the Rogue One discovery datasets ==")
    r_tr = set(pd.read_csv(FR / "train_datasett_rogue_one.csv")["Patient ID"])
    r_te = set(pd.read_csv(FR / "test_datasett_rogue_one.csv")["Patient ID"])
    line("train_datasett_rogue_one.csv rows", len(r_tr))
    line("  == data/train n scored", r_tr == (train & scored))
    line("test_datasett_rogue_one.csv rows", len(r_te))
    line("  == data/test n scored", r_te == (test & scored))

    print("\n== 6. auth_split (the partition the G1-G4 tree was fitted on) ==")
    a_tr = ids(FR / "auth_split" / "train.csv", "patient")
    a_te = ids(FR / "auth_split" / "test.csv", "patient")
    line("auth_split/train.csv patients", len(a_tr))
    line("auth_split/test.csv patients", len(a_te))
    line("auth_split union", len(a_tr | a_te))
    line("labelled patients absent from auth_split", len(labelled - a_tr - a_te))
    line("auth_split patients absent from labels_combined", len((a_tr | a_te) - labelled))
    scored_test = test & scored
    line("scored data/test patients inside auth_split/train", len(scored_test & a_tr))
    line("scored data/test patients inside auth_split/test", len(scored_test & a_te))


if __name__ == "__main__":
    main()
