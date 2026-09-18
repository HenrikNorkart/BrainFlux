"""C1 -- which admission-relative observation cutoffs are viable?

An EEG-relative cutoff is impossible (no EEG start time exists -- see
tmp/c1_anchor_probe.py). The only well-defined alternative is a cutoff measured
from the start of the EHR record, which 662/671 patients have at relative_sec 0.

This measures, for each candidate window, (i) what fraction of each feature's
measurements survive and (ii) how many patients still have a usable value --
because a window that leaves most patients Unassigned is not a usable window.

CPU only. Run from the repository root: python tmp/c1_cutoff_viability.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "filter-repository-python"
EHR = FR / "data" / "EHR"
H = 3600.0
CUTOFFS_H = [12, 24, 48, 72, 96, 168]

PAIN_TEXT_MAP = {"unable to communicate": 0.0,
                 "medication not given for pain": 0.0, ">10": 10.0}
T_PG, T_SED, T_FIO2 = 0.42668621242046356, 1.476269543170929, 51.02120018005371
NAMES = {"pain": {"Pain Score"}, "gcs": {"Glasgow Coma Score"},
         "sed": {"Sedation Score"}, "fio2": {"Oxygen % (FiO2)", "FiO2 - vent"}}


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return PAIN_TEXT_MAP.get(str(x).strip().lower(), np.nan)


def assign(pg, sed, fio2):
    if any(pd.isna(v) for v in (pg, sed, fio2)):
        return "Unassigned"
    if pg > T_PG:
        return "G1"
    if sed > T_SED:
        return "G2"
    if fio2 > T_FIO2:
        return "G3"
    return "G4"


pub = pd.read_csv(ROOT / "paper" / "ictai2026" / "data" / "per_patient_scores.csv")
pids = pub["patient_id"].astype(str).tolist()

rows = []
for i, pid in enumerate(pids):
    p = EHR / (pid + "_EHR.csv")
    if not p.exists():
        continue
    d = pd.read_csv(p, usecols=["name", "result", "relative_sec"], low_memory=False)
    d["v"] = d["result"].map(to_float)
    d["sec"] = pd.to_numeric(d["relative_sec"], errors="coerce")
    rec = {"patient": pid}
    for cut in CUTOFFS_H:
        sub = d[d["sec"] <= cut * H]
        vals = {}
        for key, nm in NAMES.items():
            s = sub.loc[sub["name"].isin(nm), "v"].dropna()
            vals[key] = float(s.mean()) if len(s) else np.nan
            rec[f"n_{key}_{cut}"] = int(len(s))
            rec[f"n_{key}_all"] = int(d.loc[d["name"].isin(nm), "v"].dropna().shape[0])
        pain = 0.0 if np.isnan(vals["pain"]) else vals["pain"]
        rec[f"grp_{cut}"] = assign(pain * vals["gcs"], vals["sed"], vals["fio2"])
    rows.append(rec)
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(pids)}")

t = pd.DataFrame(rows).merge(
    pub[["patient_id", "label", "group"]], left_on="patient", right_on="patient_id")
t.to_csv(ROOT / "tmp" / "c1_cutoff_viability.csv", index=False)

print("\n== measurement retention (share of each feature's rows inside the window) ==")
hdr = f"{'cutoff':>8s} " + " ".join(f"{k:>9s}" for k in NAMES)
print(hdr)
for cut in CUTOFFS_H:
    cells = []
    for key in NAMES:
        kept = t[f"n_{key}_{cut}"].sum()
        tot = t[f"n_{key}_all"].sum()
        cells.append(f"{kept / tot:>8.1%} ")
    print(f"{cut:>6d} h " + " ".join(cells))

print("\n== patients still assignable, and group stability vs published ==")
print(f"{'cutoff':>8s} {'assigned':>9s} {'unassigned':>11s} {'same group':>11s} "
      f"{'G1 kept':>9s} {'G2 kept':>9s}")
g1n = int((t['group'] == 'G1').sum())
g2n = int((t['group'] == 'G2').sum())
for cut in CUTOFFS_H:
    g = t[f"grp_{cut}"]
    una = int((g == "Unassigned").sum())
    same = int((g == t["group"]).sum())
    g1 = int(((t["group"] == "G1") & (g == "G1")).sum())
    g2 = int(((t["group"] == "G2") & (g == "G2")).sum())
    print(f"{cut:>6d} h {len(t) - una:>9d} {una:>11d} {same:>10.1%}  "
          f"{g1 / g1n:>8.1%} {g2 / g2n:>8.1%}")

print("\n== class balance of the re-derived groups at 24 h vs published ==")
for col in ("group", "grp_24"):
    b = t.groupby(col)["label"].agg(n="count", surv="sum")
    b["pct_non_surv"] = ((b["n"] - b["surv"]) / b["n"] * 100).round(1)
    print(f"\n  {col}")
    print(b.to_string())

print("\nwrote tmp/c1_cutoff_viability.csv")
