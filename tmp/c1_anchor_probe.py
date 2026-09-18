"""C1 -- can the EEG/SR recordings be placed on the EHR `relative_sec` clock at all?

Criterion 1 of #13. Two separate questions:
  (a) do per-measurement timestamps exist for pain / GCS / sedation / FiO2?
  (b) is EEG recording start time available, on any clock comparable to (a)?

If (b) fails there is no cut point, and criterion 2 is not executable as written.
This probe also measures how much SR signal each patient has, so the report can
say whether an EEG-start cutoff would have left any EHR data behind even if the
anchor existed.

CPU only. Run from the repository root:  python tmp/c1_anchor_probe.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "filter-repository-python"
EHR = FR / "data" / "EHR"
SR = FR / "data" / "Suppression Ratio"

FEATURE_NAMES = {
    "Pain Score": "pain",
    "Glasgow Coma Score": "gcs",
    "Sedation Score": "sedation",
    "Oxygen % (FiO2)": "fio2",
    "FiO2 - vent": "fio2",
}

# ---------------------------------------------------------------- (a) EHR side
pub = pd.read_csv(ROOT / "paper" / "ictai2026" / "data" / "per_patient_scores.csv")
pids = pub["patient_id"].astype(str).tolist()

print("== (a) per-measurement timestamps on the four feature inputs ==")
cov = defaultdict(lambda: {"patients": 0, "rows": 0, "ts_present": 0,
                           "min_sec": np.inf, "max_sec": -np.inf})
first_sec_all = []
for pid in pids:
    p = EHR / (pid + "_EHR.csv")
    if not p.exists():
        continue
    d = pd.read_csv(p, usecols=["name", "relative_sec"], low_memory=False)
    sec = pd.to_numeric(d["relative_sec"], errors="coerce")
    first_sec_all.append(float(np.nanmin(sec.values)))
    for nm, key in FEATURE_NAMES.items():
        m = d["name"] == nm
        if not m.any():
            continue
        c = cov[key]
        c["patients"] += 1
        c["rows"] += int(m.sum())
        c["ts_present"] += int(sec[m].notna().sum())
        c["min_sec"] = min(c["min_sec"], float(np.nanmin(sec[m].values)))
        c["max_sec"] = max(c["max_sec"], float(np.nanmax(sec[m].values)))

for key, c in cov.items():
    print(f"  {key:9s} rows={c['rows']:>8d}  with timestamp={c['ts_present']:>8d} "
          f"({c['ts_present'] / c['rows']:.2%})  span={c['min_sec'] / 3600:.1f}h "
          f"..{c['max_sec'] / 3600:.1f}h")

fs = np.array(first_sec_all)
print(f"\n  earliest relative_sec per patient: min={fs.min():.1f}s "
      f"median={np.median(fs):.1f}s max={fs.max():.1f}s")
print(f"  patients whose record starts at exactly 0: {(fs == 0).sum()}/{len(fs)}")

# ---------------------------------------------------------------- (b) EEG side
print("\n== (b) EEG / suppression-ratio side ==")
srfiles = sorted(SR.glob("*.npy"))
print(f"  SR files: {len(srfiles)}")
pat = re.compile(r"^(?P<pid>.+)_(?P<idx>\d+)_ar\.npy$")
by_pid = defaultdict(list)
for f in srfiles:
    m = pat.match(f.name)
    if m:
        by_pid[m.group("pid")].append(int(m.group("idx")))
print(f"  distinct patients with SR: {len(by_pid)}")
print(f"  recordings per patient: min={min(len(v) for v in by_pid.values())} "
      f"median={int(np.median([len(v) for v in by_pid.values()]))} "
      f"max={max(len(v) for v in by_pid.values())}")

print("\n  sample arrays (shape, dtype) -- looking for any time axis or metadata:")
for f in srfiles[:5]:
    a = np.load(f, allow_pickle=False)
    print(f"    {f.name:34s} shape={a.shape} dtype={a.dtype} "
          f"min={np.nanmin(a):.3f} max={np.nanmax(a):.3f}")

# does any .npy carry a structured dtype / object array (i.e. metadata)?
structured = 0
for f in srfiles[:300]:
    a = np.load(f, allow_pickle=False)
    if a.dtype.names is not None or a.dtype == object:
        structured += 1
print(f"  arrays with structured/object dtype in first 300: {structured}")

# ------------------------------------------------- (c) would a cutoff bite?
print("\n== (c) SR coverage vs EHR record span (same patients) ==")
EPOCH_SEC = 10.0   # the sibling directory is literally named 'spike_detection (10 sec)'
rows = []
for pid in pids[:400]:
    fs_ = sorted(SR.glob(f"{pid}_*_ar.npy"))
    if not fs_:
        continue
    n = sum(int(np.load(f).shape[0]) for f in fs_)
    p = EHR / (pid + "_EHR.csv")
    if not p.exists():
        continue
    sec = pd.to_numeric(pd.read_csv(p, usecols=["relative_sec"],
                                    low_memory=False)["relative_sec"],
                        errors="coerce")
    rows.append(dict(patient=pid, n_rec=len(fs_), sr_h=n * EPOCH_SEC / 3600.0,
                     ehr_h=float(np.nanmax(sec.values)) / 3600.0))
c = pd.DataFrame(rows)
c["ratio"] = c["sr_h"] / c["ehr_h"].replace(0, np.nan)
print(f"  n={len(c)} patients probed")
print(c[["sr_h", "ehr_h", "ratio"]].describe(percentiles=[.25, .5, .75]).round(2).to_string())
c.to_csv(ROOT / "tmp" / "c1_anchor_probe.csv", index=False)
print("\nwrote tmp/c1_anchor_probe.csv")
