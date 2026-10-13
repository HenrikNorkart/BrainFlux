"""Are the 'quality' signals in the SR arrays actually quality, or prognosis?

Option (a) needs a metric that says "this recording is uninterpretable" WITHOUT
speaking to the outcome. A flat or all-zero suppression-ratio trace looks like a
dead channel -- but SR=0 throughout means no suppression detected, which is a
GOOD prognostic sign, and SR=100 throughout means complete suppression, a bad one.
If these correlate with outcome, they are signal, not quality, and using them as
an eligibility criterion smuggles the outcome into the selection target.
"""
import glob, os
import numpy as np
import pandas as pd

lab = pd.concat([
    pd.read_csv("/workspace/phantom_menace/train.csv"),
    pd.read_csv("/workspace/phantom_menace/test.csv"),
])
labels = dict(zip(lab["patient"].astype(str), lab["label"].astype(float)))

rows = []
for p in sorted(glob.glob("/workspace/phantom_menace/Suppression_Ratio/*.npy")):
    pid = os.path.basename(p).rsplit("_", 1)[0]
    if pid not in labels:
        continue
    v = np.asarray(np.load(p, allow_pickle=True), dtype=float).ravel()
    if v.size == 0:
        continue
    rows.append({
        "pid": pid, "label": labels[pid], "n": v.size,
        "zero_frac": float(np.mean(v == 0)),
        "rng": float(v.max() - v.min()),
        "mean_sr": float(v.mean()),
        "all_zero": bool(np.all(v == 0)),
        "constant": bool(v.max() == v.min()),
    })
df = pd.DataFrame(rows)
print(f"recordings matched to a label: {len(df)}  patients: {df['pid'].nunique()}")


def auroc(y, s):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    r = np.empty(len(s)); base = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = base[i:j + 1].mean(); i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


y = (df["label"].to_numpy() == 1.0).astype(int)
print("\nIs each candidate 'quality' metric predictive of the OUTCOME?")
print(f"{'metric':<16}{'AUROC':>9}{'separation':>12}")
for c in ("zero_frac", "rng", "mean_sr", "n"):
    a = auroc(y, df[c].to_numpy(float))
    print(f"{c:<16}{a:>9.4f}{abs(2*a-1):>12.4f}")

print("\nDegenerate traces:")
for flag in ("all_zero", "constant"):
    sub = df[df[flag]]
    print(f"  {flag}: {len(sub)} recordings")
    if len(sub):
        for v in (0.0, 1.0):
            n = int((sub['label'] == v).sum())
            base = int((df['label'] == v).sum())
            print(f"      label {v}: {n}/{base} ({100*n/base:.1f}% of that class)")
