"""F1 AC3 -- positive verification that the shipped corpus really is 24 h-windowed.

'The file is there' is not the check. The pre-registration mandates a 24 h
admission-anchored window, so what must be shown is that NO patient carries an
event more than 86400 s after their own earliest one -- per patient, not globally,
because the anchor is each patient's own record start.
"""
import sys

import pandas as pd

path = sys.argv[1]
WINDOW_SEC = 24 * 3600

df = pd.read_csv(path, low_memory=False)
print(f"file        : {path}")
print(f"rows        : {len(df):,}")
print(f"patients    : {df['id'].nunique():,}")
print(f"columns     : {list(df.columns)}")

sec = pd.to_numeric(df["relative_sec"], errors="coerce")
g = df.assign(sec=sec).dropna(subset=["sec"]).groupby("id")["sec"]
span = (g.max() - g.min())

print(f"\nper-patient span (seconds since that patient's own first event):")
print(f"  max   : {span.max():,.1f}  ({span.max() / 3600:.2f} h)")
print(f"  mean  : {span.mean():,.1f}")
print(f"  median: {span.median():,.1f}")

violations = span[span > WINDOW_SEC + 1e-6]
print(f"\npatients exceeding {WINDOW_SEC}s window: {len(violations)}")
if len(violations):
    print(violations.sort_values(ascending=False).head(10).to_string())
    print("\nFAIL: the corpus is not 24 h-windowed")
    sys.exit(1)

print("PASS: every patient's events fall within 24 h of their own first event")
