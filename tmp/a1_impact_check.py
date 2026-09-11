"""A1 side-check: does the threshold-direction fix change any real number?

Replays the OLD (pre-fix) and NEW (post-fix) `_evaluate_thresholds` over the
real per-patient filter scores in oos_per_patient.csv, for the production
setting target_class=0, at each precision floor in the paper's grid.

Run:  python tmp/a1_impact_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1] / "filter-repository-python"
FLOORS = [0.90, 0.95, 0.98, 0.99, 1.00]
TARGET = 0


def old_eval(x, y, t, direction, target_class):
    """Verbatim pre-fix semantics: swapped inequality AND label-coded mask."""
    pred = (x >= t).astype(int) if direction == "under" else (x <= t).astype(int)
    tp = np.sum((y == target_class) & (pred == target_class))
    fp = np.sum((y != target_class) & (pred == target_class))
    fn = np.sum((y == target_class) & (pred != target_class))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(p), float(r)


def new_eval(x, y, t, direction, target_class):
    pos = x <= t if direction == "under" else x >= t
    is_t = y == target_class
    tp = int(np.sum(pos & is_t))
    fp = int(np.sum(pos & ~is_t))
    fn = int(np.sum(~pos & is_t))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r


def sweep(x, y, floor, evaluate):
    """Mimic fit(): 0.001 grid, both directions, max recall s.t. P >= floor."""
    delta = (x.max() - x.min()) / 100.0
    best = (0.0, None, None)
    for k in range(int((x.min() + delta) * 1000), int((x.max() - delta) * 1000)):
        t = k / 1000.0
        for direction in ("under", "over"):
            p, r = evaluate(x, y, t, direction, TARGET)
            if p >= floor and r > best[0]:
                best = (r, t, direction)
    return best


def main():
    df = pd.read_csv(REPO / "oos_per_patient.csv")
    y = df["label"].to_numpy()

    for col in ("score_top", "score_bot"):
        x = df[col].to_numpy(dtype=float)
        n_ties = int(
            np.sum(np.isclose(x * 1000, np.round(x * 1000), rtol=0, atol=1e-9))
        )
        print(f"\n=== {col}  (n={len(x)}, n_pos={int((y == TARGET).sum())}, "
              f"exact 0.001-grid ties={n_ties}) ===")
        print(f"{'floor':>6} | {'recall_old':>10} {'recall_new':>10} | "
              f"{'tau_old':>8} {'tau_new':>8} | dir_old dir_new")
        for floor in FLOORS:
            r_o, t_o, d_o = sweep(x, y, floor, old_eval)
            r_n, t_n, d_n = sweep(x, y, floor, new_eval)
            flag = "" if abs(r_o - r_n) < 1e-12 else "   <-- DIFFERS"
            print(f"{floor:>6.2f} | {r_o:>10.4f} {r_n:>10.4f} | "
                  f"{str(t_o):>8} {str(t_n):>8} | {d_o!s:>7} {d_n!s:>7}{flag}")


if __name__ == "__main__":
    main()
