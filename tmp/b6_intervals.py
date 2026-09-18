"""B6 (HQ #12) -- reissue every A3 precision and recall with counts and 95% CIs.

At P >= 0.98 the pooled system's precision rests on a handful of false positives. A
point estimate of 0.98 from 3 FP events is not a guarantee, and the manuscript does not
currently let a reader see that.

Every cell below carries TP / FP / FN / TN and a Wilson 95% interval (Clopper-Pearson
also computed, in the CSV). Cells resting on fewer than 10 of the rarer event are
flagged FRAGILE.

Both splits are produced: `auth_split` is the split #106 settled on and is primary;
`data` is the split tmp/a3_clean_split_results.md originally used, included so that
document's numbers can be checked cell for cell.

CPU only. Run from the repository root:  python tmp/b6_intervals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "filter-repository-python"
TMP = ROOT / "tmp"
sys.path.insert(0, str(FR))

from brainflux.utils.intervals import (  # noqa: E402
    clopper_pearson_interval,
    is_fragile,
    paired_diff_bootstrap,
    wilson_interval,
)

TARGET = 0
FLOORS = (1.00, 0.99, 0.98, 0.95)
N_BOOT = 10000


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "paper" / "ictai2026" / "data" / "per_patient_scores.csv")
    df["pid"] = df["patient_id"].astype(str)
    base = FR / ("auth_split" if split == "auth_split" else "data")
    tr = set(pd.read_csv(base / "train.csv")["patient"].astype(str))
    te = set(pd.read_csv(base / "test.csv")["patient"].astype(str))
    df["part"] = np.where(df["pid"].isin(tr), "train",
                          np.where(df["pid"].isin(te), "test", "outside"))
    return df[df["part"] != "outside"].reset_index(drop=True)


def select_on_train(score, is_tgt, floor):
    best = None
    for direction in ("under", "over"):
        for tau in np.unique(score):
            pred = score <= tau if direction == "under" else score >= tau
            tp = int(np.sum(pred & is_tgt)); fp = int(np.sum(pred & ~is_tgt))
            if tp + fp == 0 or tp / (tp + fp) < floor - 1e-12:
                continue
            rec = tp / max(int(is_tgt.sum()), 1)
            if best is None or rec > best[0]:
                best = (rec, float(tau), direction)
    return best


def counts(pred, is_tgt):
    tp = int(np.sum(pred & is_tgt)); fp = int(np.sum(pred & ~is_tgt))
    fn = int(np.sum(~pred & is_tgt)); tn = int(np.sum(~pred & ~is_tgt))
    return tp, fp, fn, tn


def row(stratum, floor, tp, fp, fn, tn, split, note=""):
    prec_n, rec_n = tp + fp, tp + fn
    pw = wilson_interval(tp, prec_n) if prec_n else None
    pc = clopper_pearson_interval(tp, prec_n) if prec_n else None
    rw = wilson_interval(tp, rec_n) if rec_n else None
    rc = clopper_pearson_interval(tp, rec_n) if rec_n else None
    return dict(
        split=split, stratum=stratum, floor=floor,
        TP=tp, FP=fp, FN=fn, TN=tn,
        precision=round(pw.point, 4) if pw else np.nan,
        prec_lo=round(pw.low, 4) if pw else np.nan,
        prec_hi=round(pw.high, 4) if pw else np.nan,
        prec_cp_lo=round(pc.low, 4) if pc else np.nan,
        prec_cp_hi=round(pc.high, 4) if pc else np.nan,
        recall=round(rw.point, 4) if rw else np.nan,
        rec_lo=round(rw.low, 4) if rw else np.nan,
        rec_hi=round(rw.high, 4) if rw else np.nan,
        rec_cp_lo=round(rc.low, 4) if rc else np.nan,
        rec_cp_hi=round(rc.high, 4) if rc else np.nan,
        meets_floor=bool(pw and pw.point >= floor - 1e-9),
        fragile_precision=bool(prec_n and is_fragile(tp, prec_n)),
        fragile_recall=bool(rec_n and is_fragile(tp, rec_n)),
        note=note,
    )


def main() -> None:
    all_rows = []
    boot_rows = []

    for split in ("auth_split", "data"):
        df = load(split)
        is_test = (df["part"] == "test").to_numpy()
        scores = df["filter_score"].to_numpy()
        tgt = (df["label"].to_numpy() == TARGET)
        grp = df["group"].to_numpy()
        assigned = grp != "Unassigned"
        print(f"\n### split={split}  train={int((~is_test).sum())} "
              f"test={int(is_test.sum())} assigned={int(assigned.sum())}")

        strata = {"Full cohort": np.ones(len(df), dtype=bool), "Assigned": assigned}
        for g in ("G1", "G2", "G3", "G4"):
            strata[g] = grp == g

        for floor in FLOORS:
            # ---- single-stratum cells ----
            for name, m in strata.items():
                tr_m, te_m = m & ~is_test, m & is_test
                if tr_m.sum() == 0 or te_m.sum() == 0:
                    continue
                sel = select_on_train(scores[tr_m], tgt[tr_m], floor)
                if sel is None:
                    all_rows.append(row(name, floor, 0, 0, int(tgt[te_m].sum()),
                                        int((~tgt[te_m]).sum()), split,
                                        "floor unreachable on train"))
                    continue
                _, tau, d = sel
                pred = (scores <= tau) if d == "under" else (scores >= tau)
                all_rows.append(row(name, floor,
                                    *counts(pred[te_m], tgt[te_m]), split))

            # ---- pooled over G1-G4 (A3's convention) ----
            pos = np.zeros(len(df), dtype=bool)
            for g in ("G1", "G2", "G3", "G4"):
                m = strata[g]
                tr_m, te_m = m & ~is_test, m & is_test
                if tr_m.sum() == 0 or te_m.sum() == 0:
                    continue
                sel = select_on_train(scores[tr_m], tgt[tr_m], floor)
                if sel is None:
                    continue
                _, tau, d = sel
                p = (scores <= tau) if d == "under" else (scores >= tau)
                pos |= (te_m & p)
            am = assigned & is_test
            all_rows.append(row("Pooled G1-G4", floor, *counts(pos[am], tgt[am]),
                                split))

            # ---- criterion 4: routed vs undivided on the SAME patients ----
            und_sel = select_on_train(scores[~is_test], tgt[~is_test], floor)
            if und_sel is not None:
                _, utau, ud = und_sel
                upred = (scores <= utau) if ud == "under" else (scores >= utau)
                # recall is over non-survivors present in the assigned test rows
                units = am & tgt
                a_hit = pos[units].astype(int)
                b_hit = upred[units].astype(int)
                pt, lo, hi, reps = paired_diff_bootstrap(
                    a_hit, b_hit, n_resamples=N_BOOT)
                boot_rows.append(dict(
                    split=split, floor=floor, n_units=int(units.sum()),
                    routed_recall=round(float(a_hit.mean()), 4),
                    undivided_recall=round(float(b_hit.mean()), 4),
                    diff=round(pt, 4), diff_lo=round(lo, 4), diff_hi=round(hi, 4),
                    excludes_zero=bool(lo > 0 or hi < 0), n_resamples=reps))
                print(f"  floor={floor:.2f} routed={a_hit.mean():.4f} "
                      f"undivided={b_hit.mean():.4f} diff={pt:+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}] n={units.sum()}")

    out = pd.DataFrame(all_rows)
    out.to_csv(TMP / "b6_intervals.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TMP / "b6_bootstrap.csv", index=False)

    pd.set_option("display.width", 250)
    for split in ("auth_split", "data"):
        s = out[out["split"] == split]
        print(f"\n=== {split}: precision and recall with Wilson 95% CI ===")
        print(s[["stratum", "floor", "TP", "FP", "FN", "TN", "precision", "prec_lo",
                 "prec_hi", "recall", "rec_lo", "rec_hi", "meets_floor",
                 "fragile_precision", "fragile_recall"]].to_string(index=False))

    n_frag = int((out["fragile_precision"] | out["fragile_recall"]).sum())
    print(f"\nFRAGILE cells (fewer than 10 of the rarer event): {n_frag} of {len(out)}")
    print("\nwrote b6_intervals.csv, b6_bootstrap.csv")


if __name__ == "__main__":
    main()
