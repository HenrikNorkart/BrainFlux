"""T4 (HQ #127) -- put the completed 24 h run's discovered features through the sham gate.

The 24 h discovery run (WandB cardiac_arrest_24h / 869q3b4z, 290 min, exit 0) found ten
raw vital-sign aggregates. tmp/b24h_rescore.py already turned them into a partition
(`grp_24disc`) with a 4-leaf tree fitted on TRAIN only against the phantom target, and
re-scored it on the clean split -- but never against the sham control. That is the only
comparison that can say whether the partition beats matched chance.

The published G1-G4 tree is put through the identical gate in the same run, at the same
seed, so the two sit side by side rather than being compared across sessions.

CPU only. From the repository root:  python tmp/t4_24h_sham_precheck.py [--n 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "filter-repository-python"))

from sham_gate import (  # noqa: E402
    DEFAULT_FLOORS,
    DEFAULT_SEED,
    evaluate,
    load_cohort,
    passes,
    published_partition,
    to_frame,
)

FEATURES = ROOT / "tmp" / "b24h_features.csv"


def shape_table(assignment: dict[str, str], cohort: pd.DataFrame) -> pd.DataFrame:
    frame = cohort[cohort["pid"].isin(assignment)].copy()
    frame["group"] = frame["pid"].map(assignment)
    out = frame.groupby("group")["label"].agg(
        n="count", survivors=lambda s: int((s != 0).sum())
    )
    out["survivor_frac"] = (out["survivors"] / out["n"]).round(4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    feats = pd.read_csv(FEATURES)
    discovered = dict(zip(feats["id"].astype(str), feats["grp_24disc"].astype(str)))
    cohort = load_cohort()

    print(f"24 h discovered partition: {len(set(discovered.values()))} groups, "
          f"{len(discovered)} patients assigned")
    print(shape_table(discovered, cohort).to_string(), "\n")
    print("published G1-G4 partition:")
    print(shape_table(published_partition(), cohort).to_string(), "\n")

    results = {}
    for name, assignment in (("discovered_24h", discovered),
                             ("published_G1_G4", published_partition())):
        print(f"--- {name}: {args.n} shams, seed {DEFAULT_SEED} ---", flush=True)
        res = evaluate(assignment, floors=DEFAULT_FLOORS, n_sham=args.n,
                       seed=DEFAULT_SEED, cohort=cohort, progress_every=500)
        table = to_frame(res)
        table.insert(0, "partition", name)
        results[name] = (res, table)
        print(table.to_string(index=False), "\n")

    combined = pd.concat([t for _, t in results.values()], ignore_index=True)
    out_csv = ROOT / "tmp" / "t4_24h_sham_precheck.csv"
    combined.to_csv(out_csv, index=False)

    print("=" * 78)
    for floor in DEFAULT_FLOORS:
        verdicts = {
            name: ("PASS" if passes(res, floor) else "FAIL")
            for name, (res, _) in results.items()
        }
        print(f"floor {floor:.2f}:  " + "   ".join(f"{k}={v}" for k, v in verdicts.items()))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
