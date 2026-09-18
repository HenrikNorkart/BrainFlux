"""A2 criterion 3 -- how badly was the published precision guarantee violated?

Fits the four-range suppression-ratio configuration (the Table 1 grid, NOT the
single-cell case study) in both selection modes at each floor, and reports the
precision the SYSTEM actually achieves after predict() ORs the cells together.

Fit on auth_split/train, evaluate on auth_split/test -- the split #106 settled on,
and the one the published tree was fitted on.

CPU only. Run from filter-repository-python/:
    python ../tmp/a2_system_precision.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent / "filter-repository-python"
sys.path.insert(0, str(HERE))
os.environ.setdefault("BASE_DATA_PATH", str(HERE / "data"))
os.environ["USE_CACHED_DATA"] = "False"

from brainflux.dataloaders.eeg.numpy_loader import NumpyLoader  # noqa: E402
from brainflux.filters.filters.range_filter import GeneralRangeFilter  # noqa: E402
from brainflux.dataclasses import suppression_ratio  # noqa: E402
from brainflux.aggregators.filter_aggregator import FilterAggregator  # noqa: E402
from brainflux.classifiers.classifiers.single_linear_classifier_1d import (  # noqa: E402
    SingleLinearClassifier1D,
)

TARGET_CLASS = 0
FLOORS = (0.95, 0.98, 0.99, 1.00)
AUTH = HERE / "auth_split"


def aggregate(label_file: Path):
    loader = NumpyLoader(label_file=label_file)
    rf = GeneralRangeFilter(
        data_source=suppression_ratio, num_ranges=4, num_time_divisions=1
    )
    return FilterAggregator(loader=loader, data_filter=rf).aggregate(use_cache=False)


def system_pr(model, data) -> tuple[float, float, int, int, int]:
    predicted = set(model.predict(data).entities)
    is_target = {pid for pid, lab in zip(data.patient_ids, data.labels)
                 if int(lab) == TARGET_CLASS}
    tp = len(predicted & is_target)
    fp = len(predicted - is_target)
    fn = len(is_target - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, tp, fp, fn


print("aggregating train...", flush=True)
train = aggregate(AUTH / "train.csv")
print(f"  train: {train.distribution.shape}, {len(train.patient_ids)} patients")
print("aggregating test...", flush=True)
test = aggregate(AUTH / "test.csv")
print(f"  test : {test.distribution.shape}, {len(test.patient_ids)} patients")

rows = []
for floor in FLOORS:
    for mode in ("per_cell", "system"):
        m = SingleLinearClassifier1D(
            target_class=TARGET_CLASS,
            strong_filter_precision_minimum=floor,
            selection_mode=mode,
        )
        m.fit(train)
        n_cells = int(np.sum(m._thresholds_scores != -1.0))
        trp, trr, *_ = system_pr(m, train)
        tep, ter, tp, fp, fn = system_pr(m, test)
        rows.append(dict(floor=floor, mode=mode, cells_used=n_cells,
                         train_precision=round(trp, 4), train_recall=round(trr, 4),
                         train_meets=trp >= floor - 1e-9,
                         test_precision=round(tep, 4), test_recall=round(ter, 4),
                         test_meets=tep >= floor - 1e-9,
                         tp=tp, fp=fp, fn=fn))
        print(f"floor={floor:.2f} mode={mode:8s} cells={n_cells} "
              f"train P={trp:.4f} R={trr:.4f} meets={trp >= floor - 1e-9} | "
              f"test P={tep:.4f} R={ter:.4f} meets={tep >= floor - 1e-9}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(Path(__file__).resolve().parent / "a2_system_precision.csv", index=False)
print("\n" + out.to_string(index=False))
print("\nwrote tmp/a2_system_precision.csv")
