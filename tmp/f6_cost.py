"""F6 (HQ #153) AC1: measure the objective's cost before optimising it.

The to-do's arithmetic: get_objective with n_sham=200 calls score_fn 201 times;
get_nested_score fits SingleLinearClassifier1D once per fold with n_folds=5, so
~1005 fits per objective evaluation, and tester.py sweeps 7 thresholds, so ~7035
fits per tester pass.

Measured here rather than assumed, and broken down so AC2 knows where to cut.

Note on AC1's wording: the objective reads Suppression_Ratio EEG aggregates plus
the label CSVs. It never touches the EHR corpus, so its cost is the SAME on the
24h corpus as on the full record. The 24h switch (F1) changes what the LLM sees,
not what the objective costs. Recorded so the budget is not re-derived per corpus.
"""
import sys, time, statistics
sys.path.insert(0, "/workspace/brainflux")
import numpy as np
from external_connectors.brainflux_filter_pipeline import BrainfluxFilterPipeline

t0 = time.perf_counter()
pipe = BrainfluxFilterPipeline(
    data_path="/workspace/phantom_menace/Suppression_Ratio",
    label_path_train="/workspace/phantom_menace/train.csv",
    label_path_test="/workspace/phantom_menace/test.csv",
    target_class=0, min_precision=0.95,
)
t_construct = time.perf_counter() - t0
print(f"pipeline construction (cached aggregates): {t_construct:.2f} s", flush=True)

ids = np.array(list(pipe.all_patient_ids_train))
cls = pipe.get_patient_classes(list(ids))
lab = np.array([cls[p] for p in ids])
rng = np.random.default_rng(4242)
excl = (list(rng.choice(ids[lab == 0], 305, replace=False))
        + list(rng.choice(ids[lab != 0], 141, replace=False)))

# ---- one score_fn call = one get_nested_score = n_folds classifier fits -------
reps = 12
ts = []
for _ in range(reps):
    t = time.perf_counter()
    pipe.get_nested_score(excl, n_folds=5, seed=513930858)
    ts.append(time.perf_counter() - t)
med_score = statistics.median(ts)
print(f"\nget_nested_score (5 folds, 5 fits): median {med_score*1000:.1f} ms "
      f"over {reps} reps  [min {min(ts)*1000:.1f}, max {max(ts)*1000:.1f}]")
print(f"  -> per classifier fit: {med_score/5*1000:.1f} ms")

# ---- one full objective evaluation at the search setting ---------------------
for n_sham in (50, 200):
    t = time.perf_counter()
    pipe.get_objective(excl, n_sham=n_sham, n_folds=5, seed=20260917)
    dt = time.perf_counter() - t
    fits = (n_sham + 1) * 5
    print(f"\nget_objective(n_sham={n_sham}): {dt:.1f} s  "
          f"({fits} fits, {dt/fits*1000:.1f} ms/fit)")
    if n_sham == 200:
        per_eval = dt

# ---- extrapolate to the units the run is scheduled in ------------------------
THRESHOLDS = 7          # tester.py sweeps 0.2..0.8
ITERS = 10              # loop iterations per configuration
print("\n" + "=" * 66)
print("BUDGET")
print(f"  per objective evaluation (n_sham=200) : {per_eval:8.1f} s")
print(f"  per tester pass ({THRESHOLDS} thresholds)      : {per_eval*THRESHOLDS/60:8.1f} min")
print(f"  per configuration ({ITERS} iterations)   : {per_eval*THRESHOLDS*ITERS/3600:8.2f} h")
print(f"  three seeds, one arm                  : {per_eval*THRESHOLDS*ITERS*3/3600:8.2f} h")
print("  (objective only -- excludes LLM, retrieval and XGBoost time)")
