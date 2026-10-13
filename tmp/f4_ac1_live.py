"""F4/AC1: run the real objective and dump the sham distribution.

bf-newobj recorded only J. This re-evaluates the objective directly, with no LLM
and no GPU, on exclusion sets matched in SIZE to the six reported trials but drawn
AT RANDOM. If random exclusions of the same size also score J ~ 16, then J = 16 is
a property of the statistic, not evidence that the discovered features found
anything.
"""
import sys, numpy as np
sys.path.insert(0, "/workspace/brainflux")

from external_connectors.brainflux_filter_pipeline import BrainfluxFilterPipeline

pipe = BrainfluxFilterPipeline(
    data_path="/workspace/phantom_menace/Suppression_Ratio",
    label_path_train="/workspace/phantom_menace/train.csv",
    label_path_test="/workspace/phantom_menace/test.csv",
    target_class=0,
    min_precision=0.95,
)

train_ids = list(pipe.all_patient_ids_train)
classes = pipe.get_patient_classes(train_ids)
lab = np.array([classes[p] for p in train_ids])
ids = np.array(train_ids)
print(f"train={len(ids)}  target(class 0)={int((lab==0).sum())}  other={int((lab!=0).sum())}",
      flush=True)

TRIALS = [("Trial 1", 305, 141), ("Trial 9", 221, 100), ("Trial 10", 252, 124)]
rng = np.random.default_rng(4242)

for name, n_t, n_o in TRIALS:
    t_pool = ids[lab == 0]; o_pool = ids[lab != 0]
    excl = list(rng.choice(t_pool, n_t, replace=False)) + \
           list(rng.choice(o_pool, n_o, replace=False))
    print(f"\n=== {name}-sized RANDOM exclusion: {n_t} target + {n_o} other ===", flush=True)
    r = pipe.get_objective(excl, n_sham=200, n_folds=5, seed=20260917)
    print(f"  real_score = {r.real_score:.10f}")
    print(f"  sham_mean  = {r.sham_mean:.10f}")
    print(f"  sham_sd    = {r.sham_sd:.3e}")
    print(f"  J          = {r.j:.6f}   <-- reported range was 15.61 .. 16.31")
    s = np.asarray(r.sham_scores)
    u, c = np.unique(np.round(s, 12), return_counts=True)
    print(f"  distinct sham values: {len(u)} of {len(s)}")
    order = np.argsort(-c)[:8]
    for i in order:
        print(f"     {u[i]:.10f}  x{c[i]}")
    print(f"  excess (real-mean) = {r.real_score - r.sham_mean:.6f} recall pts "
          f"({(r.real_score-r.sham_mean)*100:.3f} pp)")
    print(f"  empirical percentile of real = "
          f"{100.0*np.mean(s <= r.real_score):.2f}   "
          f"p = {(1+np.sum(s >= r.real_score))/(1+len(s)):.4f}", flush=True)
