"""F4: is J~16 quantisation, or circularity?

The exclusion set fed to get_objective is XGBoost's prediction of the
InvSqDistTroublemaker target. That target comes from get_biomarker_loss, which
scores a patient by how badly THE STRONG FILTER ITSELF misclassifies them.

So the search removes the classifier's own errors and then measures the
classifier's recall. Matched shams remove random patients, which cannot help.

Test: skip the LLM and XGBoost entirely. Build the exclusion set DIRECTLY from
get_biomarker_loss (the oracle the surrogate is trying to approximate), matched
to Trial 1's class counts. If J lands near 16, J~16 is circularity, and no
denominator floor would fix it.
"""
import sys, numpy as np
sys.path.insert(0, "/workspace/brainflux")
from external_connectors.brainflux_filter_pipeline import BrainfluxFilterPipeline

pipe = BrainfluxFilterPipeline(
    data_path="/workspace/phantom_menace/Suppression_Ratio",
    label_path_train="/workspace/phantom_menace/train.csv",
    label_path_test="/workspace/phantom_menace/test.csv",
    target_class=0, min_precision=0.95,
)
ids = np.array(list(pipe.all_patient_ids_train))
cls = pipe.get_patient_classes(list(ids))
lab = np.array([cls[p] for p in ids])

loss = pipe.get_biomarker_loss(split="train")
lv = np.array([loss.get(p, 0.0) for p in ids])
print(f"biomarker loss: {int((lv>0).sum())} of {len(ids)} patients misclassified "
      f"by the strong filter", flush=True)

for name, n_t, n_o in [("Trial 1", 305, 141), ("Trial 9", 221, 100), ("Trial 10", 252, 124)]:
    excl = []
    for want, mask in ((n_t, lab == 0), (n_o, lab != 0)):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-lv[idx])][:want]   # worst offenders first
        excl += ids[idx].tolist()
    r = pipe.get_objective(excl, n_sham=200, n_folds=5, seed=20260917)
    print(f"\n=== {name} sizes, ORACLE (highest biomarker loss) exclusion ===")
    print(f"  real_score {r.real_score:.6f}  sham_mean {r.sham_mean:.6f}  "
          f"sham_sd {r.sham_sd:.3e}")
    print(f"  J = {r.j:.6f}      (reported: 15.61 .. 16.31)")
    s = np.asarray(r.sham_scores)
    print(f"  distinct shams {len(np.unique(s))}/{len(s)}; "
          f"real above all shams: {bool(r.real_score > s.max())}", flush=True)
