"""#182 groundwork: does CROSS-FITTING break the circularity?

The circular target is get_biomarker_loss: 0 where the strong filter is correct,
1/d^2 where it is wrong, computed from a filter fitted on ALL the train patients
including the one being scored. The objective then refits that filter and measures
its recall with the high-loss patients deleted.

The textbook repair is cross-fitting: score each patient with a filter that never
saw them. If the boundary is stable across folds the repair does nothing, because
the out-of-fold filter makes almost the same mistakes as the in-fold one -- and
then #182 needs a genuinely different target, not a cleaner version of this one.

Three arms, all at identical exclusion sizes so the matched shams are identical:
  random       size-matched random exclusion            (honest baseline)
  in-fold      top-loss patients, filter fitted on all  (what the run does)
  out-of-fold  top-loss patients, filter never saw them (the proposed repair)
"""
import sys

import numpy as np

sys.path.insert(0, "/workspace/brainflux")

from brainflux.classifiers import SingleLinearClassifier1D
from brainflux.transform import PatientRemover
from external_connectors.brainflux_filter_pipeline import BrainfluxFilterPipeline

TARGET, PRECISION = 0, 0.95
pipe = BrainfluxFilterPipeline(
    data_path="/workspace/phantom_menace/Suppression_Ratio",
    label_path_train="/workspace/phantom_menace/train.csv",
    label_path_test="/workspace/phantom_menace/test.csv",
    target_class=TARGET, min_precision=PRECISION,
)
ids = np.array(list(pipe.all_patient_ids_train))
cls = pipe.get_patient_classes(list(ids))
lab = np.array([cls[p] for p in ids])
print(f"train {len(ids)}  target {int((lab == TARGET).sum())}  "
      f"other {int((lab != TARGET).sum())}", flush=True)


def out_of_fold_loss(n_folds=5, seed=513930858):
    """Biomarker loss for every patient, from a filter that never saw them."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    folds = np.array_split(order, n_folds)
    loss = {}
    for held in folds:
        held_ids = set(ids[held].tolist())
        # Fit on everyone EXCEPT this fold.
        fit_view = PatientRemover().transform(
            pipe.scores_train, patient_id_list=list(held_ids)
        )
        # Score ONLY this fold.
        score_view = PatientRemover().transform(
            pipe.scores_train,
            patient_id_list=[p for p in ids.tolist() if p not in held_ids],
        )
        clf = SingleLinearClassifier1D(
            target_class=TARGET, strong_filter_precision_minimum=PRECISION
        )
        clf.fit(fit_view)
        predicted = set(clf.predict(score_view).entities)
        d = clf.calculate_distance(score_view.distribution)
        per = d.reshape(d.shape[0], -1).min(axis=1)
        for i, pid in enumerate(score_view.patient_ids):
            is_t = score_view.labels[i] == TARGET
            correct = (pid in predicted) == is_t
            if correct:
                loss[pid] = 0.0
            else:
                dd = float(per[i])
                loss[pid] = 1.0 / (dd * dd) if dd > 1e-9 else 1.0e9
    return loss


in_fold = pipe.get_biomarker_loss(split="train")
oof = out_of_fold_loss()

iv = np.array([in_fold.get(p, 0.0) for p in ids])
ov = np.array([oof.get(p, 0.0) for p in ids])
print(f"\nmisclassified in-fold     : {int((iv > 0).sum())}")
print(f"misclassified out-of-fold : {int((ov > 0).sum())}")
agree = int(((iv > 0) == (ov > 0)).sum())
print(f"agreement on WHO is wrong : {agree}/{len(ids)} ({agree/len(ids):.1%})")
print("  -> if this is near 100%, cross-fitting cannot help: the out-of-fold")
print("     filter makes the same mistakes, so the target is still the filter's.",
      flush=True)


def topn(vals, n_t, n_o):
    out = []
    for want, mask in ((n_t, lab == TARGET), (n_o, lab != TARGET)):
        idx = np.where(mask)[0]
        out += ids[idx[np.argsort(-vals[idx])][:want]].tolist()
    return out


rng = np.random.default_rng(4242)
for name, n_t, n_o in [("Trial 1", 305, 141), ("Trial 10", 252, 124)]:
    print(f"\n=== {name} sizes ({n_t} target + {n_o} other) ===", flush=True)
    arms = {
        "random": (list(rng.choice(ids[lab == TARGET], n_t, replace=False))
                   + list(rng.choice(ids[lab != TARGET], n_o, replace=False))),
        "in-fold  (what the run does)": topn(iv, n_t, n_o),
        "out-of-fold (proposed repair)": topn(ov, n_t, n_o),
    }
    for arm, excl in arms.items():
        r = pipe.get_objective(excl, n_sham=200, n_folds=5, seed=20260917)
        print(f"  {arm:<32} real {r.real_score:.4f}  sham_sd {r.sham_sd:.3e}  "
              f"J {r.j:+.3f}", flush=True)
