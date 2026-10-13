"""Is a signal-quality metric derivable from the bare Suppression_Ratio arrays?

Option (a) for #182 selects exclusions on whether a recording is INTERPRETABLE.
That requires something in the trace itself that speaks to quality, independent of
the classifier and of the outcome. This checks what is actually there.
"""
import glob
import numpy as np

files = sorted(glob.glob("/workspace/phantom_menace/Suppression_Ratio/*.npy"))
print(f"files: {len(files)}")

a = np.load(files[0], allow_pickle=True)
print(f"\nexample {files[0].split('/')[-1]}")
print(f"  shape {a.shape}  dtype {a.dtype}  ndim {a.ndim}")
print(f"  first 12: {np.asarray(a).ravel()[:12]}")

lens, nan_frac, flat_frac, zero_frac, rng_ = [], [], [], [], []
for p in files[:600]:
    v = np.asarray(np.load(p, allow_pickle=True), dtype=float).ravel()
    lens.append(v.size)
    if v.size == 0:
        continue
    nan_frac.append(float(np.mean(~np.isfinite(v))))
    f = v[np.isfinite(v)]
    if f.size == 0:
        continue
    zero_frac.append(float(np.mean(f == 0)))
    d = np.diff(f)
    flat_frac.append(float(np.mean(d == 0)) if d.size else 1.0)
    rng_.append(float(f.max() - f.min()))

def stat(name, xs):
    xs = np.asarray(xs, float)
    print(f"  {name:<22} min {xs.min():9.4f}  med {np.median(xs):9.4f}  "
          f"max {xs.max():9.4f}  sd {xs.std():8.4f}")

print(f"\nover {len(lens)} recordings:")
stat("length (samples)", lens)
stat("non-finite fraction", nan_frac)
stat("exact-zero fraction", zero_frac)
stat("flat (zero-diff) frac", flat_frac)
stat("value range", rng_)

print("\nlength distribution (the censoring axis):")
L = np.asarray(lens, float)
for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95):
    print(f"  q{q:.2f}  {np.quantile(L, q):,.0f}")
print(f"  distinct lengths: {len(set(lens))}")
