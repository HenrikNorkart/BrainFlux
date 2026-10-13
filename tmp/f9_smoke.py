import sys
sys.path.insert(0, "/workspace")
from phantom_menace.agents.tester import TestResultClassification, OBJECTIVE_SCHEMA_VERSION

r = TestResultClassification(
    j=15.6, test_rmse=1.2, real_score=1.0, sham_mean=0.57, sham_sd=0.04,
    n_sham=200, n_excluded=446, retained_fraction=0.59, objective_seed=20260917,
)
d = r.to_dict()
print("schema:", OBJECTIVE_SCHEMA_VERSION)
print("keys  :", sorted(d))
assert "global_recall" not in d and "loss" not in d
for k in ("j", "real_score", "sham_mean", "sham_sd", "n_sham", "n_excluded",
          "retained_fraction", "objective_seed", "schema_version", "test_rmse"):
    assert k in d, k
print("OK: j carried, diagnostics present, old names gone")
