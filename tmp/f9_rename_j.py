"""F9 (HQ #156): stop calling J 'global_recall', and keep the diagnostics.

AC1 rename the field and column to what they carry.
AC3 carry real_score / sham_mean / sham_sd / n_excluded into the pool, so a J can
    be interrogated without re-running anything. Their absence is exactly why F4
    needed a reproduction step: df_test_pool_final.csv recorded J and nothing else.
AC4 a provenance line so an old CSV is never silently read under the new meaning.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "phantom_menace" / "agents" / "tester.py"
s = p.read_text(encoding="utf-8")
orig = s

# ------------------------------------------------------------------ AC1 + AC3
old_dc = '''@dataclass
class TestResultClassification:
    global_recall: float
    loss: float | None = None
    best_threshold: float | None = None
    target_class_patients_removed: int | None = None
    non_target_class_patients_removed: int | None = None
    top_impactful_attributes: dict[str, float] | None = None
    report: str | None = None
'''

new_dc = '''# F9 (HQ #156). CHANGE 1 (2026-09-17) made this field carry J -- the sham-
# standardised objective -- and left the name `global_recall` alone. The pool CSV
# then advertised a recall in [0,1] while holding a z-score of 15.6-16.3, and the
# Scientist selects the next focus of investigation from that pool. The earlier
# 24h run's 0.8053 was already misread once as a paper number when it was the
# Tester's own optimisation signal.
#
# `loss` was likewise repurposed: it carries test_rmse, not a loss in any sense
# the name suggests.
#
# RENAMED 2026-09-23. A df_test_pool_final.csv written before this date has a
# `global_recall` column holding J; one written after has `j`. See
# OBJECTIVE_SCHEMA_VERSION below -- do not read an old CSV under the new meaning.
OBJECTIVE_SCHEMA_VERSION = "2026-09-23"


@dataclass
class TestResultClassification:
    # The sham-standardised objective, (real - sham_mean) / sham_sd. Higher is
    # better. NOT a recall and NOT bounded to [0, 1].
    j: float
    test_rmse: float | None = None
    best_threshold: float | None = None
    target_class_patients_removed: int | None = None
    non_target_class_patients_removed: int | None = None
    top_impactful_attributes: dict[str, float] | None = None
    report: str | None = None
    # AC3: the diagnostics behind `j`. Without these a reported J cannot be
    # interrogated at all without re-running the whole objective -- which is
    # precisely what F4 (#151) had to do.
    real_score: float | None = None
    sham_mean: float | None = None
    sham_sd: float | None = None
    n_sham: int | None = None
    n_excluded: int | None = None
    retained_fraction: float | None = None
    objective_seed: int | None = None
    schema_version: str = OBJECTIVE_SCHEMA_VERSION
'''

assert old_dc in s, "dataclass block not found"
s = s.replace(old_dc, new_dc, 1)

old_td = '''                "global_recall": self.global_recall,
                "loss": self.loss,
                "best_threshold": self.best_threshold,
                "target_class_patients_removed": self.target_class_patients_removed,
                "non_target_class_patients_removed": self.non_target_class_patients_removed,
                "report": self.report,'''

new_td = '''                "j": self.j,
                "test_rmse": self.test_rmse,
                "best_threshold": self.best_threshold,
                "target_class_patients_removed": self.target_class_patients_removed,
                "non_target_class_patients_removed": self.non_target_class_patients_removed,
                "report": self.report,
                # AC3 -- the numbers behind j.
                "real_score": self.real_score,
                "sham_mean": self.sham_mean,
                "sham_sd": self.sham_sd,
                "n_sham": self.n_sham,
                "n_excluded": self.n_excluded,
                "retained_fraction": self.retained_fraction,
                "objective_seed": self.objective_seed,
                # AC4 -- so an old CSV is never read under the new meaning.
                "schema_version": self.schema_version,'''

assert old_td in s, "to_dict block not found"
s = s.replace(old_td, new_td, 1)

# ------------------------------------------------------------- the empty path
s = s.replace(
    'ConsoleManager.console_error_print("No feature columns -- skipping XGBoost, global_recall=0.0")',
    'ConsoleManager.console_error_print("No feature columns -- skipping XGBoost, j=0.0")',
    1,
)
s = s.replace(
    "return TestResultClassification(global_recall=0.0), df_attribute_explanations",
    "return TestResultClassification(j=0.0), df_attribute_explanations",
    1,
)

# --------------------------------------------------------- the reporting path
s = s.replace('out["test_global_recall"] = score', 'out["test_j"] = score', 1)

old_ret = """            return TestResultClassification(
                global_recall=score,
                loss=out["test_rmse"].iloc[0],"""
new_ret = """            return TestResultClassification(
                j=score,
                test_rmse=out["test_rmse"].iloc[0],
                # AC3: carry the objective's own diagnostics through, so the pool
                # answers "why is j that number" without a re-run.
                real_score=(
                    float(best_objective.real_score) if best_objective else None
                ),
                sham_mean=(
                    float(best_objective.sham_mean) if best_objective else None
                ),
                sham_sd=float(best_objective.sham_sd) if best_objective else None,
                n_sham=int(best_objective.n_sham) if best_objective else None,
                n_excluded=(
                    int(best_objective.n_excluded) if best_objective else None
                ),
                retained_fraction=(
                    1.0 - best_objective.n_excluded / len(
                        self._brainflux_filter_pipeline.all_patient_ids_train
                    )
                    if best_objective
                    else None
                ),
                objective_seed=self.objective_seed,"""
assert old_ret in s, "return block not found"
s = s.replace(old_ret, new_ret, 1)

assert s != orig
p.write_text(s, encoding="utf-8")
print("tester.py: j/test_rmse renamed, diagnostics carried, schema stamped")

leftover = [
    (i + 1, ln)
    for i, ln in enumerate(s.splitlines())
    if "global_recall" in ln
]
print(f"remaining 'global_recall' references: {len(leftover)}")
for n, ln in leftover:
    print(f"  {n}: {ln.strip()}")
