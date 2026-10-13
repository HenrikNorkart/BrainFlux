"""F10 (HQ #157): stamp run provenance on outputs.

AC1 per-run output directory, so no run can overwrite another.
AC2 df_test_pool carries precision_min, tm_frac, objective_seed, corpus path and
    git sha as columns.
AC3 run_manifest.json beside the outputs with the resolved config, both git shas
    and the WandB run id.

The disease this treats is the one that also produced the untracked .env and the
stale config.yml: the things that decide what a result MEANS were not recorded
alongside it. df_test_pool_final.csv has six rows and no column saying which arm
any of them came from.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "phantom_menace_main.py"
s = p.read_text(encoding="utf-8")
orig = s

# ------------------------------------------------------------------- helpers
helper = '''

# --------------------------------------------------------------------------- F10
# HQ #157. cfg.output_dir was a single shared path, so every configuration wrote
# over the last one: intermediate_steps/, tester_reports/ and the three
# *_final.csv files all belonged to whichever cell finished most recently. The
# six rows in df_test_pool_final.csv carried no arm, seed or corpus column, so
# which row came from which configuration was unrecoverable from the artefact.
def _git_sha(repo: Path) -> str:
    """HEAD of ``repo``, with a -dirty suffix if the tree has uncommitted changes."""
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip()
        # A dirty tree is the normal state here, so this must be recorded rather
        # than ignored: a sha alone would claim a reproducibility it lacks.
        return f"{sha}-dirty" if dirty else sha
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
        return f"unavailable ({exc})"


def _run_slug(precision_min: float, tm_frac: float, objective_seed: int) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_id = ""
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb_id = f"_{wandb.run.id}"
    except Exception:  # noqa: BLE001
        pass
    return (
        f"{stamp}_p{precision_min:g}_tm{tm_frac:g}_s{objective_seed}{wandb_id}"
    )


def _write_manifest(path: Path, cfg, precision_min, tm_frac, objective_seed) -> dict:
    """AC3. Everything needed to say what this run WAS, beside its outputs."""
    import json
    from datetime import datetime, timezone

    workspace = Path("/workspace")
    wandb_id = wandb_project = None
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb_id, wandb_project = wandb.run.id, wandb.run.project
    except Exception:  # noqa: BLE001
        pass

    corpus = Path(cfg.feature_file_path)
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "arm": {
            "precision_min": precision_min,
            "tm_frac": tm_frac,
            "objective_seed": objective_seed,
        },
        "corpus": {
            "path": str(corpus),
            "bytes": corpus.stat().st_size if corpus.exists() else None,
        },
        "git": {
            "parent_repo": _git_sha(workspace),
            "brainflux": _git_sha(workspace / "brainflux"),
        },
        "wandb": {"run_id": wandb_id, "project": wandb_project},
        "config": {
            k: str(v)
            for k, v in vars(cfg).items()
            if not k.startswith("_")
        },
        "output_dir": str(path),
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest

'''

anchor = "class RogueOneAgentNetwork"
if anchor not in s:
    anchor = "class PhantomMenaceAgentNetwork"
assert anchor in s, "agent network class not found"
s = s.replace(anchor, helper.lstrip("\n") + "\n" + anchor, 1)

# ------------------------------------------------------------ AC1 + AC3 wiring
old_init = """        self.df = pd.read_csv(self.cfg.feature_file_path)
"""
new_init = """        # F10/AC1: give this run its own directory before anything is written.
        self.cfg.output_dir = Path(self.cfg.output_dir) / _run_slug(
            precision_min, tm_frac, objective_seed
        )
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # F10/AC3: the manifest is written FIRST, so a run that dies mid-way still
        # says what it was. The 2026-09-18 failure left artefacts nobody could
        # attribute to an arm.
        self.run_manifest = _write_manifest(
            self.cfg.output_dir, self.cfg, precision_min, tm_frac, objective_seed
        )
        ConsoleManager.console_print(
            f"Run output directory: {self.cfg.output_dir}", style="bold cyan"
        )

        self.df = pd.read_csv(self.cfg.feature_file_path)
"""
assert old_init in s, "init anchor not found"
s = s.replace(old_init, new_init, 1)

# ------------------------------------------------------------------ AC2 columns
old_save = """    def save_data(self):
        dst_path = self.cfg.output_dir
"""
new_save = """    def save_data(self):
        dst_path = self.cfg.output_dir

        # F10/AC2: stamp the arm onto every row. Without this the pool is six
        # numbers with no way to tell which configuration produced any of them.
        if self.df_test_pool is not None:
            m = getattr(self, "run_manifest", {})
            self.df_test_pool["precision_min"] = self.precision_min
            self.df_test_pool["tm_frac"] = self.tm_frac
            self.df_test_pool["objective_seed"] = self.objective_seed
            self.df_test_pool["corpus_path"] = str(self.cfg.feature_file_path)
            self.df_test_pool["git_sha"] = m.get("git", {}).get("parent_repo")
            self.df_test_pool["git_sha_brainflux"] = m.get("git", {}).get("brainflux")
            self.df_test_pool["wandb_run_id"] = m.get("wandb", {}).get("run_id")
            self.df_test_pool["run_dir"] = str(dst_path)
"""
assert old_save in s, "save_data anchor not found"
s = s.replace(old_save, new_save, 1)

if "\nfrom pathlib import Path\n" not in s and "\nimport pathlib\n" not in s:
    s = s.replace("import asyncio", "import asyncio\nfrom pathlib import Path", 1)

assert s != orig
p.write_text(s, encoding="utf-8")
print("phantom_menace_main.py: per-run dir, manifest, and arm columns wired")
