"""F10/AC4 without a full run: does the provenance machinery keep two arms apart?

A full two-run check costs hours and would exercise the circular objective. What
F10 actually changes is the directory naming, the manifest and the pool columns,
and those can be exercised directly.
"""
import json, sys, types
from pathlib import Path

sys.path.insert(0, "/workspace")
import phantom_menace_main as pmm

base = Path("/tmp/f10_check")
arms = [(0.95, 0.1, 20260916), (0.98, 0.3, 20260917)]

cfg = types.SimpleNamespace(
    output_dir=base,
    feature_file_path="/workspace/phantom_menace/all_merged_24h.csv",
    experiment_name="f10_check",
)

dirs = []
for pmin, tm, seed in arms:
    slug = pmm._run_slug(pmin, tm, seed)
    d = base / slug
    m = pmm._write_manifest(d, cfg, pmin, tm, seed)
    dirs.append(d)
    print(f"arm p={pmin} tm={tm} seed={seed}")
    print(f"   dir      : {slug}")
    print(f"   git      : {m['git']['parent_repo'][:20]}... / "
          f"{m['git']['brainflux'][:20]}...")
    print(f"   corpus   : {Path(m['corpus']['path']).name} "
          f"({m['corpus']['bytes']:,} bytes)" if m['corpus']['bytes'] else "   corpus: missing")

assert dirs[0] != dirs[1], "two arms collided on one directory"
for d in dirs:
    assert (d / "run_manifest.json").exists(), d
    j = json.loads((d / "run_manifest.json").read_text())
    assert j["arm"]["precision_min"] and j["git"]["parent_repo"]

a = json.loads((dirs[0] / "run_manifest.json").read_text())
b = json.loads((dirs[1] / "run_manifest.json").read_text())
assert a["arm"] != b["arm"]
print("\nPASS: distinct directories, both manifests intact, arms distinguishable")
print(f"      dirty-state recorded: {'-dirty' in a['git']['parent_repo']}")
