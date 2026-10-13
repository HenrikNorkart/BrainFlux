"""F10 follow-up: git is not installed in brainflux_pm, so _git_sha degraded to
'unavailable' -- exactly the silent-provenance failure F10 exists to prevent.

Take the sha from the environment when present (the launcher captures it on the
HOST, where git exists) and fall back to the subprocess call otherwise.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "phantom_menace_main.py"
s = p.read_text(encoding="utf-8")

old = '''def _git_sha(repo: Path) -> str:
    """HEAD of ``repo``, with a -dirty suffix if the tree has uncommitted changes."""
    import subprocess

    try:'''

new = '''def _git_sha(repo: Path, env_key: str | None = None) -> str:
    """HEAD of ``repo``, with a -dirty suffix if the tree has uncommitted changes.

    `git` is NOT installed in brainflux_pm, so calling it inside the container
    yields "unavailable" -- a silent loss of exactly the provenance F10 exists to
    capture. The launcher captures the sha on the HOST and passes it in through
    the environment; the subprocess call below is the fallback for when this runs
    somewhere git does exist.
    """
    import os
    import subprocess

    if env_key:
        from_env = os.getenv(env_key, "").strip()
        if from_env:
            return from_env

    try:'''

assert old in s, "git sha helper not found"
s = s.replace(old, new, 1)

s = s.replace(
    '''            "parent_repo": _git_sha(workspace),
            "brainflux": _git_sha(workspace / "brainflux"),''',
    '''            "parent_repo": _git_sha(workspace, "BRAINFLUX_GIT_SHA"),
            "brainflux": _git_sha(
                workspace / "brainflux", "BRAINFLUX_GIT_SHA_SUBMODULE"
            ),''',
    1,
)

p.write_text(s, encoding="utf-8")
print("patched: git sha now read from the environment first")
