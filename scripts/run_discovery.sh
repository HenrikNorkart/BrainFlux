#!/usr/bin/env bash
# F10 (HQ #157): the launcher, so the launch itself is recorded.
#
# bf-newobj was started by an ad-hoc `docker run` typed at a prompt. Nothing in
# the repository said which arm it ran, which corpus it read, or which commit it
# was built from -- the same failure as the untracked .env and the stale
# config.yml, one level further out. A run whose launch command exists only in
# somebody's shell history is not reproducible no matter how good the manifest is.
#
# This script also solves a concrete problem: `git` is NOT installed in
# brainflux_pm, so the container cannot read its own provenance. The shas are
# captured HERE, on the host, and passed in.
#
# Usage:
#   scripts/run_discovery.sh                      # pre-registered arm
#   ROGUE_ONE_PRECISION_MIN=0.98 scripts/run_discovery.sh
#   GPUS=4,5,6,7 scripts/run_discovery.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

IMAGE="${IMAGE:-brainflux_pm:latest}"
GPUS="${GPUS:-4,5,6,7}"

# The pre-registered arm (tmp/t2_prereg.md sections 3-4). Overridable from the
# environment so a secondary analysis is run deliberately, not by editing code.
export ROGUE_ONE_PRECISION_MIN="${ROGUE_ONE_PRECISION_MIN:-0.95}"
export ROGUE_ONE_TM_FRAC="${ROGUE_ONE_TM_FRAC:-0.1}"
export ROGUE_ONE_OBJECTIVE_SEED="${ROGUE_ONE_OBJECTIVE_SEED:-20260917}"

# ---------------------------------------------------------------- GPU pre-flight
# Containers-only execution, at most 4 GPUs, and the visibility check must run
# before anything is launched: a run that cannot see its GPUs dies silently as a
# zombie, which is how the 2026-09-18 attempt was lost.
n_req="$(tr ',' '\n' <<<"$GPUS" | grep -c .)"
if (( n_req > 4 )); then
  echo "refusing to request $n_req GPUs; the limit on this box is 4" >&2
  exit 78
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found; refusing to launch blind" >&2
  exit 78
fi
for g in ${GPUS//,/ }; do
  used="$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo FAIL)"
  if [[ "$used" == "FAIL" ]]; then
    echo "GPU $g not visible; refusing to launch" >&2
    exit 78
  fi
  if (( used > 2000 )); then
    echo "GPU $g already holds ${used} MiB -- check it, do not assume it" >&2
    exit 78
  fi
done

# -------------------------------------------------------------- provenance (AC3)
GIT_SHA="$(git rev-parse HEAD)"
[[ -n "$(git status --porcelain)" ]] && GIT_SHA="${GIT_SHA}-dirty"
GIT_SHA_SUB="$(git -C brainflux rev-parse HEAD 2>/dev/null || echo none)"
if [[ "$GIT_SHA_SUB" != "none" ]] && [[ -n "$(git -C brainflux status --porcelain 2>/dev/null)" ]]; then
  GIT_SHA_SUB="${GIT_SHA_SUB}-dirty"
fi

echo "=============================================================="
echo " arm      : precision_min=$ROGUE_ONE_PRECISION_MIN tm_frac=$ROGUE_ONE_TM_FRAC seed=$ROGUE_ONE_OBJECTIVE_SEED"
echo " gpus     : $GPUS"
echo " git      : $GIT_SHA"
echo " brainflux: $GIT_SHA_SUB"
echo " image    : $IMAGE"
echo "=============================================================="

exec docker run --rm -d \
  --name "discovery_$(date +%Y%m%d-%H%M%S)" \
  --gpus "\"device=$GPUS\"" \
  -v "$REPO":/workspace -w /workspace \
  --env-file .env --env-file .secret.env \
  -e BRAINFLUX_GIT_SHA="$GIT_SHA" \
  -e BRAINFLUX_GIT_SHA_SUBMODULE="$GIT_SHA_SUB" \
  -e ROGUE_ONE_PRECISION_MIN="$ROGUE_ONE_PRECISION_MIN" \
  -e ROGUE_ONE_TM_FRAC="$ROGUE_ONE_TM_FRAC" \
  -e ROGUE_ONE_OBJECTIVE_SEED="$ROGUE_ONE_OBJECTIVE_SEED" \
  --entrypoint python3 "$IMAGE" /workspace/phantom_menace_main.py
