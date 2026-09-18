#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

: "${EMBEDD_API_KEY:?EMBEDD_API_KEY must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_PORT:?EMBEDD_PORT must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_MODEL:?EMBEDD_MODEL must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_GPUS:?EMBEDD_GPUS must be set in the environment or ${ENV_FILE}}"

API_KEY="${EMBEDD_API_KEY}"
PORT="${EMBEDD_PORT}"
MODEL_NAME="${EMBEDD_MODEL}"
GPUS="${EMBEDD_GPUS}"


# --- GPU visibility check -------------------------------------------------
# The container is normally launched PINNED (--gpus '"device=4,5,6"'), which
# renumbers CUDA devices CONTAINER-LOCAL 0..N-1. A stale PHYSICAL index here
# therefore selects a device that does not exist. vLLM then aborts with
# "World size (N) is larger than the number of available GPUs (M)", the daemon
# dies, and because start_vllm_servers() launches it with no logfile the error
# goes to /dev/null and is never seen -- the caller just polls the port forever.
# That cost one run five hours of doing nothing on 2026-09-18 (HQ #143).
# This check turns that silent stall into an immediate, readable failure.
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if [[ "${VISIBLE_GPUS}" -gt 0 ]]; then
    IFS=',' read -ra _REQ_GPUS <<< "${GPUS}"
    for _g in "${_REQ_GPUS[@]}"; do
        if ! [[ "${_g}" =~ ^[0-9]+$ ]] || (( _g >= VISIBLE_GPUS )); then
            echo "FATAL: requested GPU index '${_g}' is not visible to this container." >&2
            echo "       visible devices : 0..$((VISIBLE_GPUS - 1))  (${VISIBLE_GPUS} total)" >&2
            echo "       requested       : ${GPUS}" >&2
            echo "       source          : ${ENV_FILE}" >&2
            echo "       Under a pinned launch these must be CONTAINER-LOCAL indices," >&2
            echo "       not physical ones. See the note above LLM_GPUS in .env." >&2
            exit 78
        fi
    done
    echo "[vllm-launch] GPUs ${GPUS} requested, ${VISIBLE_GPUS} visible -- ok"
fi

CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    --gpu_memory_utilization 0.5 \
    "${MODEL_NAME}"