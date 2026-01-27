#!/bin/bash

source .env

API_KEY="${EMBEDD_API_KEY}"
PORT="${EMBEDD_PORT}"
MODEL_NAME="${EMBEDD_MODEL}"
GPUS="${EMBEDD_GPUS}"

CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    --task embedding \
    --gpu_memory_utilization 0.5 \
    "${MODEL_NAME}"