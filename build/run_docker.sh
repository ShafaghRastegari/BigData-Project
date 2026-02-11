#!/bin/bash
set -e

PHYS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LLM_CACHE_DIR="/llms"

docker run \
  -v "$PHYS_DIR":/workspace \
  -v "$LLM_CACHE_DIR":"$LLM_CACHE_DIR" \
  -e HF_HOME="$LLM_CACHE_DIR" \
  -e HF_TOKEN="$HF_TOKEN" \
  --rm \
  --gpus device="$CUDA_VISIBLE_DEVICES" \
  my_flux_image \
  /workspace/scripts/train.sh
