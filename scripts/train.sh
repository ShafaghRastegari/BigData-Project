#!/bin/bash

set -euo pipefail
set -x

cd /workspace

export HF_HOME="${HF_HOME:-/llms}"
export DATASET="${DATASET:-t2i}"
MODEL_SCRIPT="${MODEL_SCRIPT:-main.py}"

python "/workspace/${MODEL_SCRIPT}"
