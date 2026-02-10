#!/bin/bash

set -euo pipefail
set -x

cd /workspace

# Use shared HF cache if provided
export HF_HOME="${HF_HOME:-/llms}"

echo "train.sh received parameters:"
echo "1: $1"

python /workspace/main.py