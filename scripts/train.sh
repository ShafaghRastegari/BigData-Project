#!/bin/bash

set -euo pipefail
set -x

cd /workspace

# Use shared HF cache if provided
export HF_HOME="${HF_HOME:-/llms}"

python /workspace/main.py
