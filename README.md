# BigData-Project

Text-to-image generation pipeline for comparing diffusion models on compositional prompts from the [T2I-CompBench](https://github.com/Karine-H/T2I-CompBench) benchmark.

## Overview

This project loads prompts from `datasets/T2I-CompBench/`, samples 25 prompts per category (with a fixed seed), and generates images using one of several diffusion models. Outputs are saved per model and per category for side-by-side comparison.

**Supported models**

| Script | Model | Output directory |
|--------|-------|------------------|
| `main.py` | `black-forest-labs/FLUX.1-dev` | `datasets/Generated_Images_FLUX/` |
| `main_sd35.py` | `stabilityai/stable-diffusion-3.5-large` | `datasets/Generated_Images_SD35/` |
| `main_pixart.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | `datasets/Generated_Images_PixArt_Sigma/` |

**Prompt categories:** `spatial`, `numeric`, `3Dspatial`, `complex`

## Prerequisites

- Access to the UniboNLP SLURM cluster (master node: `faretra`)
- Docker (rootless) installed on the server
- Hugging Face account with access to gated models
- `HF_TOKEN` environment variable set before submitting jobs

## Setup

### 1. Build the Docker image

From the project root on the server:

```bash
docker build -f build/Dockerfile -t my_flux_image .
```

### 2. Choose which model to run

Edit `scripts/train.sh` and set the last line to the desired script:

```bash
python /workspace/main.py          # FLUX
python /workspace/main_sd35.py     # SD 3.5
python /workspace/main_pixart.py     # PixArt
```

## Running on the cluster

From the project directory on `faretra`:

```bash
mkdir -p logs
export HF_TOKEN="your_huggingface_token"
chmod +x build/run_docker.sh scripts/train.sh

sbatch -N 1 \
  --gpus=nvidia_geforce_rtx_3090:1 \
  -w faretra \
  --job-name=gen_run \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  build/run_docker.sh
```

The job runs inside Docker via `build/run_docker.sh`, which mounts the project folder and the shared model cache at `/llms`.

## Monitoring

**Check job status**

```bash
squeue -u $USER
sacct -u $USER --starttime today --format=JobID,JobName,State,ExitCode
```

**Read logs**

```bash
cat logs/gen_run-<jobid>.out
cat logs/gen_run-<jobid>.err
```

**GPU usage on the cluster**

Use the SLURM web dashboard or run `nvidia-smi` on the compute node.

**Verify outputs**

Generated images appear under the model-specific output directory, organized by category:

```
datasets/Generated_Images_<MODEL>/
  spatial/
  numeric/
  3Dspatial/
  complex/
```

## Project structure

```
BigData-Project/
├── main.py              # FLUX generation
├── main_sd35.py         # SD 3.5 generation
├── main_pixart.py       # PixArt generation
├── scripts/train.sh     # Entry point run inside Docker
├── build/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── run_docker.sh    # SLURM wrapper for Docker
├── datasets/
│   ├── T2I-CompBench/   # Input prompts
│   └── Generated_Images_*/  # Output images
└── logs/                # SLURM job logs
```

## Notes

- Rebuild the Docker image after changing `build/Dockerfile` or `build/requirements.txt`.
- Python-only changes (e.g. `main_*.py`) take effect without rebuilding; the project folder is mounted into the container.
- For fair model comparison, keep resolution, step count, and seed policy consistent across scripts.
