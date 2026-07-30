# BigData-Project

Generate images with diffusion models from compositional prompts (T2I-CompBench and GenEval), then compare them with OWL-ViT-based evaluation.

---

## 1. Overview

This repository runs text-to-image models on the same prompt sets and saves images per model/category for comparison.

| Component | Description |
|-----------|-------------|
| Cluster scripts | `main.py`, `main_sd35.py`, `main_pixart.py` via SLURM + Docker |
| Colab | `run_on_colab.ipynb` (SDXL); SD3.5 / PixArt / FLUX run on cluster |
| Prompt prep | `GenEval/GenEval.ipynb` extracts GenEval color attribute prompts from metadata |

### Models

| Script / notebook | Model ID | Default output folder |
|-------------------|----------|------------------------|
| `main.py` | `black-forest-labs/FLUX.1-dev` | `datasets/Generated_Images_FLUX/` |
| `main_sd35.py` | `stabilityai/stable-diffusion-3.5-large` | `datasets/Generated_Images_SD35/` |
| `main_pixart.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` or `PixArt-alpha/PixArt-Alpha-XL-2-1024-MS` | `datasets/Generated_Images_PixArt_Sigma/` `datasets/Generated_Images_PixArt_Alpha/` |
| `run_on_colab.ipynb` | `stabilityai/stable-diffusion-xl-base-1.0` | `datasets/Generated_Images_SDXL/` |

### Prompt datasets

| `DATASET` env | Source | Sampling |
|---------------|--------|----------|
| `t2i` (default) | `datasets/T2I-CompBench/` → `spatial`, `numeric`, `3Dspatial`, `complex` | 25 prompts per category (`SEED=42`) |
| `geneval` | `datasets/geneval_color_attr_prompts.txt` | **all** prompts (~100) |

GenEval color prompts are built from `GenEval/evaluation_metadata.jsonl` (tag `color_attr`) using `GenEval/GenEval.ipynb`.

### Generation settings (for fair comparison)

| Model | Resolution | Steps | Guidance | Seed |
|-------|------------|-------|----------|------|
| FLUX.1-dev | model default (~1024) | 50 | 3.5 | `42 + i` |
| SD 3.5 Large | 1024×1024 | 50 | 5.0 | `42 + i` |
| PixArt-Sigma/Alpha | 1024×1024 | 50 | 4.5 | `42 + i` |
| SDXL (Colab) | 1024×1024 | 50 | 7.5 | `42 + i` |

---

## 2. Repository layout

```text
BigData-Project/
├── main.py                 # FLUX (cluster)
├── main_sd35.py            # SD 3.5 (cluster)
├── main_pixart.py          # PixArt (cluster)
├── run_on_colab.ipynb      # SDXL on Colab
├── scripts/train.sh        # Docker entrypoint → runs MODEL_SCRIPT
├── build/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── run_docker.sh       # SLURM wrapper
├── datasets/
│   ├── T2I-CompBench/      # spatial, numeric, 3Dspatial, complex, ...
│   ├── geneval_color_attr_prompts.txt
│   ├── reconstructed_3x3_prompts.txt
│   └── Generated_Images_*/ # model outputs
├── GenEval/
│   ├── evaluation_metadata.jsonl
│   └── GenEval.ipynb       # extract color_attr prompts
├── logs/                   # SLURM .out / .err
└── README.md
```

---

## 3. Requirements (cluster)

- SSH to UniboNLP master node **faretra**: `137.204.107.40`, port `37335`  
  Guide: http://137.204.107.40:37339/slurm/guide
- Docker (rootless) working (`docker ps`)
- Hugging Face token with access to gated models (FLUX, SD 3.5, …)
- Project code available on the node (clone / `git pull`)

Store code and submit jobs from **faretra**.

---

## 4. First-time setup (once)

```bash
ssh YOUR_USERNAME@137.204.107.40 -p 37335
cd ~
git clone <THIS_REPO_URL> BigData-Project
cd BigData-Project

docker build -f build/Dockerfile -t my_flux_image .
chmod +x build/run_docker.sh scripts/train.sh
mkdir -p logs
```

Rebuild the image only after changing `build/Dockerfile` or `build/requirements.txt`.

---

## 5. Run on the cluster (every job)

### Step 1 — Hugging Face token

```bash
export HF_TOKEN="your_huggingface_token_here"
if [ -n "$HF_TOKEN" ]; then echo "HF_TOKEN set"; else echo "HF_TOKEN missing"; fi
```

### Step 2 — Choose dataset + model

You should select your prefered `DATASET` and `MODEL_SCRIPT` by the `export` command from this list and use like the example bellow to run.

Valid `MODEL_SCRIPT` values: `main.py` | `main_sd35.py` | `main_pixart.py`  
Valid `DATASET` values: `t2i` | `geneval`

> Note: If you want to run PixArt_Sigma or PixArt_Alpha you should un/comment the model name in the code file (`main_pixart.py`). Other files run the model which explain in [Models](#models) section.

```bash
# Example A: T2I-CompBench with SD 3.5
export DATASET="t2i"
export MODEL_SCRIPT="main_sd35.py"

# Example B: GenEval color with FLUX
export DATASET="geneval"
export MODEL_SCRIPT="main.py"

# Example C: GenEval color with PixArt
export DATASET="geneval"
export MODEL_SCRIPT="main_pixart.py"
```

### Step 3 — Submit

After selecting your prefered model and dataset from previous step now you can sumbit the job on the server to run.

```bash
sbatch -N 1 \
  --gpus=nvidia_geforce_rtx_3090:1 \
  -w faretra \
  --job-name=gen_run \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  build/run_docker.sh
```

Expected: `Submitted batch job <JOBID>`

### Step 4 — Monitor

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode
cat logs/gen_run-<JOBID>.out
cat logs/gen_run-<JOBID>.err
```

### Step 5 — Find outputs

When the job is finished you can find the output under the **category name** used in code:

**T2I (`DATASET=t2i`):**

```text
datasets/Generated_Images_<MODEL>/spatial/
datasets/Generated_Images_<MODEL>/numeric/
datasets/Generated_Images_<MODEL>/3Dspatial/
datasets/Generated_Images_<MODEL>/complex/
```

**GenEval (`DATASET=geneval`):**

```text
datasets/Generated_Images_<MODEL>/geneval_color/
```

The images in those folders are saved like this example per prompt:

Example filenames: `000_<sanitized_prompt>.png`

---

## 6. Full copy-paste example (GenEval + FLUX)

This is the example to run Flux model on a GenEval dataset.

```bash
cd ~/BigData-Project
git pull
mkdir -p logs
export HF_TOKEN="your_huggingface_token_here"
export DATASET="geneval"
export MODEL_SCRIPT="main.py"

sbatch -N 1 \
  --gpus=nvidia_geforce_rtx_3090:1 \
  -w faretra \
  --job-name=flux_geneval \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  build/run_docker.sh
```

To run another model on the same prompts, change only `MODEL_SCRIPT` and submit again.

---

## 7. Run on Google Colab

We use GoogleColab pro with A100 GPU to run only the SDXL model and also extracting GenEval Color_attribute prompt from its metadata file.

**Colab steps**

1. Open `run_on_colab.ipynb` on GoogleColab (GPU runtime, A100).
2. Mount Google Drive.
3. Set prompt/output paths to your Drive folders.
4. Add `HF_TOKEN` in Colab Secrets (needed for gated models).
5. Run the code from top to bottom.

**Extract GenEval prompts (optional)**

The output of this part is exist in `datasets/geneval_color_attr_prompts.txt` so you do not need to do this part.

1. Open `GenEval/GenEval.ipynb`.
2. Point to `evaluation_metadata.jsonl`.
3. It writes `geneval_color_attr_prompts.txt` (tag `color_attr`).

---

## 8. How the cluster pipeline works

```text
sbatch build/run_docker.sh
        │
        ▼
docker run my_flux_image
  - mounts project → /workspace
  - mounts /llms (HF cache)
  - passes HF_TOKEN, DATASET, MODEL_SCRIPT
        │
        ▼
scripts/train.sh
        │
        ▼
python /workspace/<MODEL_SCRIPT>
```

You do **not** edit `train.sh` for each run; set `MODEL_SCRIPT` and `DATASET` in the shell before `sbatch`.

