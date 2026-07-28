# BigData-Project

Generate images with diffusion models (FLUX, SD 3.5, PixArt) from compositional prompts, then compare the outputs.

---
## Project layout

```text
BigData-Project/
├── main.py                 # FLUX
├── main_sd35.py            # SD 3.5
├── main_pixart.py          # PixArt
├── scripts/train.sh        # runs the chosen MODEL_SCRIPT inside Docker
├── build/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── run_docker.sh       # SLURM entrypoint
├── datasets/
│   ├── T2I-CompBench/
│   ├── geneval_color_attr_prompts.txt
│   └── Generated_Images_*/
├── GenEval/
|   ├── evaluation_metadata.jsonl
│   └── GenEval.ipynb  
├── colab_pixart.ipynb
└── logs/
```

## What this project does

1. Reads prompts from a dataset.
2. Runs one model script inside Docker on the GPU cluster or on a GoogleColab Pro.
3. Saves images under `datasets/Generated_Images_<MODEL>/...`.

**Prompt datasets**

| `DATASET` value | Prompts used | How many |
|-----------------|--------------|----------|
| `t2i` (default) | `datasets/T2I-CompBench/` (`spatial`, `numeric`, `3Dspatial`, `complex`) | 25 prompts per category |
| `geneval` | `datasets/geneval_color_attr_prompts.txt` | all prompts in the file |

**Model scripts**

| `MODEL_SCRIPT` | Model | Images saved in |
|----------------|-------|-----------------|
| `main.py` | FLUX.1-dev | `datasets/Generated_Images_FLUX/` |
| `main_sd35.py` | Stable Diffusion 3.5 Large | `datasets/Generated_Images_SD35/` |
| `main_pixart.py` | PixArt-Sigma/Alpha | `datasets/Generated_Images_PixArt_Sigma/` or `datasets/Generated_Images_PixArt_Alpha/` |

---

## Requirements

- SSH access to the [UniboNLP cluster](http://137.204.107.40:37339/slurm/guide) (**faretra**: `137.204.107.40`, port `37335`)
- Project cloned on the server (same folder on every node you use)
- Docker image built once: `my_flux_image`
- Hugging Face token with access to gated models

---

## First-time setup (do this once)

### 1. Connect to the server

```bash
ssh YOUR_USERNAME@137.204.107.40 -p 37335
git clone THIS_REPO
cd ~/BigData-Project
```

### 2. Get the code

```bash
git pull
```

### 3. Build the Docker image (only once, or after changing `build/`)

```bash
docker build -f build/Dockerfile -t my_flux_image .
```

### 4. Make scripts executable

```bash
chmod +x build/run_docker.sh scripts/train.sh
mkdir -p logs
```

---

## How to run a job (every time)

### Step 1 — Set your Hugging Face token

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Check it is set:

```bash
if [ -n "$HF_TOKEN" ]; then echo "HF_TOKEN set"; else echo "HF_TOKEN missing"; fi
```

### Step 2 — Choose dataset and model

Export only one DATASET and MODEL_SCRIPT like the example.

Examples:

```bash
# T2I-CompBench + SD 3.5
export DATASET="t2i"
export MODEL_SCRIPT="main_sd35.py"

# GenEval color + FLUX
export DATASET="geneval"
export MODEL_SCRIPT="main.py"

# GenEval color + PixArt
export DATASET="geneval"
export MODEL_SCRIPT="main_pixart.py"
```

### Step 3 — Submit the job

```bash
sbatch -N 1 \
  --gpus=nvidia_geforce_rtx_3090:1 \
  -w faretra \
  --job-name=gen_run \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  build/run_docker.sh
```

You should see: `Submitted batch job <JOBID>`.

### Step 4 — Monitor

```bash
squeue -u $USER
```

When the job finishes (or if it fails):

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode
cat logs/gen_run-<JOBID>.out
cat logs/gen_run-<JOBID>.err
```

### Step 5 — Check outputs

**T2I (`DATASET=t2i`):**

```text
datasets/Generated_Images_SD35/spatial/
datasets/Generated_Images_SD35/numeric/
datasets/Generated_Images_SD35/3Dspatial/
datasets/Generated_Images_SD35/complex/
```

**GenEval (`DATASET=geneval`):**

```text
datasets/Generated_Images_FLUX/geneval_color/
datasets/Generated_Images_SD35/geneval_color/
datasets/Generated_Images_PixArt_Sigma/geneval_color/
```

## Run on Google Colab instead

All the image generation for all prompts (include T2I-CompBench and GenEval) for StableDiffusionXL model is run on GoogleColab pro using A100 GPU. Also all the models on GenEval prompt run on the GoogleColab pro too except FLUX.

1. Open `run_on_colab.ipynb` in Colab (aelect A100 GPU runtime).
2. Mount Drive.
3. Point the prompt file to your GenEval/T2I-CompBench file (e.g. on Drive).
4. Set `HF_TOKEN` in Colab Secrets.
5. Run prefered section cells top to bottom.
