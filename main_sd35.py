import os
import random

import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

# --- Configuration ---
SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
ROOT = os.path.dirname(os.path.abspath(__file__))
T2I_DIR = os.path.join(ROOT, "datasets", "T2I-CompBench")
SD35_OUTPUT_DIR = os.path.join(ROOT, "datasets", "Generated_Images_SD35")
DATASET = os.environ.get("DATASET", "t2i")  # "t2i" or "geneval"
SEED = 42

if DATASET == "geneval":
    CATEGORIES = {
        "geneval_color": os.path.join(ROOT, "datasets", "geneval_color_attr_prompts.txt"),
    }
    PROMPTS_PER_CATEGORY = None
else:
    CATEGORIES = {
        "spatial": os.path.join(T2I_DIR, "spatial_val.txt"),
        "numeric": os.path.join(T2I_DIR, "numeracy_val.txt"),
        "3Dspatial": os.path.join(T2I_DIR, "3d_spatial_val.txt"),
        "complex": os.path.join(T2I_DIR, "complex_val.txt"),
    }
    PROMPTS_PER_CATEGORY = 25

# Generation settings
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0


def sanitize_filename(text: str, max_len: int = 100) -> str:
    safe = "".join([c for c in text if c.isalnum() or c in (" ", "-", "_")]).strip()
    return safe[:max_len] or "prompt"


hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Warning: Could not login to Hugging Face: {e}")
else:
    print("Warning: HF_TOKEN environment variable not set; proceeding without login.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Dataset: {DATASET}")

sd35_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    sd35_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL_ID,
        torch_dtype=sd35_dtype,
    )

    if device == "cuda":
        sd35_pipe.enable_model_cpu_offload()
        # Some diffusers versions may not expose these methods on SD3.
        if hasattr(sd35_pipe, "vae") and hasattr(sd35_pipe.vae, "enable_slicing"):
            sd35_pipe.vae.enable_slicing()
        if hasattr(sd35_pipe, "vae") and hasattr(sd35_pipe.vae, "enable_tiling"):
            sd35_pipe.vae.enable_tiling()
    else:
        sd35_pipe.to(device)

    print("SD 3.5 model loaded successfully.")
except Exception as e:
    print(f"Error loading SD 3.5 model: {e}")
    raise SystemExit(1)


for category, file_path in CATEGORIES.items():
    print(f"\n--- Processing Category with SD 3.5: {category} ---")

    if not os.path.exists(file_path):
        print(f"Error: Prompt file not found: {file_path}")
        print(f"Skipping category {category}.")
        continue

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    if not all_prompts:
        print(f"No prompts found in {file_path}.")
        continue

    if PROMPTS_PER_CATEGORY is None:
        selected_prompts = all_prompts
    else:
        random.seed(SEED)
        if len(all_prompts) >= PROMPTS_PER_CATEGORY:
            selected_prompts = random.sample(all_prompts, PROMPTS_PER_CATEGORY)
        else:
            print(f"Warning: Only {len(all_prompts)} prompts found. Using all.")
            selected_prompts = all_prompts

    category_dir = os.path.join(SD35_OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    print(f"Generating {len(selected_prompts)} images for {category} with SD 3.5...")

    for i, prompt in enumerate(selected_prompts):
        print(f"  Generating {i + 1}/{len(selected_prompts)}: {prompt[:50]}...")

        generator = torch.Generator(device="cpu").manual_seed(SEED + i)

        image = sd35_pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        ).images[0]

        filename = f"{i:03d}_{sanitize_filename(prompt)}.png"
        save_path = os.path.join(category_dir, filename)
        image.save(save_path)

    print(f"Completed {category} with SD 3.5.")
