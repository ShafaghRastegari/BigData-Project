import os
import random

import torch
from diffusers import PixArtAlphaPipeline
from huggingface_hub import login

# --- Configuration ---
# PIXART_MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
PIXART_MODEL_ID = "PixArt‑alpha/PixArt‑Sigma‑XL‑2‑1024‑MS"
ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(ROOT, "datasets", "T2I-CompBench")
PIXART_OUTPUT_DIR = os.path.join(ROOT, "datasets", "Generated_Images_PixArt")

CATEGORIES = {
    "spatial": "spatial_val.txt",
    "numeric": "numeracy_val.txt",
    "3Dspatial": "3d_spatial_val.txt",
    "complex": "complex_val.txt",
}

PROMPTS_PER_CATEGORY = 25
SEED = 42

# Generation settings
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.5


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

pixart_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    pixart_pipe = PixArtAlphaPipeline.from_pretrained(
        PIXART_MODEL_ID,
        torch_dtype=pixart_dtype,
    )

    if device == "cuda":
        pixart_pipe.enable_model_cpu_offload()
        if hasattr(pixart_pipe, "vae") and hasattr(pixart_pipe.vae, "enable_slicing"):
            pixart_pipe.vae.enable_slicing()
        if hasattr(pixart_pipe, "vae") and hasattr(pixart_pipe.vae, "enable_tiling"):
            pixart_pipe.vae.enable_tiling()
    else:
        pixart_pipe.to(device)

    print("PixArt-XL model loaded successfully.")
except Exception as e:
    print(f"Error loading PixArt-XL model: {e}")
    raise SystemExit(1)


for category, prompt_file in CATEGORIES.items():
    print(f"\n--- Processing Category with PixArt: {category} ---")

    file_path = os.path.join(PROMPT_DIR, prompt_file)
    if not os.path.exists(file_path):
        print(f"Error: Prompt file not found: {file_path}")
        print(f"Skipping category {category}.")
        continue

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {prompt_file}: {e}")
        continue

    if not all_prompts:
        print(f"No prompts found in {prompt_file}.")
        continue

    random.seed(SEED)
    if len(all_prompts) >= PROMPTS_PER_CATEGORY:
        selected_prompts = random.sample(all_prompts, PROMPTS_PER_CATEGORY)
    else:
        print(f"Warning: Only {len(all_prompts)} prompts found in {prompt_file}. Using all.")
        selected_prompts = all_prompts

    category_dir = os.path.join(PIXART_OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    print(f"Generating {len(selected_prompts)} images for {category} with PixArt...")

    for i, prompt in enumerate(selected_prompts):
        print(f"  Generating {i + 1}/{len(selected_prompts)}: {prompt[:50]}...")

        generator = torch.Generator(device="cpu").manual_seed(SEED + i)

        image = pixart_pipe(
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

    print(f"Completed {category} with PixArt.")
