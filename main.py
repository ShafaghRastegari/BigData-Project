import os
import random
import torch
from diffusers import FluxPipeline
from huggingface_hub import login

# --- Configuration ---
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(ROOT, "datasets", "T2I-CompBench")
FLUX_OUTPUT_DIR = os.path.join(ROOT, "datasets", "Generated_Images_FLUX")
# Categories to process and their corresponding filenames in T2I-CompBench

CATEGORIES = {
    'spatial': 'spatial_val.txt',
    'numeric': 'numeracy_val.txt',
    '3Dspatial': '3d_spatial_val.txt',
    'complex': 'complex_val.txt'
}

PROMPTS_PER_CATEGORY = 25
SEED = 42
    

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Warning: Could not login to Hugging Face: {e}")
else:
    print("Warning: HF_TOKEN environment variable not set; proceeding without login.")
    

# --- FLUX Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# FLUX works better with bfloat16
if device == "cpu":
    flux_dtype = torch.float32
else:
    flux_dtype = torch.bfloat16

try:
    flux_pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=flux_dtype
    )

    # Enable memory optimizations for FLUX
    if device == "cuda":
        flux_pipe.enable_model_cpu_offload()
        flux_pipe.vae.enable_slicing()
        flux_pipe.vae.enable_tiling()

    print("FLUX model loaded successfully.")
except Exception as e:
    print(f"Error loading FLUX model: {e}")
    exit()
    
    
for category, filename in CATEGORIES.items():
    print(f"\n--- Processing Category with FLUX: {category} ---")

    file_path = os.path.join(PROMPT_DIR, filename)

    if not os.path.exists(file_path):
        print(f"Error: Prompt file not found: {file_path}")
        print(f"Skipping category {category}. Please check the filename.")
        continue

    # Read prompts
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        continue

    if not all_prompts:
        print(f"No prompts found in {filename}.")
        continue

    # Select 25 prompts
    random.seed(SEED)
    if len(all_prompts) >= PROMPTS_PER_CATEGORY:
        selected_prompts = random.sample(all_prompts, PROMPTS_PER_CATEGORY)
    else:
        print(f"Warning: Only {len(all_prompts)} prompts found in {filename}. Using all.")
        selected_prompts = all_prompts

    category_dir = os.path.join(FLUX_OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    print(f"Generating {len(selected_prompts)} images for {category} with FLUX...")

    for i, prompt in enumerate(selected_prompts):
        print(f"  Generating {i+1}/{len(selected_prompts)}: {prompt[:50]}...")

        generator = torch.Generator(device="cpu").manual_seed(SEED + i)

        # FLUX.1-schnell is optimized for 1-4 steps
        # FLUX.1-dev works better with ~50 steps
        is_schnell = "schnell" in FLUX_MODEL_ID.lower()
        num_steps = 4 if is_schnell else 50
        guidance = 0.0 if is_schnell else 3.5

        image = flux_pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=guidance,
        ).images[0]

        safe_prompt = "".join([c for c in prompt if c.isalnum() or c in (' ', '-', '_')]).strip()
        safe_prompt = safe_prompt[:100]
        filename = f"{i:03d}_{safe_prompt}.png"
        save_path = os.path.join(category_dir, filename)

        image.save(save_path)

    print(f"Completed {category} with FLUX.")