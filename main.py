import os
import random
import torch
from diffusers import FluxPipeline
from huggingface_hub import login

# --- Configuration ---
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
ROOT = os.path.dirname(os.path.abspath(__file__))
T2I_DIR = os.path.join(ROOT, "datasets", "T2I-CompBench")
FLUX_OUTPUT_DIR = os.path.join(ROOT, "datasets", "Generated_Images_FLUX")
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
print(f"Dataset: {DATASET}")

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
    
    
for category, file_path in CATEGORIES.items():
    print(f"\n--- Processing Category with FLUX: {category} ---")

    if not os.path.exists(file_path):
        print(f"Error: Prompt file not found: {file_path}")
        print(f"Skipping category {category}. Please check the filename.")
        continue

    # Read prompts
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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