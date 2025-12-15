"""Configuration settings for the self-refining image generation loop."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# API Keys (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Provider: "openai" or "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Model settings
SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
IMAGE_SIZE = 1024
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0

# Loop settings
QUALITY_THRESHOLD = 8.0  # Stop if quality score >= this value (0-10 scale)
MAX_ITERATIONS = 5       # Hard limit on iterations
CONVERGENCE_PATIENCE = 2  # Stop if no improvement for this many iterations

# Default negative prompt
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "bad proportions, extra limbs, cloned face, disfigured, "
    "out of frame, watermark, signature, text"
)

