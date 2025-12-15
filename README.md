# Self-Refining Image Generation Loop

A research prototype that iteratively generates images, critiques them, and refines prompts until quality thresholds are met. The system uses Stable Diffusion 3 for generation and GPT-4o (or Claude) for vision-based critique and prompt refinement.

```
User Prompt
     ↓
Stable Diffusion 3
     ↓
Generated Image
     ↓
Vision LLM Critique ←──┐
     ↓                 │
Quality Check          │
     ↓                 │
  [Pass] → Final Output│
     ↓                 │
Prompt Refinement ─────┘
```

## Features

- **Iterative refinement loop** with automatic stopping criteria
- **Structured critique** with explicit failure modes (composition, anatomy, lighting, etc.)
- **Seed tracking** for full reproducibility
- **Provider flexibility** — swap between OpenAI and Anthropic via config
- **Real-time UI** with Gradio showing progress and intermediate results

## Setup

```bash
# Clone and enter directory
cd image-gen-loop

# Create conda environment
conda create -n image-gen-loop python=3.12
conda activate image-gen-loop

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for Claude support)
ANTHROPIC_API_KEY=sk-ant-...

LLM_PROVIDER=openai  # or "anthropic"
```

## Usage

### Web UI

```bash
# Local access
python main.py ui

# Remote access (creates public URL)
python main.py ui --share
```

### Command Line

```bash
# Basic generation
python main.py generate "A serene Japanese garden at sunset with koi pond"

# With options
python main.py generate "cyberpunk cityscape" \
    --max-iterations 3 \
    --threshold 7.5 \
    --seed 42
```

### Python API

```python
from src import RefinementPipeline

pipeline = RefinementPipeline(
    quality_threshold=8.0,
    max_iterations=5,
)

result = pipeline.run(
    prompt="A majestic lion in golden savanna light",
    seed=12345,
)

print(f"Final image: {result.final_image_path}")
print(f"Iterations: {result.total_iterations}")
print(f"Stop reason: {result.stop_reason}")
```

## Configuration

Edit `config.py` to adjust defaults:

| Setting                | Default | Description                             |
| ---------------------- | ------- | --------------------------------------- |
| `QUALITY_THRESHOLD`    | 8.0     | Stop when score ≥ this value            |
| `MAX_ITERATIONS`       | 5       | Hard limit on iterations                |
| `CONVERGENCE_PATIENCE` | 2       | Stop if no improvement for N iterations |
| `IMAGE_SIZE`           | 1024    | Output resolution                       |
| `NUM_INFERENCE_STEPS`  | 28      | SD3 denoising steps                     |

## Project Structure

```
image-gen-loop/
├── src/
│   ├── generator.py   # SD3 image generation
│   ├── critic.py      # Vision-based critique
│   ├── refiner.py     # Prompt refinement
│   ├── pipeline.py    # Main loop orchestrator
│   ├── schemas.py     # Pydantic models
│   └── llm.py         # LangChain provider abstraction
├── ui/
│   └── app.py         # Gradio interface
├── outputs/           # Generated images + metadata
├── config.py          # Settings
└── main.py            # CLI entry point
```

## Failure Modes

The critic identifies specific failure types:

| Mode                | Description                             |
| ------------------- | --------------------------------------- |
| `composition`       | Poor framing, unbalanced elements       |
| `anatomical_errors` | Wrong proportions, extra limbs          |
| `style_mismatch`    | Style doesn't match request             |
| `lighting`          | Inconsistent or unrealistic lighting    |
| `coherence`         | Elements that don't make sense together |
| `text_rendering`    | Garbled text in image                   |
| `color_issues`      | Unnatural colors                        |
| `perspective`       | Incorrect depth/perspective             |
| `detail_loss`       | Missing details from prompt             |
| `artifact`          | Visual glitches or noise                |

## Output

Each run creates a timestamped folder in `outputs/` containing:

- `image_00.png`, `image_01.png`, ... — Generated images
- `iteration_00.json`, ... — Full metadata per iteration
- `summary.json` — Run summary with scores and prompts


