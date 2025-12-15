"""Gradio interface for the self-refining image generation loop."""

import sys
from pathlib import Path
from typing import Generator

import gradio as gr
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import RefinementPipeline
from src.schemas import IterationResult


# Custom CSS for a clean light theme
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --surface-3: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    background: var(--surface) !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 2rem !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid var(--surface-3);
    margin-bottom: 1.5rem;
}

.main-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
}

.main-header p {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    margin-top: 0.5rem !important;
}

.iteration-card {
    background: var(--surface-2) !important;
    border: 1px solid var(--surface-3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

.score-badge {
    font-family: 'JetBrains Mono', monospace;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
}

.score-high { background: #d1fae5; color: #047857; }
.score-mid { background: #fef3c7; color: #b45309; }
.score-low { background: #fee2e2; color: #b91c1c; }

.failure-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.125rem 0.5rem;
    background: var(--surface-3);
    border-radius: 4px;
    margin-right: 0.25rem;
    color: var(--text-muted);
}

.prompt-display {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    background: var(--surface-2) !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--surface-3) !important;
    color: var(--text) !important;
}

button.primary {
    background: var(--primary) !important;
    border: none !important;
    font-weight: 500 !important;
}

button.primary:hover {
    background: var(--primary-hover) !important;
}
"""


def format_iteration_html(iteration: IterationResult) -> str:
    """Format an iteration result as HTML for display."""
    score = iteration.critique.quality_score
    score_class = "score-high" if score >= 8 else "score-mid" if score >= 5 else "score-low"
    
    failure_tags = "".join(
        f'<span class="failure-tag">{mode.value.replace("_", " ")}</span>'
        for mode in iteration.critique.failure_modes
    )
    
    weaknesses = "".join(f"<li>{w}</li>" for w in iteration.critique.weaknesses)
    suggestions = "".join(f"<li>{s}</li>" for s in iteration.critique.improvement_suggestions)
    
    changes = ""
    if iteration.refined_prompt:
        changes = "".join(f"<li>{c}</li>" for c in iteration.refined_prompt.changes_made)
    
    return f"""
    <div class="iteration-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1rem; color: #1e293b;">Iteration {iteration.iteration_number + 1}</h3>
            <span class="score-badge {score_class}">{score:.1f}/10</span>
        </div>
        
        <div style="margin-bottom: 0.75rem;">
            <strong style="font-size: 0.8rem; color: #64748b;">Failure Modes:</strong>
            <div style="margin-top: 0.25rem;">{failure_tags or '<span style="color: #64748b; font-size: 0.8rem;">None detected</span>'}</div>
        </div>
        
        <details style="margin-bottom: 0.5rem;">
            <summary style="cursor: pointer; font-size: 0.85rem; color: #475569;">Weaknesses</summary>
            <ul style="font-size: 0.8rem; color: #64748b; margin: 0.5rem 0; padding-left: 1.25rem;">{weaknesses or '<li>None identified</li>'}</ul>
        </details>
        
        <details style="margin-bottom: 0.5rem;">
            <summary style="cursor: pointer; font-size: 0.85rem; color: #475569;">Suggestions</summary>
            <ul style="font-size: 0.8rem; color: #64748b; margin: 0.5rem 0; padding-left: 1.25rem;">{suggestions or '<li>None provided</li>'}</ul>
        </details>
        
        {f'''<details>
            <summary style="cursor: pointer; font-size: 0.85rem; color: #475569;">Changes Made</summary>
            <ul style="font-size: 0.8rem; color: #64748b; margin: 0.5rem 0; padding-left: 1.25rem;">{changes}</ul>
        </details>''' if changes else ''}
        
        <div class="prompt-display" style="margin-top: 0.75rem;">
            <strong style="font-size: 0.75rem; color: #64748b;">Prompt:</strong>
            <div style="margin-top: 0.25rem; color: #1e293b;">{iteration.prompt}</div>
        </div>
        
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem;">
            Seed: {iteration.seed}
        </div>
    </div>
    """


def run_refinement_loop(
    prompt: str,
    negative_prompt: str,
    quality_threshold: float,
    max_iterations: int,
    seed: int | None,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple, None, None]:
    """Run the refinement loop with streaming updates.
    
    Yields:
        Tuple of (gallery_images, iterations_html, final_prompt, status_message)
    """
    if not prompt.strip():
        yield [], "", "", "⚠️ Please enter a prompt"
        return
    
    # Convert empty seed to None
    actual_seed = seed if seed and seed > 0 else None
    max_iters = int(max_iterations)
    
    pipeline = RefinementPipeline(
        quality_threshold=quality_threshold,
        max_iterations=max_iters,
    )
    
    gallery_images: list[tuple[Image.Image, str]] = []
    iterations_html = ""
    final_prompt = ""
    
    try:
        progress(0, desc="Loading models...")
        yield gallery_images, iterations_html, "", "⏳ Loading models..."
        
        stop_reason = "max_iterations"
        
        for iteration, reason in pipeline.iterate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            seed=actual_seed,
        ):
            iter_num = iteration.iteration_number + 1
            
            # Update progress
            progress(iter_num / max_iters, desc=f"Iteration {iter_num}/{max_iters}")
            
            # Load and add image
            img = Image.open(iteration.image_path)
            gallery_images.append((img, f"Iteration {iter_num}"))
            
            # Update HTML
            iterations_html += format_iteration_html(iteration)
            final_prompt = iteration.prompt
            
            # Yield intermediate results
            status = f"⏳ Iteration {iter_num}/{max_iters} — Score: {iteration.critique.quality_score:.1f}/10"
            if reason:
                stop_reason = reason
                status = f"✓ Completed — {reason.replace('_', ' ')}"
            
            yield gallery_images, iterations_html, final_prompt, status
        
        # Final yield
        yield gallery_images, iterations_html, final_prompt, f"✓ Completed in {len(gallery_images)} iteration(s) — {stop_reason.replace('_', ' ')}"
        
    except Exception as e:
        yield gallery_images, iterations_html, final_prompt, f"⚠️ Error: {str(e)}"


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(css=CUSTOM_CSS, title="Self-Refining Image Loop") as app:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>⟳ Self-Refining Image Loop</h1>
                <p>Iterative image generation with automatic critique and prompt refinement</p>
            </div>
        """)
        
        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A serene Japanese garden at sunset, koi pond, cherry blossoms...",
                    lines=3,
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="(Uses default if empty)",
                    lines=2,
                    value="",
                )
                
                with gr.Row():
                    quality_threshold = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=config.QUALITY_THRESHOLD,
                        step=0.5,
                        label="Quality Threshold",
                    )
                    max_iterations = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=config.MAX_ITERATIONS,
                        step=1,
                        label="Max Iterations",
                    )
                
                seed_input = gr.Number(
                    label="Seed (optional)",
                    value=None,
                    precision=0,
                )
                
                run_button = gr.Button("Generate & Refine", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=True,
                )
                
                final_prompt_output = gr.Textbox(
                    label="Final Prompt",
                    interactive=False,
                    lines=3,
                )
            
            # Right column: Results
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    columns=3,
                    rows=2,
                    height=400,
                    object_fit="contain",
                )
                
                iterations_display = gr.HTML(
                    label="Iteration Details",
                    value="<p style='color: #64748b; text-align: center; padding: 2rem;'>Run the loop to see iteration details</p>",
                )
        
        # Wire up the interface
        run_button.click(
            fn=run_refinement_loop,
            inputs=[
                prompt_input,
                negative_prompt_input,
                quality_threshold,
                max_iterations,
                seed_input,
            ],
            outputs=[
                gallery,
                iterations_display,
                final_prompt_output,
                status_output,
            ],
        )
    
    return app


def main(share: bool = False, port: int = 7860):
    """Launch the Gradio app.
    
    Args:
        share: If True, creates a public URL for remote access.
        port: Port to run the server on.
    """
    app = create_ui()
    app.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=port,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public URL for remote access")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()
    main(share=args.share, port=args.port)

