#!/usr/bin/env python3
"""Entry point for the self-refining image generation loop.

Usage:
    # Run with Gradio UI
    python main.py ui
    
    # Run CLI generation
    python main.py generate "A beautiful sunset over mountains"
    
    # Run with options
    python main.py generate "prompt" --max-iterations 3 --threshold 7.5 --seed 42
"""

import argparse
import sys
from pathlib import Path

import config


def run_ui(share: bool = False, port: int = 7860):
    """Launch the Gradio UI."""
    from ui.app import main as launch_ui
    launch_ui(share=share, port=port)


def run_generate(args: argparse.Namespace):
    """Run the generation pipeline from CLI."""
    from src.pipeline import RefinementPipeline
    from src.schemas import IterationResult

    print(f"\n{'='*60}")
    print("Self-Refining Image Generation Loop")
    print(f"{'='*60}\n")
    print(f"Initial prompt: {args.prompt}")
    print(f"Quality threshold: {args.threshold}")
    print(f"Max iterations: {args.max_iterations}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print()

    def on_iteration(iteration: IterationResult):
        """Callback to print progress."""
        score = iteration.critique.quality_score
        adherence = iteration.critique.prompt_adherence
        failures = ", ".join(m.value for m in iteration.critique.failure_modes) or "None"
        
        print(f"[Iteration {iteration.iteration_number + 1}]")
        print(f"  Quality Score: {score:.1f}/10")
        print(f"  Prompt Adherence: {adherence:.1f}/10")
        print(f"  Failure Modes: {failures}")
        print(f"  Image saved: {iteration.image_path}")
        
        if iteration.critique.weaknesses:
            print(f"  Weaknesses:")
            for w in iteration.critique.weaknesses[:3]:  # Show first 3
                print(f"    - {w}")
        
        if iteration.refined_prompt:
            print(f"  → Refining prompt...")
            print(f"  Changes: {', '.join(iteration.refined_prompt.changes_made[:2])}")
        
        print()

    # Create and run pipeline
    pipeline = RefinementPipeline(
        quality_threshold=args.threshold,
        max_iterations=args.max_iterations,
    )

    result = pipeline.run(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        on_iteration=on_iteration,
    )

    print(f"{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {result.total_iterations}")
    print(f"Stop reason: {result.stop_reason.replace('_', ' ')}")
    print(f"Best image: {result.final_image_path}")
    print(f"\nFinal prompt:")
    print(f"  {result.final_prompt}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Refining Image Generation Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio web interface")
    ui_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL for remote access",
    )
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate images from CLI")
    gen_parser.add_argument(
        "prompt",
        type=str,
        help="The initial prompt for image generation",
    )
    gen_parser.add_argument(
        "--negative-prompt", "-n",
        type=str,
        default=None,
        help="Negative prompt (uses default if not specified)",
    )
    gen_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=config.QUALITY_THRESHOLD,
        help=f"Quality threshold to stop (default: {config.QUALITY_THRESHOLD})",
    )
    gen_parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=config.MAX_ITERATIONS,
        help=f"Maximum iterations (default: {config.MAX_ITERATIONS})",
    )
    gen_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    gen_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=config.OUTPUTS_DIR,
        help=f"Output directory (default: {config.OUTPUTS_DIR})",
    )

    args = parser.parse_args()

    if args.command == "ui":
        run_ui(share=args.share, port=args.port)
    elif args.command == "generate":
        run_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

