"""Main refinement loop orchestrator."""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Optional

from PIL import Image

import config

from .critic import ImageCritic
from .generator import ImageGenerator
from .refiner import PromptRefiner
from .schemas import CritiqueResult, IterationResult, PipelineResult, RefinedPrompt


class RefinementPipeline:
    """Orchestrates the iterative image generation and refinement loop."""

    def __init__(
        self,
        generator: Optional[ImageGenerator] = None,
        critic: Optional[ImageCritic] = None,
        refiner: Optional[PromptRefiner] = None,
        output_dir: Path = config.OUTPUTS_DIR,
        quality_threshold: float = config.QUALITY_THRESHOLD,
        max_iterations: int = config.MAX_ITERATIONS,
        convergence_patience: int = config.CONVERGENCE_PATIENCE,
    ):
        """Initialize the pipeline.

        Args:
            generator: Image generator instance. Creates default if None.
            critic: Image critic instance. Creates default if None.
            refiner: Prompt refiner instance. Creates default if None.
            output_dir: Directory to save outputs.
            quality_threshold: Stop if quality_score >= this value.
            max_iterations: Maximum number of iterations.
            convergence_patience: Stop if no improvement for this many iterations.
        """
        self.generator = generator or ImageGenerator()
        self.critic = critic or ImageCritic()
        self.refiner = refiner or PromptRefiner()
        self.output_dir = output_dir
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.convergence_patience = convergence_patience

    def _create_run_dir(self) -> Path:
        """Create a unique directory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_iteration_metadata(
        self,
        run_dir: Path,
        iteration: IterationResult,
    ) -> None:
        """Save iteration metadata to JSON."""
        metadata_path = run_dir / f"iteration_{iteration.iteration_number:02d}.json"
        
        # Convert to dict, handling Path objects
        data = iteration.model_dump()
        data["image_path"] = str(data["image_path"])
        
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _check_convergence(
        self,
        iterations: list[IterationResult],
    ) -> bool:
        """Check if the loop has converged (no improvement).

        Args:
            iterations: List of completed iterations.

        Returns:
            True if converged (should stop), False otherwise.
        """
        if len(iterations) < self.convergence_patience + 1:
            return False

        recent_scores = [
            it.critique.quality_score
            for it in iterations[-(self.convergence_patience + 1):]
        ]
        
        # Check if the most recent scores haven't improved
        best_before = max(recent_scores[:-1])
        current = recent_scores[-1]
        
        return current <= best_before

    def iterate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Generator[tuple[IterationResult, str | None], None, None]:
        """Run the refinement loop as a generator, yielding each iteration.

        Args:
            prompt: Initial prompt for image generation.
            negative_prompt: Initial negative prompt.
            seed: Optional seed for first iteration (random if None).

        Yields:
            Tuple of (IterationResult, stop_reason or None if continuing).
        """
        run_dir = self._create_run_dir()
        iterations: list[IterationResult] = []
        current_prompt = prompt
        current_negative = negative_prompt or config.DEFAULT_NEGATIVE_PROMPT
        current_seed = seed

        for i in range(self.max_iterations):
            # Generate image
            image_path = run_dir / f"image_{i:02d}.png"
            _, used_seed = self.generator.generate_and_save(
                prompt=current_prompt,
                output_path=image_path,
                negative_prompt=current_negative,
                seed=current_seed,
            )

            # Critique the image
            critique = self.critic.critique(
                image=image_path,
                prompt=current_prompt,
                iteration=i,
            )

            # Determine if we need to refine
            should_continue = (
                critique.quality_score < self.quality_threshold
                and i < self.max_iterations - 1
            )

            refined: Optional[RefinedPrompt] = None
            if should_continue:
                refined = self.refiner.refine(
                    original_prompt=current_prompt,
                    critique=critique,
                    current_negative_prompt=current_negative,
                    iteration=i,
                )

            # Create iteration result
            iteration_result = IterationResult(
                iteration_number=i,
                prompt=current_prompt,
                negative_prompt=current_negative,
                seed=used_seed,
                image_path=image_path,
                critique=critique,
                refined_prompt=refined,
            )

            iterations.append(iteration_result)
            self._save_iteration_metadata(run_dir, iteration_result)

            # Check stopping conditions
            stop_reason = None
            if critique.quality_score >= self.quality_threshold:
                stop_reason = "quality_threshold"
            elif self._check_convergence(iterations):
                stop_reason = "convergence"
            elif i == self.max_iterations - 1:
                stop_reason = "max_iterations"

            # Yield the iteration result
            yield iteration_result, stop_reason

            if stop_reason:
                break

            # Prepare for next iteration
            if refined:
                current_prompt = refined.new_prompt
                if refined.negative_prompt:
                    current_negative = refined.negative_prompt
                current_seed = None

            # Clear GPU cache between iterations
            self.generator.clear_cache()

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        on_iteration: Optional[Callable[[IterationResult], None]] = None,
    ) -> PipelineResult:
        """Run the refinement loop.

        Args:
            prompt: Initial prompt for image generation.
            negative_prompt: Initial negative prompt.
            seed: Optional seed for first iteration (random if None).
            on_iteration: Optional callback called after each iteration.

        Returns:
            Complete pipeline result with all iterations.
        """
        iterations: list[IterationResult] = []
        stop_reason = "max_iterations"

        for iteration_result, reason in self.iterate(prompt, negative_prompt, seed):
            iterations.append(iteration_result)
            if on_iteration:
                on_iteration(iteration_result)
            if reason:
                stop_reason = reason

        # Find best iteration
        best_iteration = max(iterations, key=lambda x: x.critique.quality_score)

        # Save final summary (get run_dir from first iteration's image path)
        run_dir = iterations[0].image_path.parent
        summary = {
            "total_iterations": len(iterations),
            "stop_reason": stop_reason,
            "final_quality_score": iterations[-1].critique.quality_score,
            "best_quality_score": best_iteration.critique.quality_score,
            "best_iteration": best_iteration.iteration_number,
            "initial_prompt": prompt,
            "final_prompt": iterations[-1].prompt,
        }
        
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return PipelineResult(
            iterations=iterations,
            final_image_path=best_iteration.image_path,
            final_prompt=best_iteration.prompt,
            total_iterations=len(iterations),
            stop_reason=stop_reason,
        )


def run_pipeline(
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    quality_threshold: float = config.QUALITY_THRESHOLD,
    max_iterations: int = config.MAX_ITERATIONS,
) -> PipelineResult:
    """Convenience function to run the refinement pipeline.

    Args:
        prompt: Initial prompt for image generation.
        negative_prompt: Initial negative prompt.
        seed: Optional seed for first iteration.
        quality_threshold: Stop if quality_score >= this value.
        max_iterations: Maximum number of iterations.

    Returns:
        Complete pipeline result.
    """
    pipeline = RefinementPipeline(
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
    )
    return pipeline.run(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

