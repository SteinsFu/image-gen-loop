"""Pydantic schemas for structured data flow in the refinement loop."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FailureMode(str, Enum):
    """Explicit failure modes for image critique."""
    
    COMPOSITION = "composition"
    STYLE_MISMATCH = "style_mismatch"
    ANATOMICAL_ERRORS = "anatomical_errors"
    TEXT_RENDERING = "text_rendering"
    LIGHTING = "lighting"
    COHERENCE = "coherence"
    COLOR_ISSUES = "color_issues"
    PERSPECTIVE = "perspective"
    DETAIL_LOSS = "detail_loss"
    ARTIFACT = "artifact"


class CritiqueResult(BaseModel):
    """Structured critique of a generated image."""
    
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Overall quality score from 0 (worst) to 10 (best)"
    )
    failure_modes: list[FailureMode] = Field(
        default_factory=list,
        description="List of identified failure modes in the image"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What the image does well"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific issues with the image"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions to improve the prompt"
    )
    prompt_adherence: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="How well the image matches the original prompt (0-10)"
    )


class RefinedPrompt(BaseModel):
    """Refined prompt based on critique feedback."""
    
    new_prompt: str = Field(
        ...,
        description="The improved prompt incorporating feedback"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of what was changed and why"
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of specific changes applied to the prompt"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Updated negative prompt if needed"
    )


class IterationResult(BaseModel):
    """Result of a single iteration in the refinement loop."""
    
    iteration_number: int = Field(
        ...,
        ge=0,
        description="Zero-indexed iteration number"
    )
    prompt: str = Field(
        ...,
        description="The prompt used for this iteration"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="The negative prompt used"
    )
    seed: int = Field(
        ...,
        description="Random seed used for reproducibility"
    )
    image_path: Path = Field(
        ...,
        description="Path to the saved image file"
    )
    critique: CritiqueResult = Field(
        ...,
        description="Critique of the generated image"
    )
    refined_prompt: Optional[RefinedPrompt] = Field(
        default=None,
        description="Refined prompt for next iteration (None if final)"
    )

    class Config:
        arbitrary_types_allowed = True


class PipelineResult(BaseModel):
    """Final result of the complete refinement pipeline."""
    
    iterations: list[IterationResult] = Field(
        default_factory=list,
        description="All iterations in the refinement loop"
    )
    final_image_path: Path = Field(
        ...,
        description="Path to the best/final image"
    )
    final_prompt: str = Field(
        ...,
        description="The final refined prompt"
    )
    total_iterations: int = Field(
        ...,
        description="Total number of iterations performed"
    )
    stop_reason: str = Field(
        ...,
        description="Why the loop stopped (threshold, max_iterations, convergence)"
    )

    class Config:
        arbitrary_types_allowed = True

