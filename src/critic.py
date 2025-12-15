"""GPT-4o/Claude vision critic for image analysis using LangChain."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Union

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from .llm import get_vision_model
from .schemas import CritiqueResult, FailureMode


CRITIQUE_SYSTEM_PROMPT = """You are an expert image critic analyzing AI-generated images. Your task is to provide structured, actionable feedback.

Evaluate the image based on:
1. **Technical Quality**: Resolution, clarity, artifacts, noise
2. **Composition**: Balance, framing, visual flow
3. **Prompt Adherence**: How well it matches the intended description
4. **Artistic Merit**: Style consistency, color harmony, lighting
5. **Anatomical Accuracy**: For images with people/animals, check proportions
6. **Coherence**: Does everything in the image make sense together?

Be specific and constructive. Focus on issues that can be fixed by refining the prompt.

Failure modes to consider:
- composition: Poor framing, unbalanced elements, bad cropping
- style_mismatch: Style doesn't match what was requested
- anatomical_errors: Wrong proportions, extra limbs, deformed features
- text_rendering: Garbled or incorrect text in the image
- lighting: Inconsistent or unrealistic lighting
- coherence: Elements that don't make sense together
- color_issues: Unnatural colors, poor color harmony
- perspective: Incorrect perspective or depth
- detail_loss: Missing important details from the prompt
- artifact: Visual glitches, noise, or generation artifacts"""


class ImageCritic:
    """Critiques images using vision-capable LLMs via LangChain."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] | None = None,
    ):
        """Initialize the critic.

        Args:
            provider: LLM provider to use. Uses config default if None.
        """
        self.provider = provider
        self._model = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            base_model = get_vision_model(provider=self.provider)
            # Use structured output with Pydantic model
            self._model = base_model.with_structured_output(CritiqueResult)
        return self._model

    def _image_to_base64(self, image: Union[Image.Image, Path, str]) -> str:
        """Convert an image to base64 string.

        Args:
            image: PIL Image, path to image file, or base64 string.

        Returns:
            Base64-encoded image string.
        """
        if isinstance(image, str):
            if Path(image).exists():
                image = Path(image)
            else:
                return image

        if isinstance(image, Path):
            image = Image.open(image)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def critique(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
        iteration: int = 0,
    ) -> CritiqueResult:
        """Analyze an image and provide structured critique.

        Args:
            image: The image to critique.
            prompt: The prompt used to generate the image.
            iteration: Current iteration number for context.

        Returns:
            Structured critique result.
        """
        image_b64 = self._image_to_base64(image)

        user_content = [
            {
                "type": "text",
                "text": f"""Analyze this AI-generated image.

**Original Prompt**: "{prompt}"
**Iteration**: {iteration + 1}

Provide a detailed critique. Be specific about what works and what needs improvement.""",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        result = self.model.invoke(messages)
        
        # Ensure failure_modes are FailureMode enums
        if isinstance(result, CritiqueResult):
            return result
        
        # Handle dict response (fallback)
        if isinstance(result, dict):
            failure_modes = [
                FailureMode(mode) if isinstance(mode, str) else mode
                for mode in result.get("failure_modes", [])
            ]
            return CritiqueResult(
                quality_score=result["quality_score"],
                failure_modes=failure_modes,
                strengths=result.get("strengths", []),
                weaknesses=result.get("weaknesses", []),
                improvement_suggestions=result.get("improvement_suggestions", []),
                prompt_adherence=result["prompt_adherence"],
            )
        
        return result
