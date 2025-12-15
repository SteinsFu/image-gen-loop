"""SD3 image generation via Diffusers."""

import random
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

import config


class ImageGenerator:
    """Generates images using Stable Diffusion 3."""

    def __init__(
        self,
        model_id: str = config.SD3_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the SD3 pipeline.

        Args:
            model_id: HuggingFace model ID for SD3.
            device: Device to run inference on.
            dtype: Torch dtype for model weights.
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self._pipeline: Optional[StableDiffusion3Pipeline] = None

    @property
    def pipeline(self) -> StableDiffusion3Pipeline:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is None:
            self._pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )
            self._pipeline.to(self.device)
            # Enable memory optimizations
            self._pipeline.enable_attention_slicing()
        return self._pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = config.NUM_INFERENCE_STEPS,
        guidance_scale: float = config.GUIDANCE_SCALE,
        width: int = config.IMAGE_SIZE,
        height: int = config.IMAGE_SIZE,
    ) -> tuple[Image.Image, int]:
        """Generate an image from a prompt.

        Args:
            prompt: The text prompt for image generation.
            negative_prompt: Things to avoid in the image.
            seed: Random seed for reproducibility. Random if None.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output image width.
            height: Output image height.

        Returns:
            Tuple of (generated PIL Image, seed used).
        """
        # Handle seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Use default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = config.DEFAULT_NEGATIVE_PROMPT

        # Generate image
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        image = result.images[0]
        return image, seed

    def generate_and_save(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> tuple[Path, int]:
        """Generate an image and save it to disk.

        Args:
            prompt: The text prompt for image generation.
            output_path: Path to save the image.
            negative_prompt: Things to avoid in the image.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to generate().

        Returns:
            Tuple of (saved image path, seed used).
        """
        image, used_seed = self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            **kwargs,
        )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        return output_path, used_seed

    def clear_cache(self) -> None:
        """Clear CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

