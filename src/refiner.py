"""Prompt refinement based on critique feedback using LangChain."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_chat_model
from .schemas import CritiqueResult, RefinedPrompt


REFINER_SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generation. Your task is to refine prompts based on critique feedback.

Guidelines for prompt refinement:
1. **Preserve Intent**: Keep the core concept and artistic vision intact
2. **Be Specific**: Replace vague terms with precise descriptions
3. **Add Details**: Include specific details for problem areas
4. **Use Effective Keywords**: Add style keywords, quality boosters, and technical terms
5. **Address Failures**: Directly counter identified failure modes
6. **Stay Concise**: Don't make prompts excessively long

Effective prompt patterns:
- Subject description → Setting/Environment → Style → Technical quality terms
- Use commas to separate concepts
- Include lighting descriptions when relevant
- Specify camera angle/perspective if needed
- Add artist references or style keywords for consistency

For anatomical errors: Add "correct anatomy, proper proportions, natural pose"
For composition issues: Specify "centered composition, rule of thirds, balanced framing"
For lighting: Add specific lighting terms like "soft lighting, golden hour, studio lighting"
For coherence: Clarify relationships between elements

Only update the negative prompt if there are specific things to avoid that aren't already covered."""


class PromptRefiner:
    """Refines prompts based on image critique feedback using LangChain."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] | None = None,
    ):
        """Initialize the refiner.

        Args:
            provider: LLM provider to use. Uses config default if None.
        """
        self.provider = provider
        self._model = None

    @property
    def model(self):
        """Lazy-load the model with structured output."""
        if self._model is None:
            base_model = get_chat_model(provider=self.provider)
            self._model = base_model.with_structured_output(RefinedPrompt)
        return self._model

    def refine(
        self,
        original_prompt: str,
        critique: CritiqueResult,
        current_negative_prompt: str | None = None,
        iteration: int = 0,
    ) -> RefinedPrompt:
        """Refine a prompt based on critique feedback.

        Args:
            original_prompt: The prompt that generated the critiqued image.
            critique: The structured critique of the generated image.
            current_negative_prompt: Current negative prompt (if any).
            iteration: Current iteration number.

        Returns:
            Refined prompt with reasoning.
        """
        failure_modes_str = ", ".join(
            mode.value.replace("_", " ") for mode in critique.failure_modes
        ) or "None identified"

        user_message = f"""Refine this image generation prompt based on the critique.

**Current Prompt**: "{original_prompt}"

**Current Negative Prompt**: "{current_negative_prompt or 'None'}"

**Critique Summary**:
- Quality Score: {critique.quality_score}/10
- Prompt Adherence: {critique.prompt_adherence}/10
- Failure Modes: {failure_modes_str}

**Weaknesses**:
{chr(10).join(f"- {w}" for w in critique.weaknesses) or "- None identified"}

**Improvement Suggestions**:
{chr(10).join(f"- {s}" for s in critique.improvement_suggestions) or "- None provided"}

**Iteration**: {iteration + 1}

Create an improved prompt that addresses these issues while preserving the original intent."""

        messages = [
            SystemMessage(content=REFINER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        result = self.model.invoke(messages)
        
        if isinstance(result, RefinedPrompt):
            return result
        
        # Handle dict response (fallback)
        if isinstance(result, dict):
            return RefinedPrompt(
                new_prompt=result["new_prompt"],
                reasoning=result["reasoning"],
                changes_made=result.get("changes_made", []),
                negative_prompt=result.get("negative_prompt"),
            )
        
        return result
