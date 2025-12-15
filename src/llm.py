"""LLM provider abstraction using LangChain."""

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

import config


def get_chat_model(
    provider: Literal["openai", "anthropic"] | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Get a chat model instance based on provider.

    Args:
        provider: LLM provider ("openai" or "anthropic"). Uses config default if None.
        model: Model name. Uses provider default if None.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        LangChain chat model instance.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=config.OPENAI_API_KEY,
            **kwargs,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            api_key=config.ANTHROPIC_API_KEY,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


def get_vision_model(
    provider: Literal["openai", "anthropic"] | None = None,
    **kwargs,
) -> BaseChatModel:
    """Get a vision-capable chat model.

    Args:
        provider: LLM provider. Uses config default if None.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        LangChain chat model with vision capabilities.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",  # Vision-capable
            api_key=config.OPENAI_API_KEY,
            max_tokens=1000,
            **kwargs,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Vision-capable
            api_key=config.ANTHROPIC_API_KEY,
            max_tokens=1000,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

