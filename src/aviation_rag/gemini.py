"""Gemini answer-generation adapter."""

from typing import Any

from google.genai import types


def generate_with_gemini(
    messages: list[dict[str, str]],
    client: Any,
    model: str,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int,
) -> dict:
    """Generate one answer using Gemini."""

    if len(messages) != 2:
        raise ValueError(
            "Exactly two messages are required"
        )

    if messages[0]["role"] != "system":
        raise ValueError(
            "First message must be system"
        )

    if messages[1]["role"] != "user":
        raise ValueError(
            "Second message must be user"
        )

    if not model.strip():
        raise ValueError(
            "Model cannot be empty"
        )

    if max_output_tokens <= 0:
        raise ValueError(
            "max_output_tokens must be positive"
        )

    response = (
        client.models.generate_content(
            model=model,
            contents=messages[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=(
                    messages[0]["content"]
                ),
                temperature=temperature,
                max_output_tokens=(
                    max_output_tokens
                ),
                thinking_config=(
                    types.ThinkingConfig(
                        thinking_budget=(
                            thinking_budget
                        )
                    )
                ),
            ),
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned no answer text"
        )

    return {
        "answer": response.text.strip(),
        "finish_reason": str(
            response.candidates[
                0
            ].finish_reason
        ),
        "usage_metadata": (
            response.usage_metadata
        ),
    }