from types import SimpleNamespace

import pytest

from aviation_rag.gemini import (
    generate_with_gemini,
    GeminiServiceError
)


VALID_MESSAGES = [
    {
        "role": "system",
        "content": "Use only evidence.",
    },
    {
        "role": "user",
        "content": "Question and evidence.",
    },
]


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


def test_generate_with_gemini():
    usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=20,
    )

    response = SimpleNamespace(
        text=" Grounded answer. ",
        candidates=[
            SimpleNamespace(
                finish_reason="STOP"
            )
        ],
        usage_metadata=usage,
    )

    client = FakeClient(response)

    result = generate_with_gemini(
        messages=VALID_MESSAGES,
        client=client,
        model="gemini-test",
        temperature=0.0,
        max_output_tokens=500,
        thinking_budget=0,
    )

    assert result["answer"] == (
        "Grounded answer."
    )
    assert result["finish_reason"] == "STOP"
    assert result["usage_metadata"] is usage

    call = client.models.calls[0]

    assert call["model"] == "gemini-test"
    assert call["contents"] == (
        "Question and evidence."
    )
    assert call["config"].system_instruction == (
        "Use only evidence."
    )
    assert call["config"].temperature == 0.0
    assert (
        call["config"].max_output_tokens
        == 500
    )
    assert (
        call["config"]
        .thinking_config
        .thinking_budget
        == 0
    )


@pytest.mark.parametrize(
    ("messages", "error_message"),
    [
        (
            [],
            "Exactly two messages",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "Wrong role.",
                },
                VALID_MESSAGES[1],
            ],
            "First message must be system",
        ),
        (
            [
                VALID_MESSAGES[0],
                {
                    "role": "system",
                    "content": "Wrong role.",
                },
            ],
            "Second message must be user",
        ),
    ],
)
def test_generate_with_gemini_rejects_invalid_messages(
    messages,
    error_message,
):
    client = FakeClient(response=None)

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        generate_with_gemini(
            messages=messages,
            client=client,
            model="gemini-test",
            temperature=0.0,
            max_output_tokens=500,
            thinking_budget=0,
        )


def test_generate_with_gemini_rejects_empty_response():
    response = SimpleNamespace(
        text="",
        candidates=[],
        usage_metadata=None,
    )

    client = FakeClient(response)

    with pytest.raises(
        RuntimeError,
        match="no answer text",
    ):
        generate_with_gemini(
            messages=VALID_MESSAGES,
            client=client,
            model="gemini-test",
            temperature=0.0,
            max_output_tokens=500,
            thinking_budget=0,
        )

def test_generate_with_gemini_wraps_provider_error():
    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("503 UNAVAILABLE")

    client = SimpleNamespace(
        models=FailingModels()
    )

    with pytest.raises(
        GeminiServiceError,
        match="503 UNAVAILABLE",
    ):
        generate_with_gemini(
            messages=[
                {
                    "role": "system",
                    "content": "Use evidence.",
                },
                {
                    "role": "user",
                    "content": "Question and evidence.",
                },
            ],
            client=client,
            model="gemini-test",
            temperature=0.0,
            max_output_tokens=100,
            thinking_budget=0,
        )