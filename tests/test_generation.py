import pandas as pd
import pytest

from aviation_rag.generation import (
    build_answer_messages,
    format_retrieved_context,
    generate_grounded_answer,
    prepare_grounded_request,
    validate_citations,
)


def test_format_retrieved_context():
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
                "chunk_text": "First evidence.",
            },
            {
                "document_id": "document_b",
                "page_number": 20,
                "chunk_text": "Second evidence.",
            },
        ]
    )

    context = format_retrieved_context(
        retrieved
    )

    assert "[SOURCE 1]" in context
    assert "[document_a, p. 10]" in context
    assert "First evidence." in context
    assert "[SOURCE 2]" in context
    assert "[document_b, p. 20]" in context
    assert "Second evidence." in context


def test_format_retrieved_context_rejects_missing_columns():
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        format_retrieved_context(retrieved)


def test_format_retrieved_context_rejects_empty_results():
    retrieved = pd.DataFrame(
        columns=[
            "document_id",
            "page_number",
            "chunk_text",
        ]
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        format_retrieved_context(retrieved)

def test_build_answer_messages():
    messages = build_answer_messages(
        question="What is the rule?",
        context="Evidence text.",
        system_instructions=(
            "Answer only from evidence."
        ),
    )

    assert len(messages) == 2
    assert messages[0] == {
        "role": "system",
        "content": (
            "Answer only from evidence."
        ),
    }
    assert messages[1]["role"] == "user"
    assert "What is the rule?" in (
        messages[1]["content"]
    )
    assert "Evidence text." in (
        messages[1]["content"]
    )


@pytest.mark.parametrize(
    (
        "question",
        "context",
        "system_instructions",
        "error_message",
    ),
    [
        (
            "",
            "Evidence",
            "Instructions",
            "Question cannot be empty",
        ),
        (
            "Question",
            "",
            "Instructions",
            "Context cannot be empty",
        ),
        (
            "Question",
            "Evidence",
            "",
            "System instructions cannot be empty",
        ),
    ],
)
def test_build_answer_messages_rejects_empty_input(
    question,
    context,
    system_instructions,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        build_answer_messages(
            question=question,
            context=context,
            system_instructions=(
                system_instructions
            ),
        )

def test_validate_citations_accepts_retrieved_source():
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
            }
        ]
    )

    result = validate_citations(
        answer=(
            "The rule applies "
            "[document_a, p. 10]."
        ),
        retrieved=retrieved,
    )

    assert result["citation_count"] == 1
    assert result["cited_sources"] == [
        ("document_a", 10)
    ]
    assert result["invalid_citations"] == []
    assert result["malformed_citations"] == []
    assert result["all_citations_valid"]


def test_validate_citations_rejects_unretrieved_source():
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
            }
        ]
    )

    result = validate_citations(
        answer="[document_b, p. 20]",
        retrieved=retrieved,
    )

    assert result["invalid_citations"] == [
        ("document_b", 20)
    ]
    assert not result["all_citations_valid"]


@pytest.mark.parametrize(
    "citation",
    [
        "[document_a, pp. 10, 11]",
        "[document_a, p. 10, 11]",
    ],
)
def test_validate_citations_rejects_malformed_citation(
    citation,
):
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
            },
            {
                "document_id": "document_a",
                "page_number": 11,
            },
        ]
    )

    result = validate_citations(
        answer=citation,
        retrieved=retrieved,
    )

    assert result["has_citation"]
    assert result["malformed_citations"] == [
        citation
    ]
    assert not result["all_citations_valid"]


def test_validate_citations_handles_no_citation():
    retrieved = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 10,
            }
        ]
    )

    result = validate_citations(
        answer="The evidence is insufficient.",
        retrieved=retrieved,
    )

    assert not result["has_citation"]
    assert not result["all_citations_valid"]


def test_validate_citations_rejects_missing_columns():
    retrieved = pd.DataFrame(
        [{"document_id": "document_a"}]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_citations(
            answer="[document_a, p. 10]",
            retrieved=retrieved,
        )

class FakeRetriever:
    def __init__(self):
        self.query = None
        self.top_k = None

    def retrieve(
        self,
        query,
        top_k,
    ):
        self.query = query
        self.top_k = top_k

        return pd.DataFrame(
            [
                {
                    "document_id": "document_a",
                    "page_number": 10,
                    "chunk_text": "Evidence text.",
                }
            ]
        )


def test_prepare_grounded_request():
    retriever = FakeRetriever()

    result = prepare_grounded_request(
        question="What is the rule?",
        retriever=retriever,
        top_k=5,
        system_instructions=(
            "Answer only from evidence."
        ),
    )

    assert retriever.query == (
        "What is the rule?"
    )
    assert retriever.top_k == 5
    assert len(result["retrieved"]) == 1
    assert "[document_a, p. 10]" in (
        result["context"]
    )
    assert result["messages"][0][
        "role"
    ] == "system"


def test_prepare_grounded_request_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        prepare_grounded_request(
            question="Question",
            retriever=FakeRetriever(),
            top_k=0,
            system_instructions=(
                "Instructions"
            ),
        )

def test_generate_grounded_answer():
    retriever = FakeRetriever()
    received_messages = []

    def fake_generator(messages):
        received_messages.extend(messages)

        return {
            "answer": (
                "Grounded answer "
                "[document_a, p. 10]."
            ),
            "finish_reason": "STOP",
        }

    result = generate_grounded_answer(
        question="What is the rule?",
        retriever=retriever,
        generator=fake_generator,
        system_instructions=(
            "Answer only from evidence."
        ),
        top_k=5,
    )

    assert result["question"] == (
        "What is the rule?"
    )
    assert result["answer"].startswith(
        "Grounded answer"
    )
    assert result["finish_reason"] == "STOP"
    assert len(received_messages) == 2
    assert len(result["retrieved"]) == 1
    assert result["citation_check"][
        "all_citations_valid"
    ]


def test_generate_grounded_answer_rejects_missing_answer():
    def invalid_generator(messages):
        return {
            "finish_reason": "STOP"
        }

    with pytest.raises(
        ValueError,
        match="answer string",
    ):
        generate_grounded_answer(
            question="What is the rule?",
            retriever=FakeRetriever(),
            generator=invalid_generator,
            system_instructions=(
                "Answer only from evidence."
            ),
            top_k=5,
        )