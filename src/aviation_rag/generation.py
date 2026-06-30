"""Grounded answer-generation utilities."""

import pandas as pd
import re

import pandas as pd
from typing import Any
from typing import Any, Callable

REQUIRED_RETRIEVAL_COLUMNS = {
    "document_id",
    "page_number",
    "chunk_text",
}


def format_retrieved_context(
    retrieved: pd.DataFrame,
) -> str:
    """Format ranked chunks as cited evidence."""

    missing_columns = (
        REQUIRED_RETRIEVAL_COLUMNS
        - set(retrieved.columns)
    )

    if missing_columns:
        raise ValueError(
            "Retrieved results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if retrieved.empty:
        raise ValueError(
            "Retrieved results cannot be empty"
        )

    source_blocks = []

    for source_number, row in enumerate(
        retrieved.itertuples(index=False),
        start=1,
    ):
        source_blocks.append(
            "\n".join(
                [
                    f"[SOURCE {source_number}]",
                    (
                        "Citation: "
                        f"[{row.document_id}, "
                        f"p. {int(row.page_number)}]"
                    ),
                    "Text:",
                    row.chunk_text,
                ]
            )
        )

    return "\n\n".join(source_blocks)

def build_answer_messages(
    question: str,
    context: str,
    system_instructions: str,
) -> list[dict[str, str]]:
    """Build provider-independent RAG messages."""

    if not question.strip():
        raise ValueError(
            "Question cannot be empty"
        )

    if not context.strip():
        raise ValueError(
            "Context cannot be empty"
        )

    if not system_instructions.strip():
        raise ValueError(
            "System instructions cannot be empty"
        )

    user_message = f"""
Question:
{question}

Evidence:
{context}

Provide a grounded answer with page citations.
""".strip()

    return [
        {
            "role": "system",
            "content": system_instructions,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

VALID_CITATION_PATTERN = re.compile(
    r"\[([A-Za-z0-9_-]+),\s*p\.\s*(\d+)\]"
)

CITATION_ATTEMPT_PATTERN = re.compile(
    r"\[[A-Za-z0-9_-]+,\s*p{1,2}\.[^\]]+\]"
)


def validate_citations(
    answer: str,
    retrieved: pd.DataFrame,
) -> dict:
    """Validate generated citations against retrieval."""

    required_columns = {
        "document_id",
        "page_number",
    }

    missing_columns = (
        required_columns
        - set(retrieved.columns)
    )

    if missing_columns:
        raise ValueError(
            "Retrieved results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    citation_attempts = (
        CITATION_ATTEMPT_PATTERN.findall(
            answer
        )
    )

    valid_matches = list(
        VALID_CITATION_PATTERN.finditer(
            answer
        )
    )

    cited_sources = sorted(
        {
            (
                match.group(1),
                int(match.group(2)),
            )
            for match in valid_matches
        }
    )

    retrieved_sources = {
        (
            row.document_id,
            int(row.page_number),
        )
        for row in retrieved.itertuples(
            index=False
        )
    }

    invalid_citations = [
        source
        for source in cited_sources
        if source not in retrieved_sources
    ]

    malformed_citations = [
        citation
        for citation in citation_attempts
        if VALID_CITATION_PATTERN.fullmatch(
            citation
        )
        is None
    ]

    return {
        "citation_count": len(
            cited_sources
        ),
        "cited_sources": cited_sources,
        "invalid_citations": (
            invalid_citations
        ),
        "malformed_citations": (
            malformed_citations
        ),
        "has_citation": bool(
            citation_attempts
        ),
        "all_citations_valid": (
            bool(citation_attempts)
            and not invalid_citations
            and not malformed_citations
        ),
    }

def prepare_grounded_request(
    question: str,
    retriever: Any,
    top_k: int,
    system_instructions: str,
) -> dict:
    """Retrieve evidence and prepare messages."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    retrieved = retriever.retrieve(
        query=question,
        top_k=top_k,
    )

    context = format_retrieved_context(
        retrieved
    )

    messages = build_answer_messages(
        question=question,
        context=context,
        system_instructions=(
            system_instructions
        ),
    )

    return {
        "messages": messages,
        "context": context,
        "retrieved": retrieved,
    }

def generate_grounded_answer(
    question: str,
    retriever: Any,
    generator: Callable[
        [list[dict[str, str]]],
        dict,
    ],
    system_instructions: str,
    top_k: int,
) -> dict:
    """Retrieve, generate, and validate one answer."""

    prepared = prepare_grounded_request(
        question=question,
        retriever=retriever,
        top_k=top_k,
        system_instructions=(
            system_instructions
        ),
    )

    generated = generator(
        prepared["messages"]
    )

    answer = generated.get("answer")

    if not isinstance(answer, str):
        raise ValueError(
            "Generator result must contain "
            "an answer string"
        )

    citation_check = validate_citations(
        answer=answer,
        retrieved=prepared["retrieved"],
    )

    return {
        **generated,
        "question": question,
        "context": prepared["context"],
        "retrieved": prepared["retrieved"],
        "citation_check": citation_check,
    }