"""FastAPI application for the aviation RAG service."""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from aviation_rag.gemini import GeminiServiceError


AnswerService = Callable[[str], dict[str, Any]]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        cleaned = " ".join(value.split())

        if not cleaned:
            raise ValueError("Question cannot be blank")

        return cleaned


class CitationResponse(BaseModel):
    document_id: str
    page_number: int


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]


def create_app(
    answer_service: AnswerService | None = None,
    lifespan: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Malaysian Aviation Consumer RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.answer_service = answer_service

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask_question(request: AskRequest) -> AskResponse:
        service = app.state.answer_service

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Answer service is not configured",
            )

        try:
            result = service(request.question)
        except GeminiServiceError as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Generation service is temporarily unavailable"
                ),
            ) from error
        citation_check = result["citation_check"]

        if (
            citation_check.get("invalid_citations")
            or citation_check.get("malformed_citations")
        ):
            raise HTTPException(
                status_code=502,
                detail=(
                    "Generated answer failed citation validation"
                ),
            )

        cited_sources = citation_check["cited_sources"]

        citations = [
            CitationResponse(
                document_id=document_id,
                page_number=page_number,
            )
            for document_id, page_number in cited_sources
        ]

        return AskResponse(
            question=request.question,
            answer=result["answer"],
            citations=citations,
        )

    return app


app = create_app()