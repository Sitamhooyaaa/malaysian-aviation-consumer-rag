from fastapi.testclient import TestClient

from aviation_rag.api import create_app
from aviation_rag.gemini import GeminiServiceError


def test_health_check_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_grounded_answer() -> None:
    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "The Code started on 1 July 2016.",
            "citation_check": {
                "cited_sources": [
                    ("macpc_principal_2016", 39)
                ]
            },
        }

    client = TestClient(
        create_app(
            answer_service=fake_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "When did the Code start?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "When did the Code start?",
        "answer": "The Code started on 1 July 2016.",
        "citations": [
            {
                "document_id": "macpc_principal_2016",
                "page_number": 39,
            }
        ],
    }


def test_ask_without_service_returns_503() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={"question": "When did the Code start?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Answer service is not configured"
    }


def test_ask_rejects_blank_question() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422

def test_ask_rejects_invalid_generated_citation() -> None:
    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "Unsupported claim [fake_doc, p. 99].",
            "citation_check": {
                "cited_sources": [
                    ("fake_doc", 99)
                ],
                "invalid_citations": [
                    ("fake_doc", 99)
                ],
                "malformed_citations": [],
            },
        }

    client = TestClient(
        create_app(
            answer_service=fake_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is the rule?"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Generated answer failed citation validation"
        )
    }

def test_ask_returns_503_when_gemini_fails() -> None:
    def failing_answer_service(question: str) -> dict:
        raise GeminiServiceError(
            "Gemini request failed: 503 UNAVAILABLE"
        )

    client = TestClient(
        create_app(
            answer_service=failing_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is the rule?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Generation service is temporarily unavailable"
        )
    }