"""Runtime configuration and application assembly."""

import hashlib
import json
from pathlib import Path
from pyexpat.errors import messages
from typing import Any

import pandas as pd
from aviation_rag.gemini import generate_with_gemini

from aviation_rag.retrieval import (
    HybridRetriever,
    SemanticRetriever,
    TfidfRetriever,
)

from collections.abc import Callable
from functools import partial

from aviation_rag.generation import generate_grounded_answer
from aviation_rag.retry import call_with_retry

REQUIRED_GENERATION_CONFIG_KEYS = {
    "prompt_version",
    "system_prompt_sha256",
    "system_instructions",
    "model",
    "temperature",
    "max_output_tokens",
    "thinking_budget",
    "retrieval_top_k",
}


def load_generation_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load and validate the frozen generation configuration."""

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError(
            "Generation configuration must be a JSON object"
        )

    missing_keys = (
        REQUIRED_GENERATION_CONFIG_KEYS - set(config)
    )

    if missing_keys:
        raise ValueError(
            "Generation configuration is missing keys: "
            f"{sorted(missing_keys)}"
        )

    prompt = config["system_instructions"]

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "System instructions must be a non-empty string"
        )

    actual_hash = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    if actual_hash != config["system_prompt_sha256"]:
        raise ValueError(
            "System prompt SHA-256 does not match configuration"
        )

    if config["retrieval_top_k"] <= 0:
        raise ValueError(
            "retrieval_top_k must be greater than zero"
        )

    return config

REQUIRED_RETRIEVAL_CONFIG_KEYS = {
    "status",
    "semantic_retriever",
    "hybrid_retriever",
}


def load_retrieval_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load and validate the frozen retrieval configuration."""

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError(
            "Retrieval configuration must be a JSON object"
        )

    missing_keys = (
        REQUIRED_RETRIEVAL_CONFIG_KEYS - set(config)
    )

    if missing_keys:
        raise ValueError(
            "Retrieval configuration is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if (
        config["status"]
        != "locked_retrieval_evaluation_completed"
    ):
        raise ValueError(
            "Retrieval configuration has not completed "
            "locked evaluation"
        )

    return config


def build_hybrid_retriever(
    chunks: pd.DataFrame,
    config: dict[str, Any],
    semantic_model: Any,
) -> HybridRetriever:
    """Build the evaluated hybrid retriever."""

    semantic_config = config["semantic_retriever"]
    hybrid_config = config["hybrid_retriever"]

    lexical_retriever = TfidfRetriever(chunks)

    semantic_retriever = SemanticRetriever(
        chunks=chunks,
        model=semantic_model,
        query_prefix=semantic_config[
            "query_instruction"
        ],
    )

    return HybridRetriever(
        lexical_retriever=lexical_retriever,
        semantic_retriever=semantic_retriever,
        candidate_k=hybrid_config["candidate_k"],
        rrf_constant=hybrid_config["rrf_constant"],
        lexical_weight=hybrid_config["lexical_weight"],
    )

def build_answer_service(
    retriever: Any,
    generator: Callable[
        [list[dict[str, str]]],
        dict[str, Any],
    ],
    generation_config: dict[str, Any],
) -> Callable[[str], dict[str, Any]]:
    """Bind the frozen components into one question service."""

    return partial(
        generate_grounded_answer,
        retriever=retriever,
        generator=generator,
        system_instructions=generation_config[
            "system_instructions"
        ],
        top_k=generation_config["retrieval_top_k"],
    )

def build_runtime_answer_service(
    chunks_path: Path,
    retrieval_config_path: Path,
    generation_config_path: Path,
    gemini_client: Any,
    semantic_model_loader: (
        Callable[[str], Any] | None
    ) = None,
    retry_max_attempts: int = 2,
    retry_base_wait_seconds: float = 2,
) -> Callable[[str], dict[str, Any]]:
    """Load frozen artifacts and build the answer service."""

    retrieval_config = load_retrieval_config(
        retrieval_config_path
    )
    generation_config = load_generation_config(
        generation_config_path
    )

    chunks = pd.read_json(
        chunks_path,
        lines=True,
    )

    if semantic_model_loader is None:
        from sentence_transformers import (
            SentenceTransformer,
        )

        semantic_model_loader = SentenceTransformer

    model_name = retrieval_config[
        "semantic_retriever"
    ]["model"]

    semantic_model = semantic_model_loader(model_name)

    retriever = build_hybrid_retriever(
        chunks=chunks,
        config=retrieval_config,
        semantic_model=semantic_model,
    )

    raw_generator = partial(
    generate_with_gemini,
    client=gemini_client,
    model=generation_config["model"],
    temperature=generation_config["temperature"],
    max_output_tokens=generation_config[
        "max_output_tokens"
    ],
    thinking_budget=generation_config[
        "thinking_budget"
    ],
)
    def generator(
    messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return call_with_retry(
        operation=lambda: raw_generator(messages),
        max_attempts=retry_max_attempts,
        base_wait_seconds=retry_base_wait_seconds,
    )

    return build_answer_service(
        retriever=retriever,
        generator=generator,
        generation_config=generation_config,
    )