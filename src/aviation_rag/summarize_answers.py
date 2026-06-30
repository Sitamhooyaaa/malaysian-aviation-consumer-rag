"""Summarize saved answer-evaluation scores."""

import argparse
import json
from pathlib import Path

import pandas as pd

from aviation_rag.answer_evaluation import (
    summarize_answer_scores,
)
from aviation_rag.checkpoint import (
    load_jsonl,
)


def summarize_scores_file(
    scores_path: Path,
) -> dict:
    """Load and summarize one score file."""

    records = load_jsonl(
        scores_path
    )

    if not records:
        raise ValueError(
            "Score file contains no records"
        )

    scores = pd.DataFrame(records)

    return summarize_answer_scores(
        scores
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize RAG answer scores."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Path to a JSONL score file.",
    )

    args = parser.parse_args()

    summary = summarize_scores_file(
        args.scores
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()