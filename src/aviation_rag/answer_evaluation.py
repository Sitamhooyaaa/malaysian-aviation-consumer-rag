"""Answer-quality evaluation summaries."""

import pandas as pd


REQUIRED_SCORE_COLUMNS = {
    "question_id",
    "answerable",
    "required_facts_met",
    "required_facts_total",
    "required_fact_coverage",
    "citation_validity_pass",
    "citation_support_pass",
    "unsupported_claim_count",
    "faithfulness_pass",
    "forbidden_inference_pass",
    "refusal_correct",
}


def classify_answer_outcomes(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Add strict answer and refusal outcomes."""

    missing_columns = (
        REQUIRED_SCORE_COLUMNS
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "Score data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if scores.empty:
        raise ValueError(
            "Score data cannot be empty"
        )

    if not scores["question_id"].is_unique:
        raise ValueError(
            "question_id values must be unique"
        )

    classified = scores.copy()

    answerable = classified[
        "answerable"
    ].eq(True)

    classified["full_pass"] = False

    classified.loc[
        answerable,
        "full_pass",
    ] = (
        classified.loc[
            answerable,
            "required_fact_coverage",
        ].eq(1.0)
        & classified.loc[
            answerable,
            "citation_validity_pass",
        ].eq(True)
        & classified.loc[
            answerable,
            "citation_support_pass",
        ].eq(True)
        & classified.loc[
            answerable,
            "faithfulness_pass",
        ].eq(True)
        & classified.loc[
            answerable,
            "forbidden_inference_pass",
        ].eq(True)
    )

    refusal = ~answerable

    refusal_fact_complete = (
        classified[
            "required_facts_total"
        ].eq(0)
        | classified[
            "required_fact_coverage"
        ].eq(1.0)
    )

    if (
        "refusal_expected_behavior_pass"
        in classified.columns
    ):
        expected_behavior_pass = (
            classified[
                "refusal_expected_behavior_pass"
            ].fillna(True).eq(True)
        )
    else:
        expected_behavior_pass = pd.Series(
            True,
            index=classified.index,
        )

    classified[
        "clean_refusal_pass"
    ] = False

    classified.loc[
        refusal,
        "clean_refusal_pass",
    ] = (
        classified.loc[
            refusal,
            "refusal_correct",
        ].eq(True)
        & refusal_fact_complete.loc[
            refusal
        ]
        & expected_behavior_pass.loc[
            refusal
        ]
        & classified.loc[
            refusal,
            "unsupported_claim_count",
        ].eq(0)
        & classified.loc[
            refusal,
            "forbidden_inference_pass",
        ].eq(True)
    )

    return classified


def summarize_answer_scores(
    scores: pd.DataFrame,
) -> dict:
    """Calculate end-to-end answer metrics."""

    classified = classify_answer_outcomes(
        scores
    )

    answerable = classified.loc[
        classified["answerable"].eq(True)
    ]

    refusals = classified.loc[
        classified["answerable"].eq(False)
    ]

    if answerable.empty:
        raise ValueError(
            "At least one answerable score "
            "is required"
        )

    if refusals.empty:
        raise ValueError(
            "At least one refusal score "
            "is required"
        )

    return {
        "answerable_questions": len(
            answerable
        ),
        "answerable_micro_fact_coverage": (
            answerable[
                "required_facts_met"
            ].sum()
            / answerable[
                "required_facts_total"
            ].sum()
        ),
        "answerable_fact_complete_rate": (
            answerable[
                "required_fact_coverage"
            ].eq(1.0).mean()
        ),
        "answerable_full_pass_rate": (
            answerable["full_pass"].mean()
        ),
        "answerable_citation_validity_rate": (
            answerable[
                "citation_validity_pass"
            ].mean()
        ),
        "answerable_citation_support_rate": (
            answerable[
                "citation_support_pass"
            ].mean()
        ),
        "answerable_faithfulness_rate": (
            answerable[
                "faithfulness_pass"
            ].mean()
        ),
        "refusal_questions": len(
            refusals
        ),
        "correct_refusal_rate": (
            refusals[
                "refusal_correct"
            ].mean()
        ),
        "clean_refusal_pass_rate": (
            refusals[
                "clean_refusal_pass"
            ].mean()
        ),
    }