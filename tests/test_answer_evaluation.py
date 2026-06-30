import pandas as pd
import pytest

from aviation_rag.answer_evaluation import (
    classify_answer_outcomes,
    summarize_answer_scores,
)


def make_scores():
    return pd.DataFrame(
        [
            {
                "question_id": "A1",
                "answerable": True,
                "required_facts_met": 2,
                "required_facts_total": 2,
                "required_fact_coverage": 1.0,
                "citation_validity_pass": True,
                "citation_support_pass": True,
                "unsupported_claim_count": 0,
                "faithfulness_pass": True,
                "forbidden_inference_pass": True,
                "refusal_correct": None,
                "refusal_expected_behavior_pass": None,
            },
            {
                "question_id": "A2",
                "answerable": True,
                "required_facts_met": 1,
                "required_facts_total": 2,
                "required_fact_coverage": 0.5,
                "citation_validity_pass": True,
                "citation_support_pass": True,
                "unsupported_claim_count": 0,
                "faithfulness_pass": True,
                "forbidden_inference_pass": True,
                "refusal_correct": None,
                "refusal_expected_behavior_pass": None,
            },
            {
                "question_id": "R1",
                "answerable": False,
                "required_facts_met": 0,
                "required_facts_total": 0,
                "required_fact_coverage": None,
                "citation_validity_pass": None,
                "citation_support_pass": None,
                "unsupported_claim_count": 0,
                "faithfulness_pass": True,
                "forbidden_inference_pass": True,
                "refusal_correct": True,
                "refusal_expected_behavior_pass": True,
            },
            {
                "question_id": "R2",
                "answerable": False,
                "required_facts_met": 0,
                "required_facts_total": 0,
                "required_fact_coverage": None,
                "citation_validity_pass": None,
                "citation_support_pass": None,
                "unsupported_claim_count": 0,
                "faithfulness_pass": True,
                "forbidden_inference_pass": True,
                "refusal_correct": True,
                "refusal_expected_behavior_pass": False,
            },
        ]
    )


def test_classify_answer_outcomes():
    classified = classify_answer_outcomes(
        make_scores()
    ).set_index("question_id")

    assert classified.loc[
        "A1",
        "full_pass",
    ]
    assert not classified.loc[
        "A2",
        "full_pass",
    ]
    assert classified.loc[
        "R1",
        "clean_refusal_pass",
    ]
    assert not classified.loc[
        "R2",
        "clean_refusal_pass",
    ]


def test_summarize_answer_scores():
    summary = summarize_answer_scores(
        make_scores()
    )

    assert summary[
        "answerable_questions"
    ] == 2
    assert summary[
        "answerable_micro_fact_coverage"
    ] == pytest.approx(0.75)
    assert summary[
        "answerable_fact_complete_rate"
    ] == pytest.approx(0.5)
    assert summary[
        "answerable_full_pass_rate"
    ] == pytest.approx(0.5)
    assert summary[
        "correct_refusal_rate"
    ] == pytest.approx(1.0)
    assert summary[
        "clean_refusal_pass_rate"
    ] == pytest.approx(0.5)


def test_refusal_scoring_without_expected_behavior_column():
    scores = make_scores().drop(
        columns=[
            "refusal_expected_behavior_pass"
        ]
    )

    classified = classify_answer_outcomes(
        scores
    ).set_index("question_id")

    assert classified.loc[
        "R1",
        "clean_refusal_pass",
    ]
    assert classified.loc[
        "R2",
        "clean_refusal_pass",
    ]


def test_classification_rejects_missing_columns():
    scores = make_scores().drop(
        columns=["citation_support_pass"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        classify_answer_outcomes(scores)


def test_classification_rejects_duplicate_ids():
    scores = pd.concat(
        [
            make_scores(),
            make_scores().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        classify_answer_outcomes(scores)


@pytest.mark.parametrize(
    "answerable_value",
    [True, False],
)
def test_summary_requires_both_question_types(
    answerable_value,
):
    scores = make_scores().loc[
        make_scores()["answerable"].eq(
            answerable_value
        )
    ]

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        summarize_answer_scores(scores)