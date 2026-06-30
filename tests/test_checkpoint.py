import pytest

from aviation_rag.checkpoint import (
    load_jsonl,
    upsert_record,
    write_jsonl,
)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"

    records = [
        {
            "question_id": "Q1",
            "answer": "First answer",
        },
        {
            "question_id": "Q2",
            "answer": "Café",
        },
    ]

    write_jsonl(path, records)

    assert load_jsonl(path) == records
    assert not path.with_name(
        path.name + ".tmp"
    ).exists()


def test_load_jsonl_returns_empty_for_missing_file(
    tmp_path,
):
    path = tmp_path / "missing.jsonl"

    assert load_jsonl(path) == []


def test_load_jsonl_reports_invalid_line(
    tmp_path,
):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"question_id": "Q1"}\n'
        "invalid json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="line 2",
    ):
        load_jsonl(path)


def test_load_jsonl_rejects_non_object(
    tmp_path,
):
    path = tmp_path / "list.jsonl"

    path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be an object",
    ):
        load_jsonl(path)


def test_upsert_record_appends_new_record():
    records = [
        {
            "question_id": "Q1",
            "score": 1,
        }
    ]

    updated = upsert_record(
        records=records,
        new_record={
            "question_id": "Q2",
            "score": 2,
        },
    )

    assert len(updated) == 2
    assert updated[1]["question_id"] == "Q2"
    assert len(records) == 1


def test_upsert_record_replaces_existing_record():
    records = [
        {
            "question_id": "Q1",
            "score": 1,
        }
    ]

    updated = upsert_record(
        records=records,
        new_record={
            "question_id": "Q1",
            "score": 5,
        },
    )

    assert updated == [
        {
            "question_id": "Q1",
            "score": 5,
        }
    ]
    assert records[0]["score"] == 1


def test_upsert_record_rejects_duplicate_existing_ids():
    records = [
        {"question_id": "Q1"},
        {"question_id": "Q1"},
    ]

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        upsert_record(
            records=records,
            new_record={
                "question_id": "Q2"
            },
        )