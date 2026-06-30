"""JSONL checkpoint utilities."""

import json
from pathlib import Path


def load_jsonl(
    path: Path,
) -> list[dict]:
    """Load JSON objects from a JSONL file."""

    if not path.exists():
        return []

    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid JSON on line "
                f"{line_number} of {path}"
            ) from error

        if not isinstance(record, dict):
            raise ValueError(
                "Each JSONL record must be "
                "an object"
            )

        records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    """Write JSONL records atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(path)


def upsert_record(
    records: list[dict],
    new_record: dict,
    id_field: str = "question_id",
) -> list[dict]:
    """Insert or replace one identified record."""

    if id_field not in new_record:
        raise ValueError(
            f"New record is missing {id_field}"
        )

    existing_ids = [
        record.get(id_field)
        for record in records
    ]

    if len(existing_ids) != len(
        set(existing_ids)
    ):
        raise ValueError(
            f"Existing {id_field} values "
            "must be unique"
        )

    updated_records = [
        record.copy()
        for record in records
    ]

    new_id = new_record[id_field]

    for index, record in enumerate(
        updated_records
    ):
        if record.get(id_field) == new_id:
            updated_records[index] = (
                new_record.copy()
            )
            break
    else:
        updated_records.append(
            new_record.copy()
        )

    return updated_records