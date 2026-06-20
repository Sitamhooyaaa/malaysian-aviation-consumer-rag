"""Tests for page-dataset orchestration."""

import pandas as pd
import pytest
import fitz

from aviation_rag.build_pages import (
    apply_page_exclusions,
    build_page_dataset,
    load_document_manifest,
    load_page_exclusions,
)

def test_load_document_manifest_accepts_valid_file(
    tmp_path,
):
    manifest_path = tmp_path / "documents.csv"

    pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "file_name": "doc.pdf",
                "title": "Test Document",
                "document_type": "consumer_report",
                "authority_level": "official_report",
                "source_url": "https://example.com/doc.pdf",
                "extraction_method": "pymupdf",
                "indexed_start_page": 1,
                "indexed_end_page": 2,
                "include_in_v1": True,
                "gazette_number": None,
                "publication_date": None,
                "effective_from": None,
                "effective_date_note": None,
                "reporting_period_start": "2024-01-01",
                "reporting_period_end": "2024-06-30",
            }
        ]
    ).to_csv(manifest_path, index=False)

    manifest = load_document_manifest(manifest_path)

    assert len(manifest) == 1
    assert manifest.loc[0, "document_id"] == "doc_1"


def test_load_document_manifest_rejects_missing_columns(
    tmp_path,
):
    manifest_path = tmp_path / "documents.csv"

    pd.DataFrame(
        [{"document_id": "doc_1"}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        load_document_manifest(manifest_path)


def test_load_page_exclusions_rejects_duplicates(
    tmp_path,
):
    exclusions_path = tmp_path / "page_exclusions.csv"

    pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "page_number": 1,
                "exclusion_reason": "cover",
            },
            {
                "document_id": "doc_1",
                "page_number": 1,
                "exclusion_reason": "cover",
            },
        ]
    ).to_csv(exclusions_path, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate document-page pairs",
    ):
        load_page_exclusions(exclusions_path)

def test_apply_page_exclusions_marks_matching_page():
    page_records = pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "page_number": 1,
                "text": "Cover",
            },
            {
                "document_id": "doc_1",
                "page_number": 2,
                "text": "Useful evidence",
            },
        ]
    )

    exclusions = pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "page_number": 1,
                "exclusion_reason": "cover",
            }
        ]
    )

    result = apply_page_exclusions(
        page_records,
        exclusions,
    )

    page_1 = result.loc[result["page_number"] == 1].iloc[0]
    page_2 = result.loc[result["page_number"] == 2].iloc[0]

    assert not bool(page_1["retrieval_eligible"])
    assert page_1["exclusion_reason"] == "cover"
    assert bool(page_2["retrieval_eligible"])
    assert pd.isna(page_2["exclusion_reason"])


def test_apply_page_exclusions_rejects_unknown_page():
    page_records = pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "page_number": 1,
                "text": "Useful evidence",
            }
        ]
    )

    exclusions = pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "page_number": 99,
                "exclusion_reason": "cover",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        apply_page_exclusions(
            page_records,
            exclusions,
        )

def test_build_page_dataset_end_to_end(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    manifest_dir = tmp_path / "data" / "manifests"
    output_path = (
        tmp_path
        / "data"
        / "processed"
        / "pages.jsonl"
    )

    raw_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    pdf_path = raw_dir / "test_report.pdf"

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Useful evidence for retrieval.",
        )
        document.save(pdf_path)

    document_manifest_path = (
        manifest_dir / "documents.csv"
    )

    pd.DataFrame(
        [
            {
                "document_id": "test_report",
                "file_name": pdf_path.name,
                "title": "Test Consumer Report",
                "document_type": "consumer_report",
                "authority_level": "official_report",
                "source_url": "https://example.com/test.pdf",
                "extraction_method": "pymupdf",
                "indexed_start_page": 1,
                "indexed_end_page": 1,
                "include_in_v1": True,
                "gazette_number": None,
                "publication_date": None,
                "effective_from": None,
                "effective_date_note": None,
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-06-30",
            }
        ]
    ).to_csv(document_manifest_path, index=False)

    exclusions_path = (
        manifest_dir / "page_exclusions.csv"
    )

    pd.DataFrame(
        columns=[
            "document_id",
            "page_number",
            "exclusion_reason",
        ]
    ).to_csv(exclusions_path, index=False)

    result = build_page_dataset(
        document_manifest_path=document_manifest_path,
        exclusions_path=exclusions_path,
        raw_data_dir=raw_dir,
        output_path=output_path,
    )

    saved_result = pd.read_json(
        output_path,
        lines=True,
    )

    assert output_path.exists()
    assert len(result) == 1
    assert len(saved_result) == 1
    assert bool(result.loc[0, "retrieval_eligible"])
    assert "Useful evidence" in result.loc[0, "text"]