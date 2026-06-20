"""Tests for document-cleaning utilities."""

import pytest

from aviation_rag.ingestion import (
    clean_gazette_page_text,
    clean_report_page_text,
    extract_document_pages,
)

def test_clean_gazette_removes_header_and_page_number():
    raw_text = (
        "P.U. (B) 305\n"
        "Legal content remains.\n"
        "39\n"
    )

    cleaned_text, removed_lines = clean_gazette_page_text(
        raw_text,
        pdf_page_number=39,
    )

    assert cleaned_text == "Legal content remains."
    assert set(removed_lines) == {"P.U. (B) 305", "39"}


def test_clean_gazette_preserves_reference_inside_body():
    raw_text = (
        "Legal discussion\n"
        "Reference to P.U. (B) 305\n"
        "More legal content"
    )

    cleaned_text, removed_lines = clean_gazette_page_text(
        raw_text,
        pdf_page_number=10,
    )

    assert "Reference to P.U. (B) 305" in cleaned_text
    assert removed_lines == []


def test_clean_report_removes_verified_noise():
    disclaimer = (
        "DO NOT DUPLICATE OR DISTRIBUTE WITHOUT WRITTEN "
        "PERMISSION FROM MALAYSIAN AVIATION COMMISSION"
    )

    raw_text = (
        "CONSUMER REPORT\n"
        "6\n"
        "Executive Summary\n"
        "Evidence from 2023 and 2024 remains.\n"
        f"{disclaimer}\n"
    )

    cleaned_text, removed_lines = clean_report_page_text(
        raw_text,
        pdf_page_number=6,
    )

    assert cleaned_text == (
        "Executive Summary\n"
        "Evidence from 2023 and 2024 remains."
    )
    assert "CONSUMER REPORT" in removed_lines
    assert "6" in removed_lines
    assert disclaimer in removed_lines

def test_extract_document_pages_rejects_missing_file(
    tmp_path,
):
    manifest_row = {
        "file_name": "missing.pdf",
        "indexed_start_page": 1,
        "indexed_end_page": 1,
        "extraction_method": "pymupdf",
    }

    with pytest.raises(
        FileNotFoundError,
        match="Source PDF not found",
    ):
        extract_document_pages(
            manifest_row,
            raw_data_dir=tmp_path,
        )


def test_extract_document_pages_rejects_unknown_method(
    tmp_path,
):
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"test")

    manifest_row = {
        "file_name": dummy_pdf.name,
        "indexed_start_page": 1,
        "indexed_end_page": 1,
        "extraction_method": "unknown",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported extraction method",
    ):
        extract_document_pages(
            manifest_row,
            raw_data_dir=tmp_path,
        )