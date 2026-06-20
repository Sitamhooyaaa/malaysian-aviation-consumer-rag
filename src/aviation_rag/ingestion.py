"""Document extraction and cleaning utilities."""

import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber


GAZETTE_HEADER_PATTERN = re.compile(
    r"^P\.U\. \(B\) \d+$"
)

REPORT_DISCLAIMER_LINES = {
    (
        "DO NOT DUPLICATE OR DISTRIBUTE WITHOUT WRITTEN "
        "PERMISSION FROM MALAYSIAN AVIATION COMMISSION"
    ),
    (
        "PRIVATE & CONFIDENTIAL - DO NOT DUPLICATE OR "
        "DISTRIBUTE WITHOUT WRITTEN PERMISSION FROM "
        "MALAYSIAN AVIATION COMMISSION"
    ),
}


def clean_gazette_page_text(
    text: str,
    pdf_page_number: int,
) -> tuple[str, list[str]]:
    """Remove verified Gazette noise from one page."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    removed_lines = []

    if lines and GAZETTE_HEADER_PATTERN.fullmatch(lines[0]):
        removed_lines.append(lines.pop(0))

    page_number_text = str(pdf_page_number)

    if lines and lines[0] == page_number_text:
        removed_lines.append(lines.pop(0))

    if lines and lines[-1] == page_number_text:
        removed_lines.append(lines.pop())

    return "\n".join(lines), removed_lines


def clean_report_page_text(
    text: str,
    pdf_page_number: int,
) -> tuple[str, list[str]]:
    """Remove verified repeated noise from one report page."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_lines = []
    removed_lines = []

    for line_index, line in enumerate(lines):
        if line in REPORT_DISCLAIMER_LINES:
            removed_lines.append(line)
            continue

        if line_index < 3 and line == "CONSUMER REPORT":
            removed_lines.append(line)
            continue

        cleaned_lines.append(line)

    page_number_text = str(pdf_page_number)

    if cleaned_lines and cleaned_lines[0] == page_number_text:
        removed_lines.append(cleaned_lines.pop(0))

    if cleaned_lines and cleaned_lines[-1] == page_number_text:
        removed_lines.append(cleaned_lines.pop())

    return "\n".join(cleaned_lines), removed_lines

def extract_document_pages(
    manifest_row: dict[str, Any],
    raw_data_dir: Path,
) -> list[dict[str, Any]]:
    """Extract and clean every indexed page for one document."""

    pdf_path = raw_data_dir / manifest_row["file_name"]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {pdf_path}"
        )

    start_page = int(manifest_row["indexed_start_page"])
    end_page = int(manifest_row["indexed_end_page"])
    extraction_method = manifest_row["extraction_method"]

    extracted_pages = []

    if extraction_method == "pdfplumber":
        with pdfplumber.open(pdf_path) as document:
            for page_number in range(
                start_page,
                end_page + 1,
            ):
                raw_text = (
                    document.pages[
                        page_number - 1
                    ].extract_text()
                    or ""
                )

                cleaned_text, removed_lines = (
                    clean_gazette_page_text(
                        raw_text,
                        page_number,
                    )
                )

                extracted_pages.append(
                    {
                        "page_number": page_number,
                        "text": cleaned_text,
                        "removed_lines": removed_lines,
                    }
                )

    elif extraction_method == "pymupdf":
        with fitz.open(pdf_path) as document:
            for page_number in range(
                start_page,
                end_page + 1,
            ):
                raw_text = document[
                    page_number - 1
                ].get_text("text")

                cleaned_text, removed_lines = (
                    clean_report_page_text(
                        raw_text,
                        page_number,
                    )
                )

                extracted_pages.append(
                    {
                        "page_number": page_number,
                        "text": cleaned_text,
                        "removed_lines": removed_lines,
                    }
                )

    else:
        raise ValueError(
            "Unsupported extraction method: "
            f"{extraction_method}"
        )

    return extracted_pages