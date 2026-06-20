"""Build the validated page-level dataset."""

from pathlib import Path

import pandas as pd

from aviation_rag.ingestion import extract_document_pages

import argparse


REQUIRED_DOCUMENT_COLUMNS = {
    "document_id",
    "file_name",
    "title",
    "document_type",
    "authority_level",
    "source_url",
    "extraction_method",
    "indexed_start_page",
    "indexed_end_page",
    "include_in_v1",
    "gazette_number",
    "publication_date",
    "effective_from",
    "effective_date_note",
    "reporting_period_start",
    "reporting_period_end",
}

REQUIRED_EXCLUSION_COLUMNS = {
    "document_id",
    "page_number",
    "exclusion_reason",
}


def load_document_manifest(
    manifest_path: Path,
) -> pd.DataFrame:
    """Load and validate the document manifest."""

    manifest = pd.read_csv(manifest_path)

    missing_columns = (
        REQUIRED_DOCUMENT_COLUMNS - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Document manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if manifest["document_id"].duplicated().any():
        raise ValueError(
            "Document manifest contains duplicate document IDs."
        )

    if manifest["file_name"].duplicated().any():
        raise ValueError(
            "Document manifest contains duplicate filenames."
        )

    return manifest


def load_page_exclusions(
    exclusions_path: Path,
) -> pd.DataFrame:
    """Load and validate page-level retrieval exclusions."""

    exclusions = pd.read_csv(exclusions_path)

    missing_columns = (
        REQUIRED_EXCLUSION_COLUMNS
        - set(exclusions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Page exclusions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if exclusions.duplicated(
        ["document_id", "page_number"]
    ).any():
        raise ValueError(
            "Page exclusions contain duplicate document-page pairs."
        )

    return exclusions

PAGE_METADATA_COLUMNS = [
    "document_id",
    "title",
    "document_type",
    "authority_level",
    "gazette_number",
    "publication_date",
    "effective_from",
    "effective_date_note",
    "reporting_period_start",
    "reporting_period_end",
    "source_url",
]


def clean_nullable_value(value):
    """Convert pandas missing values into normal Python None."""

    return None if pd.isna(value) else value


def build_page_records(
    manifest: pd.DataFrame,
    raw_data_dir: Path,
) -> pd.DataFrame:
    """Extract included documents into page-level records."""

    included_manifest = manifest[
        manifest["include_in_v1"]
    ].copy()

    page_records = []

    for manifest_row in included_manifest.to_dict(
        orient="records"
    ):
        extracted_pages = extract_document_pages(
            manifest_row,
            raw_data_dir,
        )

        document_metadata = {
            column: clean_nullable_value(
                manifest_row.get(column)
            )
            for column in PAGE_METADATA_COLUMNS
        }

        for page in extracted_pages:
            page_records.append(
                {
                    **document_metadata,
                    "page_number": page["page_number"],
                    "text": page["text"],
                    "character_count": len(page["text"]),
                    "removed_lines": page["removed_lines"],
                }
            )

    page_records_df = pd.DataFrame(page_records)

    expected_page_count = (
        included_manifest["indexed_end_page"]
        - included_manifest["indexed_start_page"]
        + 1
    ).sum()

    if len(page_records_df) != int(expected_page_count):
        raise ValueError(
            "Extracted page count does not match manifest ranges."
        )

    if page_records_df["text"].str.strip().eq("").any():
        raise ValueError(
            "Extracted page records contain empty text."
        )

    if page_records_df.duplicated(
        ["document_id", "page_number"]
    ).any():
        raise ValueError(
            "Extracted page records contain duplicate pages."
        )

    return page_records_df

def apply_page_exclusions(
    page_records: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach retrieval eligibility and exclusion reasons."""

    pages_with_policy = page_records.merge(
        exclusions,
        on=["document_id", "page_number"],
        how="left",
        validate="one_to_one",
    )

    matched_exclusion_count = (
        pages_with_policy["exclusion_reason"]
        .notna()
        .sum()
    )

    if matched_exclusion_count != len(exclusions):
        raise ValueError(
            "One or more page exclusions do not match "
            "an extracted document page."
        )

    pages_with_policy["retrieval_eligible"] = (
        pages_with_policy["exclusion_reason"].isna()
    )

    return pages_with_policy

def build_page_dataset(
    document_manifest_path: Path,
    exclusions_path: Path,
    raw_data_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build, validate, and save the page-level dataset."""

    manifest = load_document_manifest(
        document_manifest_path
    )

    exclusions = load_page_exclusions(
        exclusions_path
    )

    page_records = build_page_records(
        manifest,
        raw_data_dir,
    )

    page_records = apply_page_exclusions(
        page_records,
        exclusions,
    )

    if not page_records["retrieval_eligible"].any():
        raise ValueError(
            "Page dataset contains no retrieval-eligible pages."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_records.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    return page_records

def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the cleaned page-level aviation RAG dataset."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/raw and data/manifests.",
    )

    return parser.parse_args()


def main():
    """Run the page-dataset build pipeline."""

    arguments = parse_arguments()
    project_root = arguments.project_root.resolve()

    page_records = build_page_dataset(
        document_manifest_path=(
            project_root
            / "data"
            / "manifests"
            / "documents.csv"
        ),
        exclusions_path=(
            project_root
            / "data"
            / "manifests"
            / "page_exclusions.csv"
        ),
        raw_data_dir=project_root / "data" / "raw",
        output_path=(
            project_root
            / "data"
            / "processed"
            / "pages.jsonl"
        ),
    )

    eligible_count = int(
        page_records["retrieval_eligible"].sum()
    )

    excluded_count = len(page_records) - eligible_count

    print("Page dataset built successfully.")
    print(f"Total pages: {len(page_records)}")
    print(f"Retrieval-eligible pages: {eligible_count}")
    print(f"Excluded pages: {excluded_count}")


if __name__ == "__main__":
    main()