
"""
Inspect Clinical Chunks
-----------------------
Loads a real clinical extraction API response from a JSON fixture,
runs the existing clinical chunker, and prints the generated chunks.

This script does not modify the chunker, database, or source payload.
"""

import json
import logging
import sys
from pathlib import Path
from pprint import pprint

from app.services.chunking.clinical_chunker import chunk_extraction


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FIXTURE_PATH = (
    BASE_DIR
    / "tests"
    / "fixtures"
    / "clinical_extraction_c0141.json"
)

EXPECTED_PATIENT_ID = 4
EXPECTED_REPORT_ID = 52
EXPECTED_CASE_ID = "C0144"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Load the clinical extraction payload
# ---------------------------------------------------------

def load_payload() -> dict:
    """Load and validate the saved clinical extraction response."""

    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError(
            f"Clinical extraction fixture not found: {FIXTURE_PATH}\n"
            "Check that tests/fixtures/clinical_extraction_c0143.json "
            "exists in your backend directory."
        )

    try:
        with FIXTURE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in fixture {FIXTURE_PATH}, "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("The extraction payload must be a JSON object.")

    for field in ("patient_id", "report_id", "case_id"):
        if field not in payload:
            raise ValueError(
                f"Required field '{field}' is missing from the payload."
            )

    if not isinstance(payload.get("clinical_extraction"), dict):
        raise ValueError(
            "The payload must contain a clinical_extraction object."
        )

    if str(payload["patient_id"]) != str(EXPECTED_PATIENT_ID):
        raise ValueError(
            f"Unexpected patient_id: {payload['patient_id']}"
        )

    if str(payload["report_id"]) != str(EXPECTED_REPORT_ID):
        raise ValueError(
            f"Unexpected report_id: {payload['report_id']}"
        )

    if str(payload["case_id"]) != EXPECTED_CASE_ID:
        raise ValueError(
            f"Unexpected case_id: {payload['case_id']}"
        )

    return payload


# ---------------------------------------------------------
# Inspect one chunk
# ---------------------------------------------------------

def print_chunk(index: int, chunk, chunk_type: str) -> None:
    """Print the actual content and metadata of a chunk."""

    text = chunk.text
    metadata = chunk.metadata

    print("\n" + "=" * 75)
    print(f"{chunk_type} CHUNK #{index}")
    print("=" * 75)

    print(f"Chunk ID   : {chunk.chunk_id}")
    print(f"Text length: {len(text)} characters")

    print("\n--- CHUNK TEXT ---")
    print(text if text.strip() else "[EMPTY TEXT]")

    print("\n--- CHUNK METADATA ---")
    pprint(metadata, sort_dicts=False)

    if not text.strip():
        logger.warning(
            "Chunk has empty text",
            extra={"chunk_id": str(chunk.chunk_id)},
        )


# ---------------------------------------------------------
# Main inspection pipeline
# ---------------------------------------------------------

def main() -> int:
    """Run the chunker and inspect its output."""

    try:
        print("\nLoading clinical extraction payload...")
        payload = load_payload()

        print("\n--- SOURCE REPORT ---")
        print(f"Patient ID : {payload['patient_id']}")
        print(f"Report ID  : {payload['report_id']}")
        print(f"Case ID    : {payload['case_id']}")

        print("\nRunning clinical chunker...")

        result = chunk_extraction(payload)

        if result.parent is None:
            raise ValueError("The chunker returned no parent chunk.")

        if not result.parent.text.strip():
            raise ValueError("The parent chunk contains empty text.")

        if not result.children:
            logger.warning("The chunker returned no child chunks.")

        all_chunks = [result.parent, *result.children]

        print("\n" + "#" * 75)
        print("CHUNKING SUMMARY")
        print("#" * 75)

        print(f"Parent chunk ID : {result.parent.chunk_id}")
        print(f"Child chunk count: {len(result.children)}")
        print(f"Total chunks     : {len(all_chunks)}")
        print(
            "Total characters : "
            f"{sum(len(chunk.text) for chunk in all_chunks)}"
        )

        # Display the parent chunk.
        print_chunk(
            index=1,
            chunk=result.parent,
            chunk_type="PARENT",
        )

        # Display every child chunk.
        for index, child in enumerate(result.children, start=1):
            print_chunk(
                index=index,
                chunk=child,
                chunk_type="CHILD",
            )

        # -------------------------------------------------
        # Basic structural checks
        # -------------------------------------------------

        print("\n" + "#" * 75)
        print("STRUCTURAL VALIDATION")
        print("#" * 75)

        chunk_ids = [str(chunk.chunk_id) for chunk in all_chunks]

        unique_ids = len(chunk_ids) == len(set(chunk_ids))

        print(f"Unique chunk IDs: {unique_ids}")

        if not unique_ids:
            logger.error("Duplicate chunk IDs detected.")

        parent_id = str(result.parent.chunk_id)

        for index, child in enumerate(result.children, start=1):
            metadata = child.metadata

            child_parent_id = metadata.get("parent_id")

            parent_matches = (
                str(child_parent_id) == parent_id
                if child_parent_id is not None
                else False
            )

            patient_matches = (
                str(metadata.get("patient_id"))
                == str(payload["patient_id"])
            )

            print(
                f"Child {index}: "
                f"parent_id_matches={parent_matches}, "
                f"patient_id_matches={patient_matches}"
            )

        print("\nInspection completed.")
        print(
            "Review the printed text and metadata before integrating "
            "the chunks with the embedding service."
        )

        return 0

    except FileNotFoundError:
        logger.exception("Required fixture file was not found.")
        return 1

    except (ValueError, TypeError):
        logger.exception("Payload validation or chunk inspection failed.")
        return 1

    except Exception:
        logger.exception(
            "Unexpected error while inspecting clinical chunks."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
