import json
from pathlib import Path
import copy
import pytest

from app.services.chunking.clinical_chunker import chunk_extraction

from transformers import AutoModel, AutoTokenizer

from app.services.chunking.clinical_chunker import (
    chunk_extraction,
    MissingPatientIdError,
    ConflictingPatientIdError,
)


def test_real_extracted_report_chunks_successfully():
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "clinical_extraction_c0141.json"
    )

    assert fixture_path.is_file(), (
        f"Clinical extraction fixture not found: {fixture_path}"
    )

    with fixture_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(payload, dict), "Fixture must contain a JSON object"

    # Patient identity belongs to the ingestion context.
    # Use a test-only ID if the sample API response omits it.
    test_patient_id = 101
    payload["patient_id"] = test_patient_id

    # Ensure a nested extraction object cannot override the test identity
    # with a missing or null patient ID.
    def set_existing_patient_ids(value):
        if isinstance(value, dict):
            if "patient_id" in value:
                value["patient_id"] = test_patient_id

            for nested_value in value.values():
                set_existing_patient_ids(nested_value)

        elif isinstance(value, list):
            for item in value:
                set_existing_patient_ids(item)

    set_existing_patient_ids(payload)

    result = chunk_extraction(payload)

    # The real report must produce meaningful output.
    assert result.parent is not None
    assert result.parent.text.strip(), "Parent chunk must contain text"
    assert result.children, "Report must produce clinical child chunks"

    # Every chunk must remain scoped to the same patient.
    for chunk in [result.parent, *result.children]:
        assert chunk.metadata.get("patient_id") == test_patient_id

    # Child chunks must link back to their parent.
    for chunk in result.children:
        assert chunk.metadata.get("parent_id") == result.parent.chunk_id

    # The output must contain usable clinical content.
    assert any(chunk.text.strip() for chunk in result.children)


def test_real_report_preserves_patient_identity_and_rejects_unsafe_scope():
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "clinical_extraction_c0141.json"
    )

    assert fixture_path.is_file(), f"Fixture not found: {fixture_path}"

    with fixture_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(payload, dict), "Expected a JSON object"

    # Do not inject or overwrite patient_id: test the real fixture as-is.
    result = chunk_extraction(payload)

    # This fixture corresponds to the API request using patient_id=4.
    expected_patient_id = 4

    assert result.parent is not None
    assert result.parent.text.strip()
    assert result.children, "The report must generate clinical chunks"

    for chunk in [result.parent, *result.children]:
        assert str(chunk.metadata.get("patient_id")) == str(
            expected_patient_id
        ), f"Incorrect patient scope on chunk {chunk.chunk_id}"

    for child in result.children:
        assert child.metadata.get("parent_id") == result.parent.chunk_id

    # Removing all patient IDs must make the chunker reject the payload.
    missing_payload = copy.deepcopy(payload)

    def remove_patient_ids(value):
        if isinstance(value, dict):
            value.pop("patient_id", None)
            for nested in value.values():
                remove_patient_ids(nested)
        elif isinstance(value, list):
            for item in value:
                remove_patient_ids(item)

    remove_patient_ids(missing_payload)

    with pytest.raises(MissingPatientIdError):
        chunk_extraction(missing_payload)

    # A conflicting wrapper ID must not override the fixture's identity.
    conflicting_payload = copy.deepcopy(payload)
    conflicting_payload["patient_id"] = 999

    with pytest.raises(ConflictingPatientIdError):
        chunk_extraction(conflicting_payload)
