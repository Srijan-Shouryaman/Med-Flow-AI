"""
Quality-gate tests for MedFlow AI's clinical chunker.

Expected project layout:
    backend/app/services/chunking/clinical_chunker.py
    backend/tests/test_clinical_chunker_quality.py

Run from the backend/ directory:
    python -m pytest -q tests/test_clinical_chunker_quality.py

These tests intentionally check clinical meaning and patient isolation, not just
whether the function returns a list. A failing test identifies a quality gap;
do not weaken a test merely to make the suite green.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest

from app.services.chunking.clinical_chunker import chunk_extraction


# ---------------------------------------------------------------------------
# Small, deterministic fixture matching the extraction contract.
# Keep this fixture synthetic; never use real patient information in tests.
# ---------------------------------------------------------------------------

REPORT_TEXT = (
    "The patient reports heartburn for approximately three weeks. "
    "The patient denies vomiting blood and black stools. "
    "Allergy status is not confirmed. "
    "Blood pressure is 126/80 mmHg. ALT is 28 U/L and AST is 25 U/L. "
    "No endoscopy result is supplied. "
    "The patient's mother had diabetes."
)


def _span(text: str, role: str = "finding") -> list[dict[str, Any]]:
    """Build evidence spans whose offsets point to an exact source substring."""
    start = REPORT_TEXT.index(text)
    return [
        {
            "role": role,
            "span": [start, start + len(text)],
            "text": text,
            "sentence_span": [start, start + len(text)],
            "sentence_index": 0,
        }
    ]


def _fact(
    name: str,
    fact_id: str | None,
    assertion: str,
    evidence: str,
    *,
    entity_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "fact_id": fact_id,
        "entity_id": entity_id or f"entity-{fact_id or name}",
        "assertion": assertion,
        "evidence": evidence,
        "evidence_spans": _span(
            evidence,
            "data_gap" if assertion in {"UNKNOWN", "NOT_PROVIDED"} else "finding",
        ),
        "negated": assertion in {"ABSENT", "NOT_PROVIDED"},
        **extra,
    }


def make_payload(
    *,
    patient_id: int | None = 104,
    case_id: str = "C-TEST-104",
    report_id: int = 9001,
) -> dict[str, Any]:
    """Create a representative wrapper and extraction payload."""
    extraction_json = {
        "common": {
            "symptoms": [
                _fact(
                    "heartburn",
                    "fact-heartburn",
                    "PRESENT",
                    "The patient reports heartburn for approximately three weeks.",
                    duration="approximately three weeks",
                    status="MENTIONED",
                ),
                _fact(
                    "vomiting blood",
                    "fact-vomiting-blood",
                    "ABSENT",
                    "The patient denies vomiting blood and black stools.",
                    status="NEGATED",
                ),
                _fact(
                    "black stools",
                    "fact-black-stools",
                    "ABSENT",
                    "The patient denies vomiting blood and black stools.",
                    status="NEGATED",
                ),
            ],
            "allergies": [
                {
                    **_fact(
                        "unspecified allergy status",
                        None,
                        "UNKNOWN",
                        "Allergy status is not confirmed.",
                        entity_id="entity-allergy-unknown",
                    ),
                    "status": "UNCERTAIN",
                    "substance": "unspecified",
                    "evidence_role": "data_gap",
                }
            ],
            "diagnoses": [],
            "procedures": [],
            "medications": [],
            "vital_signs": [
                {
                    **_fact(
                        "blood pressure",
                        "fact-bp",
                        "PRESENT",
                        "Blood pressure is 126/80 mmHg.",
                    ),
                    "measurement": "blood pressure",
                    "value": "126/80",
                    "unit": "mmHg",
                }
            ],
            "investigations": [
                {
                    **_fact("ALT", "fact-alt", "PRESENT", "ALT is 28 U/L and AST is 25 U/L."),
                    "name": "ALT",
                    "result": "28",
                    "unit": "U/L",
                },
                {
                    **_fact("AST", "fact-ast", "PRESENT", "ALT is 28 U/L and AST is 25 U/L."),
                    "name": "AST",
                    "result": "25",
                    "unit": "U/L",
                },
            ],
            "other_findings": [],
            "clinical_history": [
                _fact(
                    "diabetes",
                    "fact-family-diabetes",
                    "PRESENT",
                    "The patient's mother had diabetes.",
                    relation="mother",
                    experiencer="family_member",
                    category="family_history",
                )
            ],
        },
        "patient_id": patient_id,
        "department_specific": {
            "gastroenterology": {
                "findings": [],
                "symptoms": [],
                "diagnoses": [],
                "procedures": [
                    {
                        **_fact(
                            "endoscopy",
                            "fact-endoscopy-gap",
                            "NOT_PROVIDED",
                            "No endoscopy result is supplied.",
                            status="NEGATED",
                            evidence_role="data_gap",
                        ),
                        "category": "procedures",
                    }
                ],
                "investigations": [],
            },
            "radiology": {"findings": [], "procedures": [], "investigations": []},
            "orthopedic": {
                "findings": [],
                "symptoms": [],
                "diagnoses": [],
                "procedures": [],
                "investigations": [],
            },
            "cardiovascular_pulmonary": {
                "findings": [],
                "symptoms": [],
                "diagnoses": [],
                "procedures": [],
                "investigations": [],
            },
            "consult_history_physical": {
                "history": [],
                "findings": [],
                "symptoms": [],
                "diagnoses": [],
                "medications": [],
            },
        },
    }
    return {
        "message": "Report processed successfully",
        "patient_id": patient_id,
        "report_id": report_id,
        "case_id": case_id,
        "prediction": {
            "predicted_department": "Gastroenterology",
            "confidence_score": 0.46,
        },
        "clinical_extraction": {
            "report_id": report_id,
            "case_id": case_id,
            "extraction_json": extraction_json,
        },
    }


def _chunks(payload: dict[str, Any], full_text: str = REPORT_TEXT):
    collection = chunk_extraction(payload, full_text=full_text)
    assert hasattr(collection, "parent"), "chunk_extraction must return a collection with parent"
    assert hasattr(collection, "children"), "collection must expose child chunks"
    return collection


def _all_chunks(collection):
    return [collection.parent, *collection.children]


def _by_type(collection, chunk_type: str):
    return [c for c in collection.children if c.chunk_type == chunk_type]


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# ---------------------------------------------------------------------------
# Contract and patient/case/report isolation
# ---------------------------------------------------------------------------

def test_returns_parent_and_child_collection():
    collection = _chunks(make_payload())
    assert collection.parent.chunk_type == "parent_report"
    assert isinstance(collection.children, list)
    assert collection.children, "Expected at least one clinical child chunk"


def test_every_chunk_is_scoped_to_same_patient_case_and_report():
    payload = make_payload(patient_id=104, case_id="C-TEST-104", report_id=9001)
    collection = _chunks(payload)

    for chunk in _all_chunks(collection):
        assert chunk.metadata.get("patient_id") == 104
        assert chunk.metadata.get("case_id") == "C-TEST-104"
        assert chunk.metadata.get("report_id") == 9001


def test_every_child_points_to_the_parent_chunk_id():
    collection = _chunks(make_payload())
    for child in collection.children:
        assert child.metadata.get("parent_id") == collection.parent.chunk_id
        assert child.metadata.get("is_parent") is False


def test_parent_has_no_parent_and_is_marked_as_parent():
    collection = _chunks(make_payload())
    assert collection.parent.metadata.get("parent_id") is None
    assert collection.parent.metadata.get("is_parent") is True


def test_chunk_ids_are_unique_including_parent():
    collection = _chunks(make_payload())
    ids = [chunk.chunk_id for chunk in _all_chunks(collection)]
    duplicates = [chunk_id for chunk_id, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"Duplicate chunk IDs: {duplicates}"


def test_same_input_produces_stable_chunk_ids():
    first = _chunks(make_payload())
    second = _chunks(make_payload())
    assert [c.chunk_id for c in _all_chunks(first)] == [
        c.chunk_id for c in _all_chunks(second)
    ]


def test_patient_id_changes_chunk_id_namespace():
    first = _chunks(make_payload(patient_id=104))
    second = _chunks(make_payload(patient_id=205, case_id="C-TEST-205"))
    first_ids = {c.chunk_id for c in _all_chunks(first)}
    second_ids = {c.chunk_id for c in _all_chunks(second)}
    assert first_ids.isdisjoint(second_ids), "Different patients must not share chunk IDs"


def test_missing_patient_id_is_rejected_before_chunks_can_be_indexed():
    """
    Security gate: the current implementation may only log a warning here.
    If so, this test should fail until ingestion rejects unscoped patient data.
    """
    with pytest.raises((ValueError, TypeError)):
        chunk_extraction(make_payload(patient_id=None), full_text=REPORT_TEXT)


# ---------------------------------------------------------------------------
# Clinical assertion correctness and uncertainty preservation
# ---------------------------------------------------------------------------

def test_present_heartburn_is_in_positive_symptoms_not_negatives():
    collection = _chunks(make_payload())
    positive_text = " ".join(c.text.casefold() for c in _by_type(collection, "symptoms_present"))
    negative_text = " ".join(c.text.casefold() for c in _by_type(collection, "pertinent_negatives"))

    assert "heartburn" in positive_text
    assert "heartburn" not in negative_text


def test_negated_symptoms_are_not_promoted_to_present_symptoms():
    collection = _chunks(make_payload())
    positive_text = " ".join(c.text.casefold() for c in _by_type(collection, "symptoms_present"))
    negative_text = " ".join(c.text.casefold() for c in _by_type(collection, "pertinent_negatives"))

    for symptom in ("vomiting blood", "black stools"):
        assert symptom not in positive_text, f"Negated symptom leaked into positives: {symptom}"
        assert symptom in negative_text, f"Negated symptom missing from negatives: {symptom}"



def test_unknown_allergy_status_is_preserved_and_never_claimed_as_no_allergies():
    collection = _chunks(make_payload())

    allergy_chunks = _by_type(collection, "allergy_status")

    assert allergy_chunks, (
        "An allergy-status chunk must exist even with no fact_id"
    )

    allergy_text = " ".join(
        chunk.text.casefold() for chunk in allergy_chunks
    )

    # Unknown status must remain explicit.
    assert any(
        term in allergy_text
        for term in (
            "unknown",
            "uncertain",
            "not confirmed",
            "not documented",
        )
    ), "Unknown allergy status must be preserved"

    # Reject statements that assert the patient has no allergies.
    # Do not reject a disclaimer explaining that unknown status
    # does not establish the absence of allergies.
    assert "nkda is documented" not in allergy_text
    assert "no known drug allergies" not in allergy_text
    assert "no known allergies" not in allergy_text

    # Preserve the safety disclaimer if the chunker emits it.
    assert (
        "does not mean the patient has no allergies"
        in allergy_text
    ), "The safety disclaimer must be preserved"



def test_data_gap_does_not_claim_procedure_was_not_performed():
    collection = _chunks(make_payload())
    gap_chunks = _by_type(collection, "data_gaps")
    assert gap_chunks, "Missing endoscopy information should produce a data-gap chunk"

    gap_text = " ".join(c.text.casefold() for c in gap_chunks)
    assert "endoscopy" in gap_text
    assert any(
        phrase in gap_text
        for phrase in ("not supplied", "not provided", "not documented", "result is missing", "unknown")
    )
    assert not re.search(r"\bendoscopy (?:was )?not performed\b", gap_text)


def test_lab_results_are_in_investigation_chunks_not_vitals():
    collection = _chunks(make_payload())
    investigation_text = " ".join(c.text.casefold() for c in _by_type(collection, "investigations"))
    vital_text = " ".join(c.text.casefold() for c in _by_type(collection, "vitals"))

    assert "alt" in investigation_text and "28" in investigation_text
    assert "ast" in investigation_text and "25" in investigation_text
    assert "alt" not in vital_text and "ast" not in vital_text


def test_family_history_is_not_misrepresented_as_patient_diagnosis():
    collection = _chunks(make_payload())
    diagnosis_text = " ".join(c.text.casefold() for c in _by_type(collection, "diagnoses"))
    family_text = " ".join(
        c.text.casefold()
        for c in collection.children
        if c.chunk_type == "family_history"
    )

    assert "diabetes" not in diagnosis_text
    assert "diabetes" in family_text


def test_predicted_department_is_metadata_not_clinical_assertion_text():
    collection = _chunks(make_payload())
    for child in collection.children:
        assert "predicted department" not in child.text.casefold()
        assert "gastroenterology prediction" not in child.text.casefold()


# ---------------------------------------------------------------------------
# Fact traceability, source grounding, duplication, and size distribution
# ---------------------------------------------------------------------------

def test_fact_ids_are_unique_within_each_child_metadata():
    collection = _chunks(make_payload())
    for child in collection.children:
        fact_ids = child.metadata.get("fact_ids", [])
        assert len(fact_ids) == len(set(fact_ids)), (
            f"{child.chunk_type} contains duplicate fact IDs: {fact_ids}"
        )


def test_evidence_spans_point_to_exact_substrings_of_original_report():
    collection = _chunks(make_payload(), full_text=REPORT_TEXT)
    for chunk in collection.children:
        for evidence in chunk.metadata.get("evidence_spans", []):
            span = evidence.get("span")
            text = evidence.get("text")
            if span is None or text is None:
                continue
            assert isinstance(span, (list, tuple)) and len(span) == 2
            start, end = span
            assert isinstance(start, int) and isinstance(end, int)
            assert 0 <= start <= end <= len(REPORT_TEXT)
            assert REPORT_TEXT[start:end] == text, (
                f"Evidence span is not grounded in source text for {chunk.chunk_type}: {text!r}"
            )


def test_child_text_is_nonempty_and_not_a_duplicate_of_parent_text():
    collection = _chunks(make_payload())
    parent_text = _normalized(collection.parent.text)
    for child in collection.children:
        assert child.text.strip(), f"Empty child text: {child.chunk_id}"
        assert _normalized(child.text) != parent_text


def test_no_duplicate_normalized_child_text_without_an_explicit_exception():
    """
    This intentionally reports exact duplicate text even when IDs differ.
    If duplicates are semantically intentional, document a narrow exception
    rather than globally suppressing duplicates.
    """
    collection = _chunks(make_payload())
    by_text: dict[str, list[str]] = {}
    for child in collection.children:
        by_text.setdefault(_normalized(child.text), []).append(child.chunk_id)

    duplicates = {
        text: ids for text, ids in by_text.items()
        if text and len(ids) > 1
    }
    assert not duplicates, f"Duplicate normalized child text: {duplicates}"


def test_child_chunk_size_is_within_configured_character_safety_bounds():
    """
    Temporary deterministic proxy until the project's embedding tokenizer is
    wired into the quality gate. Character counts are not token counts.
    This catches empty and grossly oversized chunks without pretending to be
    a tokenizer-accurate 20–300 token check.
    """
    collection = _chunks(make_payload())
    for child in collection.children:
        assert len(child.text.strip()) > 0
        assert len(child.text) <= 4000, (
            f"Oversized child chunk ({len(child.text)} chars): {child.chunk_type}"
        )


def test_all_chunks_mark_human_review_required():
    collection = _chunks(make_payload())
    for chunk in _all_chunks(collection):
        assert chunk.metadata.get("requires_human_review") is True


@pytest.mark.parametrize(
    ("patient_id", "case_id", "report_id"),
    [
        (1, "CASE-A", 11),
        (2, "CASE-B", 22),
        ("patient-x", "CASE-X", "REPORT-X"),
    ],
)
def test_identity_metadata_is_preserved_for_supported_identifier_types(
    patient_id, case_id, report_id
):
    collection = _chunks(
        make_payload(patient_id=patient_id, case_id=case_id, report_id=report_id)
    )
    for chunk in _all_chunks(collection):
        assert chunk.metadata.get("patient_id") == patient_id
        assert chunk.metadata.get("case_id") == case_id
        assert chunk.metadata.get("report_id") == report_id


def test_original_report_text_is_not_falsely_claimed_when_not_supplied():
    collection = chunk_extraction(make_payload(), full_text=None)
    assert collection.parent.metadata.get("original_text_supplied") is False
    assert "original report text was not supplied" in collection.parent.text.casefold()


def test_parent_preserves_exact_original_report_text_when_supplied():
    collection = _chunks(make_payload(), full_text=REPORT_TEXT)
    assert collection.parent.text == REPORT_TEXT
    assert collection.parent.metadata.get("original_text_supplied") is True
