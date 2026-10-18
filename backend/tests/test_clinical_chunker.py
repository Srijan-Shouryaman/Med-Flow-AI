
"""Golden and regression tests for clinical_chunker.py.

Run from the backend root:
    pytest tests/test_clinical_chunker.py -v

Tests the chunker only. No database or API required.
"""

import pytest

from app.services.chunking.clinical_chunker import chunk_extraction


GAP = (
    "No endoscopy, abdominal imaging, or H. pylori test "
    "result is supplied."
)

NEG = (
    "No vomiting blood, black stools, persistent vomiting, "
    "or documented unintentional weight loss is reported "
    "in this synthetic history."
)

HPI = (
    "Intermittent upper abdominal discomfort for six weeks, "
    "post-meal bloating for one month, and occasional heartburn "
    "for approximately three weeks."
)

VIT = (
    "Blood pressure: 126/80 mmHg; heart rate: 78 beats/min; "
    "respiratory rate: 16 breaths/min; temperature: 36.8 °C; "
    "oxygen saturation: 99% on room air."
)


def _sym(name, status, negated, evidence, duration=None):
    return {
        "name": name,
        "status": status,
        "negated": negated,
        "duration": duration,
        "evidence": evidence,
        "entity_id": f"e-{name}",
        "experiencer": None,
        "severity": None,
    }


def _vital(measurement, value, unit):
    return {
        "measurement": measurement,
        "value": value,
        "unit": unit,
        "evidence": VIT,
        "entity_id": f"v-{measurement}",
    }


def by_type(collection):
    """Group child chunks by chunk_type."""
    grouped = {}

    for chunk in collection.children:
        grouped.setdefault(chunk.chunk_type, []).append(chunk)

    return grouped


def all_child_text(collection):
    return " ".join(chunk.text for chunk in collection.children)


# Legacy record shaped like the synthetic C0138 sample.
# It intentionally has no assertion metadata at the top level.
LEGACY = {
    "report_id": 46,
    "case_id": "C0138",
    "prediction": {
        "predicted_department": "Gastroenterology",
        "confidence_score": 0.466,
        "confidence_level": "Low",
    },
    "clinical_extraction": {
        "extraction_json": {
            "common": {
                "symptoms": [
                    _sym(
                        "upper abdominal discomfort",
                        "MENTIONED",
                        False,
                        HPI,
                        "six weeks",
                    ),
                    _sym(
                        "bloating",
                        "MENTIONED",
                        False,
                        HPI,
                        "one month",
                    ),
                    _sym(
                        "heartburn",
                        "MENTIONED",
                        False,
                        HPI,
                        "approximately three weeks",
                    ),
                    _sym(
                        "vomiting blood",
                        "NEGATED",
                        True,
                        NEG,
                    ),
                    _sym(
                        "black stools",
                        "NEGATED",
                        True,
                        NEG,
                    ),
                    _sym(
                        "unintentional weight loss",
                        "NEGATED",
                        True,
                        NEG,
                    ),
                ],
                "allergies": [
                    {
                        "status": "UNCERTAIN",
                        "substance": "unspecified",
                        "evidence": (
                            "No complete prescription medication list "
                            "or confirmed allergy status is available."
                        ),
                        "entity_id": "a1",
                    }
                ],
                "diagnoses": [],
                "medications": [],
                "clinical_history": [],
                "procedures": [
                    {
                        "name": "endoscopy",
                        "status": "NEGATED",
                        "evidence": GAP,
                        "entity_id": "p1",
                    }
                ],
                "vital_signs": [
                    _vital("blood pressure", "126/80", "mmHg"),
                    _vital("heart rate", "78", "bpm"),
                    _vital("oxygen saturation", "99", "%"),
                ],
                "investigations": [
                    {
                        "name": "hemoglobin",
                        "unit": "g/dL",
                        "result": "13.8",
                        "evidence": "Hemoglobin: 13.8 g/dL;",
                        "entity_id": "i1",
                    },
                    {
                        "name": "oxygen saturation",
                        "unit": None,
                        "result": None,
                        "evidence": VIT,
                        "entity_id": "i2",
                    },
                    {
                        "name": "h. pylori",
                        "unit": None,
                        "result": None,
                        "evidence": GAP,
                        "entity_id": "i3",
                    },
                ],
            },
            "department_specific": {
                "gastroenterology": {
                    "findings": [
                        {
                            "name": "guarding",
                            "status": "NEGATED",
                            "negated": True,
                            "assertion": "ABSENT",
                            "evidence": (
                                "No guarding or rebound tenderness "
                                "is recorded."
                            ),
                            "evidence_role": "finding",
                            "fact_id": "f_g",
                            "sentence_index": 43,
                            "span": [1662, 1670],
                        }
                    ],
                    "procedures": [
                        {
                            "name": "endoscopy",
                            "status": "NEGATED",
                            "assertion": "NOT_PROVIDED",
                            "evidence": GAP,
                            "evidence_role": "data_gap",
                            "fact_id": "f_e",
                            "sentence_index": 48,
                            "span": [1945, 1954],
                        }
                    ],
                    "investigations": [
                        {
                            "name": "h. pylori",
                            "status": "NEGATED",
                            "assertion": "NOT_PROVIDED",
                            "evidence": GAP,
                            "evidence_role": "data_gap",
                            "fact_id": "f_h",
                            "sentence_index": 48,
                            "span": [1978, 1987],
                        }
                    ],
                }
            },
        }
    },
}


# Newer extraction record containing family history and allergy metadata.
NEW = {
    "case_id": "C0200",
    "report_id": 7,
    "extraction_json": {
        "common": {
            "symptoms": [
                {
                    "name": "fever",
                    "assertion": "PRESENT",
                    "status": "MENTIONED",
                    "evidence": "Mother reports the child has fever.",
                    "fact_id": "f1",
                }
            ],
            "diagnoses": [
                {
                    "name": "type 2 diabetes",
                    "assertion": "PRESENT",
                    "status": "MENTIONED",
                    "evidence": "Patient has type 2 diabetes.",
                    "fact_id": "f2",
                }
            ],
            "allergies": [
                {
                    "assertion": "NKDA",
                    "substance": "unspecified",
                    "status": "NEGATED",
                    "evidence": "Allergies: NKDA",
                }
            ],
            "clinical_history": [
                {
                    "history_type": "family",
                    "relation": "mother",
                    "assertion": "FAMILY",
                    "description": "Mother had hypertension.",
                    "evidence": "Mother had hypertension.",
                    "fact_id": "f3",
                    "evidence_spans": [
                        {
                            "sentence_index": 3,
                            "span": [90, 102],
                        }
                    ],
                }
            ],
        },
        "department_specific": {},
        "metadata": {
            "extraction_schema_version": "2.1",
            "data_gaps": [],
            "allergy_status": {
                "status": "NONE_KNOWN",
            },
        },
    },
}


# ---------------------------------------------------------
# 1. Legacy symptoms and pertinent negatives
# ---------------------------------------------------------

def test_legacy_negatives_never_appear_in_positive_symptoms():
    chunks = by_type(chunk_extraction(LEGACY))

    positive_text = chunks["symptoms_present"][0].text
    negative_text = chunks["pertinent_negatives"][0].text

    assert "vomiting blood" not in positive_text
    assert "black stools" not in positive_text
    assert "vomiting blood" in negative_text
    assert "guarding" in negative_text

    assert (
        chunks["pertinent_negatives"][0]
        .metadata["retrieval_polarity"]
        == "negated"
    )

    assert "for six weeks" in positive_text


# ---------------------------------------------------------
# 2. Allergy uncertainty must not become no allergies
# ---------------------------------------------------------

def test_legacy_allergy_unknown_not_none():
    chunks = by_type(chunk_extraction(LEGACY))
    allergy = chunks["allergy_status"][0]

    assert "UNKNOWN" in allergy.text
    assert "does NOT mean" in allergy.text

    # The placeholder must not become a clinical allergy substance.
    assert (
        "unspecified" not in allergy.text
        .replace("unspecified allergy", "")
        .lower()
    )


def test_allergy_status_from_metadata_and_always_present():
    chunks = by_type(chunk_extraction(NEW))

    allergy = chunks["allergy_status"][0]

    assert "NKDA" in allergy.text
    assert allergy.metadata["allergy_status"] == "NONE_KNOWN"

    empty = {
        "case_id": "C1",
        "report_id": 1,
        "extraction_json": {
            "common": {},
            "department_specific": {},
        },
    }

    empty_chunks = by_type(chunk_extraction(empty))
    empty_allergy = empty_chunks["allergy_status"][0]

    assert "NOT DOCUMENTED" in empty_allergy.text
    assert "does NOT mean" in empty_allergy.text


# ---------------------------------------------------------
# 3. Data gaps must not become confirmed negative procedures
# ---------------------------------------------------------

def test_legacy_data_gap_not_a_negated_procedure():
    collection = chunk_extraction(LEGACY)
    chunks = by_type(collection)

    assert "data_gaps" in chunks
    assert "endoscopy" in chunks["data_gaps"][0].text

    if "procedures_tests" in chunks:
        assert all(
            "endoscopy" not in chunk.text.lower()
            for chunk in chunks["procedures_tests"]
        )

    assert all(
        "endoscopy" not in chunk.text.lower()
        for chunk in chunks["pertinent_negatives"]
    )

    assert all(
        "h. pylori" not in chunk.text.lower()
        for chunk in chunks["pertinent_negatives"]
    )

    assert "not performed" in chunks["data_gaps"][0].text.lower()


# ---------------------------------------------------------
# 4. Avoid duplicate, valueless oxygen saturation
# ---------------------------------------------------------

def test_oxygen_saturation_only_as_vital():
    chunks = by_type(chunk_extraction(LEGACY))

    assert "vitals" in chunks
    assert "Oxygen saturation: 99 %" in chunks["vitals"][0].text

    # No result means the duplicate investigation should be dropped.
    if "investigations" in chunks:
        investigation_text = " ".join(
            chunk.text.lower()
            for chunk in chunks["investigations"]
        )

        assert "oxygen saturation" not in investigation_text


# ---------------------------------------------------------
# 5. Predictions belong in metadata, not clinical text
# ---------------------------------------------------------

def test_prediction_is_metadata_only():
    collection = chunk_extraction(LEGACY, "full text")

    assert all(
        "Gastroenterology" not in chunk.text
        for chunk in collection.children
    )

    assert (
        collection.children[0].metadata["predicted_department"]
        == "Gastroenterology"
    )

    assert collection.parent.metadata["is_parent"] is True

    assert (
        collection.children[0].metadata["parent_id"]
        == collection.parent.chunk_id
    )


# ---------------------------------------------------------
# 6. Family history must be isolated
# ---------------------------------------------------------

def test_family_history_isolated():
    chunks = by_type(chunk_extraction(NEW))

    assert "family_history" in chunks
    assert "diagnoses" in chunks

    family_text = chunks["family_history"][0].text

    assert "mother: Mother had hypertension." in family_text

    diagnosis_text = " ".join(
        chunk.text.lower()
        for chunk in chunks["diagnoses"]
    )

    assert "hypertension" not in diagnosis_text

    other_text = " ".join(
        chunk.text.lower()
        for chunk in chunk_extraction(NEW).children
        if chunk.chunk_type != "family_history"
    )

    assert "hypertension" not in other_text

    assert (
        chunks["family_history"][0]
        .metadata["retrieval_polarity"]
        == "family"
    )


# ---------------------------------------------------------
# 7. Deterministic IDs and unique chunks
# ---------------------------------------------------------

def test_ids_stable_and_unique():
    first = chunk_extraction(LEGACY)
    second = chunk_extraction(LEGACY)

    first_ids = [
        chunk.chunk_id for chunk in first.children
    ]

    second_ids = [
        chunk.chunk_id for chunk in second.children
    ]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_chunk_id_changes_when_text_changes():
    original = chunk_extraction(LEGACY)

    changed = chunk_extraction(
        LEGACY,
        full_text="A different source report.",
    )

    assert original.parent.chunk_id != changed.parent.chunk_id


# ---------------------------------------------------------
# 8. Parent-child relationships and provenance
# ---------------------------------------------------------

def test_all_children_link_to_parent():
    collection = chunk_extraction(LEGACY, full_text=HPI)

    assert collection.parent.metadata["is_parent"] is True
    assert collection.parent.text == HPI

    for chunk in collection.children:
        assert (
            chunk.metadata["parent_id"]
            == collection.parent.chunk_id
        )

        assert chunk.metadata["is_parent"] is False


def test_chunk_metadata_contains_case_report_and_version():
    collection = chunk_extraction(LEGACY)

    for chunk in collection.children:
        assert chunk.metadata["case_id"] == "C0138"
        assert chunk.metadata["report_id"] == 46
        assert chunk.metadata["chunker_version"]
        assert chunk.metadata["requires_human_review"] is True


# ---------------------------------------------------------
# 9. Clinical safety: no invented findings
# ---------------------------------------------------------

def test_empty_extraction_does_not_invent_diagnoses_or_symptoms():
    payload = {
        "case_id": "EMPTY1",
        "report_id": 100,
        "extraction_json": {
            "common": {},
            "department_specific": {},
            "metadata": {},
        },
    }

    collection = chunk_extraction(payload)
    chunks = by_type(collection)

    assert "diagnoses" not in chunks
    assert "symptoms_present" not in chunks
    assert "pertinent_negatives" not in chunks

    # Allergy status is intentionally always emitted.
    assert "allergy_status" in chunks


def test_family_history_is_not_promoted_to_patient_diagnosis():
    payload = {
        "case_id": "FAM1",
        "report_id": 101,
        "extraction_json": {
            "common": {
                "diagnoses": [],
                "clinical_history": [
                    {
                        "history_type": "family",
                        "relation": "father",
                        "description": "Father had asthma.",
                        "assertion": "FAMILY",
                        "fact_id": "family-asthma-1",
                    }
                ],
            },
            "department_specific": {},
        },
    }

    chunks = by_type(chunk_extraction(payload))

    assert "family_history" in chunks
    assert "diagnoses" not in chunks
    assert "asthma" in chunks["family_history"][0].text.lower()


def test_uncertain_symptom_is_not_marked_as_present():
    payload = {
        "case_id": "UNCERTAIN1",
        "report_id": 102,
        "extraction_json": {
            "common": {
                "symptoms": [
                    {
                        "name": "possible dizziness",
                        "assertion": "UNCERTAIN",
                        "evidence": "Possible dizziness was mentioned.",
                        "entity_id": "uncertain-1",
                    }
                ]
            },
            "department_specific": {},
        },
    }

    chunks = by_type(chunk_extraction(payload))

    assert "symptoms_uncertain" in chunks
    assert "symptoms_present" not in chunks


# ---------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------

def test_invalid_payload_type_raises_clear_error():
    with pytest.raises(TypeError, match="payload must be a JSON object"):
        chunk_extraction(None)


def test_missing_common_object_raises_clear_error():
    payload = {
        "case_id": "INVALID1",
        "extraction_json": {
            "department_specific": {},
        },
    }

    with pytest.raises(
        ValueError,
        match="Expected extraction_json.common",
    ):
        chunk_extraction(payload)
