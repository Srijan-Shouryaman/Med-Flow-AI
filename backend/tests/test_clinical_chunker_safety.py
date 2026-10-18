import copy
import pytest

from app.services.chunking.clinical_chunker import chunk_extraction


# ============================================================
# Test data
# ============================================================

def make_payload(
    patient_id=101,
    case_id="C-TEST-101",
    report_id=7,
    common=None,
    department_specific=None,
):
    extraction = {
        "patient_id": patient_id,
        "common": common or {},
        "department_specific": department_specific or {},
    }

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "report_id": report_id,
        "extraction_json": extraction,
    }


def all_chunks(result):
    return [result.parent, *result.children]


def chunks_of_type(result, chunk_type):
    return [
        chunk for chunk in result.children
        if chunk.chunk_type == chunk_type
    ]


def combined_text(chunks):
    return "\n".join(chunk.text for chunk in chunks).lower()


# ============================================================
# 1. Patient identity and isolation
# ============================================================

def test_missing_patient_id_is_rejected():
    payload = make_payload()
    del payload["patient_id"]
    del payload["extraction_json"]["patient_id"]

    with pytest.raises(ValueError):
        chunk_extraction(payload)


@pytest.mark.parametrize(
    "patient_id",
    ["", "   ", "null", "undefined", None],
)
def test_invalid_patient_id_is_rejected(patient_id):
    payload = make_payload(patient_id=patient_id)

    with pytest.raises(ValueError):
        chunk_extraction(payload)


def test_conflicting_patient_ids_are_rejected():
    payload = make_payload(patient_id=101)
    payload["extraction_json"]["patient_id"] = 202

    with pytest.raises(ValueError):
        chunk_extraction(payload)


def test_different_patients_have_distinct_chunk_ids():
    common = {
        "symptoms": [
            {
                "name": "cough",
                "assertion": "PRESENT",
                "fact_id": "fact-cough",
            }
        ]
    }

    patient_a = chunk_extraction(
        make_payload(patient_id=101, common=copy.deepcopy(common))
    )

    patient_b = chunk_extraction(
        make_payload(patient_id=202, common=copy.deepcopy(common))
    )

    ids_a = {chunk.chunk_id for chunk in patient_a.children}
    ids_b = {chunk.chunk_id for chunk in patient_b.children}

    assert ids_a
    assert ids_b
    assert ids_a.isdisjoint(ids_b)


def test_every_chunk_preserves_patient_scope():
    result = chunk_extraction(
        make_payload(
            patient_id=101,
            common={
                "symptoms": [
                    {
                        "name": "cough",
                        "assertion": "PRESENT",
                        "fact_id": "fact-cough",
                    }
                ]
            },
        )
    )

    for chunk in all_chunks(result):
        assert chunk.metadata.get("patient_id") == 101


# ============================================================
# 2. Clinical negation safety
# ============================================================

def test_absent_symptom_does_not_leak_into_positive_symptoms():
    result = chunk_extraction(
        make_payload(
            common={
                "symptoms": [
                    {
                        "name": "heartburn",
                        "assertion": "PRESENT",
                        "fact_id": "fact-heartburn",
                    },
                    {
                        "name": "vomiting blood",
                        "assertion": "ABSENT",
                        "fact_id": "fact-vomiting-blood",
                    },
                ]
            }
        )
    )

    positive_text = combined_text(
        chunks_of_type(result, "symptoms_present")
    )

    negative_text = combined_text(
        chunks_of_type(result, "pertinent_negatives")
    )

    assert "vomiting blood" not in positive_text
    assert "vomiting blood" in negative_text


def test_absent_diagnosis_is_not_recorded_as_confirmed():
    result = chunk_extraction(
        make_payload(
            common={
                "diagnoses": [
                    {
                        "name": "malignancy",
                        "assertion": "ABSENT",
                        "fact_id": "fact-malignancy",
                    }
                ]
            }
        )
    )

    diagnosis_text = combined_text(
        chunks_of_type(result, "diagnoses")
    )

    negative_text = combined_text(
        chunks_of_type(result, "pertinent_negatives")
    )

    assert "malignancy" not in diagnosis_text
    assert "malignancy" in negative_text


def test_unknown_symptom_is_not_assumed_present():
    result = chunk_extraction(
        make_payload(
            common={
                "symptoms": [
                    {
                        "name": "chest pain",
                        "assertion": "UNKNOWN",
                        "fact_id": "fact-chest-pain",
                    }
                ]
            }
        )
    )

    positive_text = combined_text(
        chunks_of_type(result, "symptoms_present")
    )

    assert "chest pain" not in positive_text


# ============================================================
# 3. Allergy safety
# ============================================================


def test_unknown_allergy_status_is_not_treated_as_nkda():
    result = chunk_extraction(
        make_payload(
            common={
                "allergies": [
                    {
                        "name": "allergy status",
                        "assertion": "UNKNOWN",
                        "status": "UNCERTAIN",
                        "fact_id": None,
                    }
                ]
            }
        )
    )

    allergy_chunks = [
        chunk
        for chunk in result.children
        if "allerg" in chunk.chunk_type.lower()
        or "allerg" in chunk.text.lower()
    ]

    assert allergy_chunks, (
        "An allergy-status chunk must exist even without a fact_id"
    )

    allergy_text = combined_text(allergy_chunks)

    # Unknown allergy status must be explicitly preserved.
    assert any(
        term in allergy_text
        for term in (
            "unknown",
            "uncertain",
            "not confirmed",
            "not documented",
        )
    )

    # Unknown status must not be converted into confirmed NKDA.
    assert "nkda is documented" not in allergy_text
    assert "no known drug allergies" not in allergy_text
    assert "no known allergies" not in allergy_text

    # Preserve the distinction between unknown status and no allergies.
    assert (
        "does not mean the patient has no allergies"
        in allergy_text
    )



# ============================================================
# 4. Missing data safety
# ============================================================

def test_unreported_endoscopy_is_not_recorded_as_performed():
    result = chunk_extraction(
        make_payload(
            common={
                "procedures": [
                    {
                        "name": "endoscopy",
                        "assertion": "NOT_PROVIDED",
                        "fact_id": "fact-endoscopy-gap",
                    }
                ]
            }
        )
    )

    procedure_text = combined_text(
        chunks_of_type(result, "procedures_tests")
    )

    gap_text = combined_text(
        chunks_of_type(result, "data_gaps")
    )

    assert "endoscopy performed" not in procedure_text
    assert "endoscopy" in gap_text


def test_missing_investigation_is_not_a_negative_result():
    result = chunk_extraction(
        make_payload(
            common={
                "investigations": [
                    {
                        "name": "chest x-ray",
                        "assertion": "NOT_PROVIDED",
                        "fact_id": "fact-xray-gap",
                    }
                ]
            }
        )
    )

    investigation_text = combined_text(
        chunks_of_type(result, "investigations")
    )

    gap_text = combined_text(
        chunks_of_type(result, "data_gaps")
    )

    assert "normal chest x-ray" not in investigation_text
    assert "no abnormality" not in investigation_text
    assert "chest x-ray" in gap_text


# ============================================================
# 5. Chunk integrity
# ============================================================

def test_chunk_ids_are_unique():
    result = chunk_extraction(
        make_payload(
            common={
                "symptoms": [
                    {
                        "name": "cough",
                        "assertion": "PRESENT",
                        "fact_id": "fact-cough",
                    },
                    {
                        "name": "fever",
                        "assertion": "ABSENT",
                        "fact_id": "fact-fever",
                    },
                ],
                "diagnoses": [
                    {
                        "name": "asthma",
                        "assertion": "PRESENT",
                        "fact_id": "fact-asthma",
                    }
                ],
            }
        )
    )

    ids = [chunk.chunk_id for chunk in all_chunks(result)]

    assert len(ids) == len(set(ids))


def test_chunking_is_deterministic():
    payload = make_payload(
        common={
            "symptoms": [
                {
                    "name": "cough",
                    "assertion": "PRESENT",
                    "fact_id": "fact-cough",
                }
            ]
        }
    )

    first = chunk_extraction(copy.deepcopy(payload))
    second = chunk_extraction(copy.deepcopy(payload))

    first_ids = [chunk.chunk_id for chunk in all_chunks(first)]
    second_ids = [chunk.chunk_id for chunk in all_chunks(second)]

    assert first_ids == second_ids


def test_children_reference_the_correct_parent():
    result = chunk_extraction(
        make_payload(
            common={
                "symptoms": [
                    {
                        "name": "cough",
                        "assertion": "PRESENT",
                        "fact_id": "fact-cough",
                    }
                ]
            }
        )
    )

    parent_id = result.parent.chunk_id

    for child in result.children:
        assert child.metadata.get("parent_id") == parent_id