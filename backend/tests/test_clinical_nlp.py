from app.services.clinical_nlp.extraction_service import (
    extract_clinical_entities,
)


def test_extract_clinical_entities():
    text = (
        "Patient reports chest pain and shortness of breath. "
        "Blood pressure: 120/80 mmHg. "
        "No fever."
    )

    result = extract_clinical_entities(text)

    assert any(
        entity.name == "chest pain"
        for entity in result.symptoms
    )
    assert any(
        entity.name == "shortness of breath"
        for entity in result.symptoms
    )
    assert any(
        entity.name == "fever" and entity.negated is True
        for entity in result.symptoms
    )
    assert any(
        entity.measurement == "blood pressure"
        and entity.value == "120/80"
        for entity in result.vital_signs
    )
