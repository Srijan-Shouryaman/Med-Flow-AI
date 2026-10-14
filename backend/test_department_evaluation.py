
from app.services.clinical_nlp.extraction_service import (
    extract_department_specific_entities,
)


def get_names(result, department, category):
    """Return normalized extracted entity names."""
    return {
        item["name"].lower()
        for item in result[department][category]
    }


def test_cardiovascular_pulmonary_case():
    text = (
        "The patient has atrial fibrillation. "
        "ECG shows an irregular rhythm."
    )
    result = extract_department_specific_entities(text)

    assert "atrial fibrillation" in get_names(
        result, "cardiovascular_pulmonary", "diagnoses"
    )
    assert "irregular rhythm" in get_names(
        result, "cardiovascular_pulmonary", "findings"
    )


def test_consultation_and_medical_history_case():
    text = (
        "Family history of diabetes mellitus. "
        "Physical examination reveals abdominal tenderness."
    )
    result = extract_department_specific_entities(text)

    assert "family history" in get_names(
        result, "consult_history_physical", "history"
    )
    assert "abdominal tenderness" in get_names(
        result, "consult_history_physical", "findings"
    )


def test_gastroenterology_case():
    text = (
        "The patient reports melena and abdominal pain. "
        "Examination reveals ascites."
    )
    result = extract_department_specific_entities(text)

    assert "melena" in get_names(
        result, "gastroenterology", "symptoms"
    )
    assert "abdominal pain" in get_names(
        result, "gastroenterology", "symptoms"
    )
    assert "ascites" in get_names(
        result, "gastroenterology", "findings"
    )


def test_orthopedic_case():
    text = (
        "The patient has a meniscal tear. "
        "Examination reveals joint instability."
    )
    result = extract_department_specific_entities(text)

    assert "meniscal tear" in get_names(
        result, "orthopedic", "diagnoses"
    )
    assert "joint instability" in get_names(
        result, "orthopedic", "findings"
    )


def test_radiology_case():
    text = (
        "Chest CT shows a lung nodule and pleural effusion."
    )
    result = extract_department_specific_entities(text)

    investigations = get_names(
        result, "radiology", "investigations"
    )
    findings = get_names(
        result, "radiology", "findings"
    )

    assert any(
        name in investigations
        for name in (
            "ct scan",
            "computed tomography",
            "chest ct",
        )
    ), f"CT investigation missing: {investigations}"

    assert "lung nodule" in findings, (
        f"Lung nodule missing: {findings}"
    )

    assert "pleural effusion" in findings, (
        f"Pleural effusion missing: {findings}"
    )


def test_mixed_specialty_report():
    text = (
        "The patient has atrial fibrillation. "
        "The patient also reports abdominal pain. "
        "Abdominal CT shows ascites and an osteophyte."
    )
    result = extract_department_specific_entities(text)

    assert "atrial fibrillation" in get_names(
        result, "cardiovascular_pulmonary", "diagnoses"
    )
    assert "abdominal pain" in get_names(
        result, "gastroenterology", "symptoms"
    )
    assert "ascites" in get_names(
        result, "gastroenterology", "findings"
    )
    assert "osteophyte" in get_names(
        result, "radiology", "findings"
    )


def test_negated_finding_preserves_context():
    text = "No pleural effusion is seen on chest radiograph."
    result = extract_department_specific_entities(text)

    findings = [
        item
        for item in result["radiology"]["findings"]
        if item["name"].lower() == "pleural effusion"
    ]

    assert findings, (
        "Expected pleural effusion mention to be extracted"
    )
    assert any(
        item["status"] == "NEGATED" for item in findings
    ), findings
    assert all(
        "pleural effusion" in item["evidence"].lower()
        for item in findings
    ), findings


def test_uncertain_diagnosis_preserves_context():
    text = "Possible acute pancreatitis."
    result = extract_department_specific_entities(text)

    diagnoses = [
        item
        for item in result["gastroenterology"]["diagnoses"]
        if item["name"].lower() == "acute pancreatitis"
    ]

    assert diagnoses, (
        "Expected possible diagnosis to be extracted"
    )
    assert any(
        item["status"] == "UNCERTAIN" for item in diagnoses
    ), diagnoses


def run_tests():
    tests = [
        test_cardiovascular_pulmonary_case,
        test_consultation_and_medical_history_case,
        test_gastroenterology_case,
        test_orthopedic_case,
        test_radiology_case,
        test_mixed_specialty_report,
        test_negated_finding_preserves_context,
        test_uncertain_diagnosis_preserves_context,
    ]

    passed = 0
    failed = 0
    errors = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as error:
            print(f"FAIL: {test.__name__}: {error}")
            failed += 1
        except Exception as error:
            print(f"ERROR: {test.__name__}: {error}")
            errors += 1

    print("\nDepartment Evaluation Results")
    print("-----------------------------")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Total:  {len(tests)}")

    return 1 if failed or errors else 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
