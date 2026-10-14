
from app.services.clinical_nlp.extraction_service import (
    extract_clinical_entities,
    extract_department_specific_entities,
)


# Existing regression tests — preserve known working behavior

def test_lab_values():
    text = (
        "White blood cell count 7,800 cells/µL; "
        "platelet count 245,000 cells/µL."
    )
    result = extract_clinical_entities(text)
    labs = {
        (item.name.lower(), item.result)
        for item in result.investigations
        if item.result is not None
    }
    assert ("white blood cell count", "7,800") in labs, labs
    assert ("platelet count", "245,000") in labs, labs


def test_allergy_heading():
    result = extract_clinical_entities("Allergy Information")
    assert result.allergies == [], result.allergies


def test_negative_allergy_history():
    result = extract_clinical_entities(
        "No known drug allergies are documented."
    )
    assert all(
        item.substance.lower() != "are documented"
        for item in result.allergies
    ), result.allergies
    assert any(
        item.status == "NEGATED" for item in result.allergies
    ), result.allergies


def test_duplicate_troponin():
    result = extract_clinical_entities(
        "High-sensitivity troponin: 8 ng/L."
    )
    troponin = [
        item for item in result.investigations
        if item.name.lower() == "troponin"
    ]
    assert len(troponin) == 1, troponin
    assert troponin[0].result == "8", troponin
    assert troponin[0].unit.lower() == "ng/l", troponin


def test_ecg_heart_rate_extraction():
    result = extract_clinical_entities(
        "Electrocardiogram (ECG): Sinus rhythm at 96 beats/min."
    )
    heart_rates = [
        item for item in result.vital_signs
        if item.measurement.lower() == "heart rate"
    ]
    assert len(heart_rates) == 1, heart_rates
    assert heart_rates[0].value == "96", heart_rates
    assert heart_rates[0].unit.lower() == "bpm", heart_rates


def test_do_not_infer_missing_vitals():
    result = extract_clinical_entities(
        "The patient reports a persistent dry cough."
    )
    assert result.vital_signs == [], result.vital_signs


# Department 1: Cardiovascular / Pulmonary

def test_cardiovascular_symptom_and_diagnosis():
    result = extract_clinical_entities(
        "The patient reports chest pain. "
        "The patient has hypertension."
    )
    symptoms = {item.name.lower() for item in result.symptoms}
    diagnoses = {item.name.lower() for item in result.diagnoses}
    assert "chest pain" in symptoms, symptoms
    assert "hypertension" in diagnoses, diagnoses


def test_negated_cardiovascular_symptom():
    result = extract_clinical_entities(
        "The patient denies chest pain."
    )
    chest_pain = [
        item for item in result.symptoms
        if item.name.lower() == "chest pain"
    ]
    assert chest_pain, result.symptoms
    assert any(
        item.status == "NEGATED" and item.negated is True
        for item in chest_pain
    ), chest_pain


# Department 2: Consult - History and Phy.

def test_historical_diagnosis_and_history_record():
    result = extract_clinical_entities("History of asthma.")
    asthma = [
        item for item in result.diagnoses
        if item.name.lower() == "asthma"
    ]
    assert asthma, result.diagnoses
    assert asthma[0].status == "HISTORICAL", asthma
    assert any(
        item.status == "HISTORICAL"
        and "history of asthma" in item.description.lower()
        for item in result.clinical_history
    ), result.clinical_history


def test_medication_extraction():
    result = extract_clinical_entities(
        "The patient takes aspirin and metformin."
    )
    medications = {
        item.name.lower() for item in result.medications
    }
    assert "aspirin" in medications, medications
    assert "metformin" in medications, medications


# Department 3: Gastroenterology

def test_gastrointestinal_symptoms():
    result = extract_clinical_entities(
        "The patient reports abdominal pain. "
        "The patient also reports nausea and vomiting."
    )
    symptoms = {item.name.lower() for item in result.symptoms}
    assert "abdominal pain" in symptoms, symptoms
    assert "nausea" in symptoms, symptoms
    assert "vomiting" in symptoms, symptoms


def test_gastroenterology_expanded_symptoms():
    result = extract_clinical_entities(
        "The patient reports bloating and heartburn. "
        "The patient has diarrhea and difficulty swallowing."
    )
    symptoms = {item.name.lower() for item in result.symptoms}
    for expected in (
        "bloating",
        "heartburn",
        "diarrhea",
        "difficulty swallowing",
    ):
        assert expected in symptoms, symptoms


def test_gastroenterology_diagnoses():
    result = extract_clinical_entities(
        "The patient has gastroesophageal reflux disease. "
        "Acute pancreatitis is suspected."
    )
    diagnoses = {
        item.name.lower(): item for item in result.diagnoses
    }
    assert "gastroesophageal reflux disease" in diagnoses, diagnoses
    assert "acute pancreatitis" in diagnoses, diagnoses
    assert (
        diagnoses["acute pancreatitis"].status == "UNCERTAIN"
    ), diagnoses


def test_gastroenterology_numeric_investigation():
    result = extract_clinical_entities(
        "ALT: 58 U/L. Serum lipase: 42 U/L."
    )
    labs = {
        (item.name.lower(), item.result)
        for item in result.investigations
        if item.result is not None
    }
    assert ("alt", "58") in labs, labs
    assert ("serum lipase", "42") in labs, labs


# Department 4: Orthopedic

def test_musculoskeletal_related_findings():
    result = extract_clinical_entities(
        "The patient reports swelling and weakness."
    )
    symptoms = {item.name.lower() for item in result.symptoms}
    assert "swelling" in symptoms, symptoms
    assert "weakness" in symptoms, symptoms


# Department 5: Radiology

def test_imaging_procedure_extraction():
    result = extract_clinical_entities(
        "A chest X-ray was performed. An MRI was reviewed."
    )
    procedures = {
        item.name.lower() for item in result.procedures
    }
    assert "x-ray" in procedures, procedures
    assert "mri" in procedures, procedures


# Cross-department checks

def test_uncertain_diagnosis_is_not_marked_as_confirmed():
    result = extract_clinical_entities("Possible pneumonia.")
    pneumonia = [
        item for item in result.diagnoses
        if item.name.lower() == "pneumonia"
    ]
    assert pneumonia, result.diagnoses
    assert pneumonia[0].status == "UNCERTAIN", pneumonia
    assert pneumonia[0].negated is None, pneumonia


def test_duplicate_symptom_mentions_are_consolidated():
    result = extract_clinical_entities(
        "The patient reports cough. "
        "The patient continues to report cough."
    )
    cough = [
        item for item in result.symptoms
        if item.name.lower() == "cough"
    ]
    assert len(cough) == 1, cough
    assert " | " in cough[0].evidence, cough


# Department-specific extraction capability

def test_department_specific_cardiovascular_pulmonary_findings():
    result = extract_department_specific_entities(
        "Cardiac examination reveals a heart murmur. "
        "Lung auscultation reveals crackles."
    )
    cv = result["cardiovascular_pulmonary"]
    names = {item["name"].lower() for item in cv["findings"]}
    assert "heart murmur" in names, names
    assert "crackles" in names, names


def test_department_specific_consult_history_physical_findings():
    result = extract_department_specific_entities(
        "Physical examination reveals abdominal tenderness."
    )
    findings = result["consult_history_physical"]["findings"]
    assert any(
        item["name"].lower() == "abdominal tenderness"
        for item in findings
    ), findings


def test_department_specific_gastroenterology_findings():
    result = extract_department_specific_entities(
        "Examination reveals rebound tenderness and ascites."
    )
    findings = result["gastroenterology"]["findings"]
    names = {item["name"].lower() for item in findings}
    assert "rebound tenderness" in names, names
    assert "ascites" in names, names


def test_department_specific_orthopedic_findings():
    result = extract_department_specific_entities(
        "Examination reveals reduced range of motion "
        "and joint deformity."
    )
    findings = result["orthopedic"]["findings"]
    names = {item["name"].lower() for item in findings}
    assert "reduced range of motion" in names, names
    assert "joint deformity" in names, names


def test_department_specific_radiology_findings():
    result = extract_department_specific_entities(
        "Chest CT shows consolidation and pleural effusion."
    )
    findings = result["radiology"]["findings"]
    names = {item["name"].lower() for item in findings}
    assert "consolidation" in names, names
    assert "pleural effusion" in names, names


def test_department_specific_negated_radiology_finding():
    result = extract_department_specific_entities(
        "No pleural effusion is seen on chest radiograph."
    )
    findings = [
        item
        for item in result["radiology"]["findings"]
        if item["name"].lower() == "pleural effusion"
    ]
    assert findings, result["radiology"]["findings"]
    assert any(
        item["status"] == "NEGATED" for item in findings
    ), findings
    assert all(
        "pleural effusion" in item["evidence"].lower()
        for item in findings
    ), findings


def run_tests():
    tests = [
        test_lab_values,
        test_allergy_heading,
        test_negative_allergy_history,
        test_duplicate_troponin,
        test_ecg_heart_rate_extraction,
        test_do_not_infer_missing_vitals,
        test_cardiovascular_symptom_and_diagnosis,
        test_negated_cardiovascular_symptom,
        test_historical_diagnosis_and_history_record,
        test_medication_extraction,
        test_gastrointestinal_symptoms,
        test_gastroenterology_expanded_symptoms,
        test_gastroenterology_diagnoses,
        test_gastroenterology_numeric_investigation,
        test_musculoskeletal_related_findings,
        test_imaging_procedure_extraction,
        test_uncertain_diagnosis_is_not_marked_as_confirmed,
        test_duplicate_symptom_mentions_are_consolidated,
        test_department_specific_cardiovascular_pulmonary_findings,
        test_department_specific_consult_history_physical_findings,
        test_department_specific_gastroenterology_findings,
        test_department_specific_orthopedic_findings,
        test_department_specific_radiology_findings,
        test_department_specific_negated_radiology_finding,
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
            print(f"FAIL: {test.__name__}")
            print(f"  Details: {error}")
            failed += 1
        except Exception as error:
            print(f"ERROR: {test.__name__}: {error}")
            errors += 1

    print("\nRegression Test Results")
    print("-----------------------")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Total:  {len(tests)}")

    if failed or errors:
        print("\nSome tests need attention.")
    else:
        print("\nAll regression tests passed.")

    return 1 if failed or errors else 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
