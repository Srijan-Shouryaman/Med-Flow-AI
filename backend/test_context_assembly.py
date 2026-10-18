
from app.services.retrieval.context_assembly_service import (
    assemble_clinical_context,
)


def make_candidate(
    chunk_id,
    text,
    patient_id=4,
    case_id="C0148",
    report_id=56,
    reranker_score=1.0,
    reranker_rank=1,
):
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "patient_id": patient_id,
        "case_id": case_id,
        "report_id": report_id,
        "reranker_score": reranker_score,
        "reranker_rank": reranker_rank,
        "metadata": {},
    }


def test_context_assembly():
    candidates = [
        make_candidate(
            "chunk-1",
            "The patient reports abdominal pain and nausea.",
            reranker_score=3.0,
            reranker_rank=1,
        ),
        make_candidate(
            "chunk-2",
            "Ultrasound findings indicate gallstones.",
            reranker_score=2.0,
            reranker_rank=2,
        ),
    ]

    result = assemble_clinical_context(
        query="What are the patient's symptoms?",
        candidates=candidates,
        patient_id=4,
        case_id="C0148",
        token_budget=1000,
    )

    assert result["has_evidence"] is True
    assert result["included_chunk_count"] == 2
    assert result["patient_id"] == 4
    assert result["case_id"] == "C0148"
    assert result["sources"][0]["chunk_id"] == "chunk-1"
    assert result["sources"][1]["chunk_id"] == "chunk-2"
    assert len(result["sources"]) == 2
    assert "abdominal pain and nausea" in result["context"]
    assert "gallstones" in result["context"]

    print("[PASS] Basic context assembly")


def test_patient_isolation():
    candidates = [
        make_candidate(
            "chunk-1",
            "Patient has abdominal pain.",
        ),
        make_candidate(
            "chunk-other",
            "Different patient's clinical information.",
            patient_id=5,
            reranker_rank=2,
        ),
    ]

    try:
        assemble_clinical_context(
            query="abdominal pain",
            candidates=candidates,
            patient_id=4,
            case_id="C0148",
        )
    except ValueError:
        print("[PASS] Cross-patient evidence rejected")
    else:
        raise AssertionError(
            "Cross-patient evidence was not rejected"
        )


def test_case_isolation():
    candidates = [
        make_candidate(
            "chunk-other-case",
            "Clinical information from another case.",
            case_id="C9999",
        ),
    ]

    try:
        assemble_clinical_context(
            query="clinical information",
            candidates=candidates,
            patient_id=4,
            case_id="C0148",
        )
    except ValueError:
        print("[PASS] Cross-case evidence rejected")
    else:
        raise AssertionError(
            "Cross-case evidence was not rejected"
        )


def test_exact_duplicate_removal():
    candidates = [
        make_candidate(
            "chunk-1",
            "Patient reports abdominal pain.",
            reranker_rank=1,
        ),
        make_candidate(
            "chunk-2",
            "Patient reports abdominal pain.",
            reranker_rank=2,
        ),
    ]

    result = assemble_clinical_context(
        query="abdominal pain",
        candidates=candidates,
        patient_id=4,
        case_id="C0148",
    )

    assert result["included_chunk_count"] == 1
    assert result["duplicate_chunk_count"] == 1
    assert len(result["sources"]) == 1

    print("[PASS] Exact duplicate removal")


def test_token_budget():
    candidates = [
        make_candidate(
            "chunk-1",
            "A" * 500,
            reranker_rank=1,
        ),
        make_candidate(
            "chunk-2",
            "B" * 500,
            reranker_rank=2,
        ),
    ]

    result = assemble_clinical_context(
        query="clinical findings",
        candidates=candidates,
        patient_id=4,
        case_id="C0148",
        token_budget=30,
    )

    assert result["token_count"] <= 30
    assert result["excluded_chunk_count"] >= 1

    print("[PASS] Context token budget enforced")


def test_empty_candidates():
    result = assemble_clinical_context(
        query="What are the clinical findings?",
        candidates=[],
        patient_id=4,
        case_id="C0148",
    )

    assert result["has_evidence"] is False
    assert result["included_chunk_count"] == 0
    assert result["sources"] == []
    assert result["context"] == ""

    print("[PASS] Empty candidate handling")


if __name__ == "__main__":
    tests = [
        test_context_assembly,
        test_patient_isolation,
        test_case_isolation,
        test_exact_duplicate_removal,
        test_token_budget,
        test_empty_candidates,
    ]

    for test in tests:
        test()

    print("\nRESULT: CONTEXT ASSEMBLY TESTS PASSED")
