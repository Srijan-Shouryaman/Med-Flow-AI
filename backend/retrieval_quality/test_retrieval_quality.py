"""Fast sanity/safety checks plus optional live retrieval tests."""
import pytest
from .golden_set import CASES, CHUNK_TYPES, FORBIDDEN, LABELS, ALLERGY_EXPECTED
from .adapter import retrieve
from .metrics import norm, is_relevant, score_case

def test_golden_set_covers_all_child_chunk_types():
    covered={t for case in CASES for t in case.types}
    assert set(CHUNK_TYPES) <= covered

def test_golden_case_ids_are_unique():
    ids=[case.id for case in CASES]
    assert len(ids)==len(set(ids))

def test_allergy_expectations_cover_all_three_reports():
    assert set(ALLERGY_EXPECTED.values()) == {"NONE_KNOWN","PRESENT","CONFLICTING"}
    assert len(ALLERGY_EXPECTED)==3

def test_relevance_predicate_requires_type_report_and_evidence():
    case=next(c for c in CASES if c.id=="inv01")
    good=norm({"chunk_type":"investigations","report_date":"2026-06-04","text":"Troponin was elevated"})
    wrong_type=norm({"chunk_type":"clinical_history","report_date":"2026-06-04","text":"Troponin was elevated"})
    wrong_date=norm({"chunk_type":"investigations","report_date":"2026-03-12","text":"Troponin was elevated"})
    assert is_relevant(case,good)
    assert not is_relevant(case,wrong_type)
    assert not is_relevant(case,wrong_date)

def test_score_case_metrics_are_bounded():
    case=next(c for c in CASES if c.id=="inv01")
    rows=[norm({"chunk_type":"investigations","report_date":"2026-06-04","text":"Troponin 2450"})]
    score=score_case(case,rows,5)
    assert score["hit"]==1.0
    assert score["rr"]==1.0
    assert score["precision"]==pytest.approx(0.2)
    assert 0 <= score["report_recall"] <= 1

@pytest.mark.parametrize("mode",["sparse","dense","hybrid","rerank"])
def test_live_retrieval_returns_patient_scoped_chunks(mode):
    rows=retrieve("troponin NSTEMI coronary angiography",patient_id=5,k=5,mode=mode)
    assert rows, f"{mode} returned no results. Confirm Patient 5 reports are processed."
    for row in rows:
        assert row["patient_id"] in (None,5), f"Patient isolation issue: {row['patient_id']}"

@pytest.mark.parametrize("mode",["sparse","dense","hybrid","rerank"])
def test_invalid_patient_id_fails_before_database_access(mode):
    with pytest.raises(ValueError):
        retrieve("cardiac history",patient_id=0,k=5,mode=mode)

@pytest.mark.parametrize("mode",["sparse","dense","hybrid","rerank"])
@pytest.mark.parametrize("case",CASES,ids=lambda case:case.id)
def test_golden_query_has_relevant_result_in_top_k(mode,case):
    rows=[norm(c) for c in retrieve(case.query,patient_id=5,k=5,mode=mode)]
    assert rows, f"{mode} returned no results for {case.id}"
    assert any(is_relevant(case,c) for c in rows), (
        f"{mode} missed {case.id}; expected types={case.types}, reports={case.reports}, "
        f"terms={case.terms}. Returned: {[(c['chunk_type'],c['_report'],c['_text'][:120]) for c in rows]}"
    )

@pytest.mark.parametrize("mode",["sparse","dense","hybrid","rerank"])
@pytest.mark.parametrize("rule",FORBIDDEN,ids=lambda rule:rule[0])
def test_assertion_type_separation(mode,rule):
    _,query,forbidden_type,term,reports,why=rule
    rows=[norm(c) for c in retrieve(query,patient_id=5,k=5,mode=mode)]
    leaks=[c for c in rows if c["chunk_type"]==forbidden_type and term in c["_text"]
           and (not reports or c["_report"] in reports)]
    assert not leaks, f"{mode} assertion/type separation: {why}; leaked {leaks}"

@pytest.mark.parametrize("mode",["sparse","dense","hybrid","rerank"])
@pytest.mark.parametrize("label",LABELS,ids=lambda label:label[0])
def test_procedure_completion_labels(mode,label):
    _,query,expected_type,term,report,must_any,forbid_any=label
    rows=[norm(c) for c in retrieve(query,patient_id=5,k=5,mode=mode)]
    candidates=[c for c in rows if c["chunk_type"]==expected_type and term in c["_text"] and c["_report"]==report]
    if not candidates:
        pytest.fail(f"{mode} did not retrieve expected procedure chunk for {query!r} ({report})")
    text=candidates[0]["_text"]
    if must_any:
        assert any(x in text for x in must_any), f"Unconfirmed procedure not labeled as mention: {text}"
    assert not any(x in text for x in forbid_any), f"Completed procedure incorrectly labeled mention-only: {text}"
