
"""Mocked end-to-end tests for the MedFlow AI RAG pipeline."""

from unittest.mock import patch, MagicMock

from app.services.rag.rag_pipeline_service import answer_clinical_question


def make_candidates():
    return [
        {
            "chunk_id": "chunk-001",
            "patient_id": 4,
            "case_id": "C0148",
            "report_id": 56,
            "chunk_text": "The patient reports abdominal pain and nausea.",
            "metadata": {},
            "rrf_score": 0.03,
            "dense_rank": 1,
            "sparse_rank": 2,
        },
        {
            "chunk_id": "chunk-002",
            "patient_id": 4,
            "case_id": "C0148",
            "report_id": 56,
            "chunk_text": "The report documents suspected gallstones.",
            "metadata": {},
            "rrf_score": 0.02,
            "dense_rank": 2,
            "sparse_rank": 1,
        },
    ]


def make_assembled_context():
    return {
        "patient_id": 4,
        "case_id": "C0148",
        "query": "What symptoms are documented?",
        "context": (
            "[Source: chunk-001]\n"
            "The patient reports abdominal pain and nausea.\n\n"
            "[Source: chunk-002]\n"
            "The report documents suspected gallstones."
        ),
        "sources": [
            {
                "chunk_id": "chunk-001",
                "patient_id": 4,
                "case_id": "C0148",
                "report_id": 56,
                "reranker_rank": 1,
                "reranker_score": 2.1,
                "dense_rank": 1,
                "sparse_rank": 2,
                "rrf_score": 0.03,
            },
            {
                "chunk_id": "chunk-002",
                "patient_id": 4,
                "case_id": "C0148",
                "report_id": 56,
                "reranker_rank": 2,
                "reranker_score": 1.4,
                "dense_rank": 2,
                "sparse_rank": 1,
                "rrf_score": 0.02,
            },
        ],
        "included_chunk_count": 2,
        "excluded_chunk_count": 0,
        "duplicate_chunk_count": 0,
        "token_count": 100,
        "token_budget": 6000,
        "has_evidence": True,
    }


def make_generation():
    return {
        "answer": "The documented symptoms are abdominal pain and nausea.",
        "cited_chunk_ids": ["chunk-001"],
        "sources": [{"chunk_id": "chunk-001", "report_id": 56}],
        "insufficient_evidence": False,
        "evidence_used": True,
        "patient_id": 4,
        "case_id": "C0148",
        "model": "mock-model",
        "needs_human_review": True,
    }


def test_complete_pipeline():
    db = MagicMock()
    candidates = make_candidates()
    assembled = make_assembled_context()
    generation = make_generation()

    with (
        patch(
            "app.services.rag.rag_pipeline_service."
            "search_clinical_chunks_hybrid",
            return_value=candidates,
        ) as hybrid,
        patch(
            "app.services.rag.rag_pipeline_service."
            "rerank_clinical_chunks",
            return_value=candidates,
        ) as reranker,
        patch(
            "app.services.rag.rag_pipeline_service."
            "assemble_clinical_context",
            return_value=assembled,
        ) as assembler,
        patch(
            "app.services.rag.rag_pipeline_service."
            "generate_clinical_answer",
            return_value=generation,
        ) as generator,
    ):
        result = answer_clinical_question(
            db=db,
            query="What symptoms are documented?",
            patient_id=4,
            case_id="C0148",
        )

        assert result["patient_id"] == 4
        assert result["case_id"] == "C0148"
        assert result["retrieval_count"] == 2
        assert result["reranked_count"] == 2
        assert result["generation"]["answer"].startswith(
            "The documented symptoms"
        )
        assert result["needs_human_review"] is True

        hybrid.assert_called_once()
        reranker.assert_called_once()
        assembler.assert_called_once()
        generator.assert_called_once()

    print("[PASS] Complete pipeline orchestration")


def test_empty_retrieval():
    db = MagicMock()

    with (
        patch(
            "app.services.rag.rag_pipeline_service."
            "search_clinical_chunks_hybrid",
            return_value=[],
        ) as hybrid,
        patch(
            "app.services.rag.rag_pipeline_service."
            "rerank_clinical_chunks",
        ) as reranker,
        patch(
            "app.services.rag.rag_pipeline_service."
            "assemble_clinical_context",
            return_value={
                "patient_id": 4,
                "case_id": "C0148",
                "query": "Unknown information",
                "context": "",
                "sources": [],
                "included_chunk_count": 0,
                "excluded_chunk_count": 0,
                "duplicate_chunk_count": 0,
                "token_count": 0,
                "token_budget": 6000,
                "has_evidence": False,
            },
        ) as assembler,
        patch(
            "app.services.rag.rag_pipeline_service."
            "generate_clinical_answer",
            return_value={
                "answer": "Insufficient evidence.",
                "cited_chunk_ids": [],
                "sources": [],
                "insufficient_evidence": True,
                "evidence_used": False,
                "patient_id": 4,
                "case_id": "C0148",
                "model": "mock-model",
                "needs_human_review": True,
            },
        ) as generator,
    ):
        result = answer_clinical_question(
            db=db,
            query="Unknown information",
            patient_id=4,
            case_id="C0148",
        )

        assert result["retrieval_count"] == 0
        assert result["reranked_count"] == 0
        reranker.assert_not_called()
        assembler.assert_called_once()
        generator.assert_called_once()
        assert result["generation"]["insufficient_evidence"] is True

    print("[PASS] Empty retrieval handled safely")


def test_pipeline_failure_propagates():
    db = MagicMock()

    with patch(
        "app.services.rag.rag_pipeline_service."
        "search_clinical_chunks_hybrid",
        side_effect=RuntimeError("Simulated retrieval failure"),
    ):
        try:
            answer_clinical_question(
                db=db,
                query="Test query",
                patient_id=4,
                case_id="C0148",
            )
        except RuntimeError as exc:
            assert "Simulated retrieval failure" in str(exc)
        else:
            raise AssertionError("Pipeline silently swallowed a failure")

    print("[PASS] Unexpected failures propagate")


def main():
    test_complete_pipeline()
    test_empty_retrieval()
    test_pipeline_failure_propagates()
    print("\nRESULT: RAG PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
