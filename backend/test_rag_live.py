
"""Live end-to-end RAG integration test for MedFlow AI."""

import importlib
import logging
import sys
import traceback
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.rag.rag_pipeline_service import answer_clinical_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_live_test")

# Existing test data. No database writes are performed by this test.
PATIENT_ID = 4
CASE_ID = "C0148"
QUERY = "What symptoms and diagnoses are documented in this patient's reports?"

CANDIDATE_K = 20
RERANK_TOP_K = 5
CONTEXT_TOKEN_BUDGET = 6000


def create_session():
    """Locate the application's SQLAlchemy session factory."""
    candidates = [
        ("app.db.session", "SessionLocal"),
        ("app.database", "SessionLocal"),
        ("app.core.database", "SessionLocal"),
        ("app.db.database", "SessionLocal"),
    ]

    errors = []

    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute_name, None)

            if callable(factory):
                session = factory()
                logger.info(
                    "Database session created using %s.%s",
                    module_name,
                    attribute_name,
                )
                return session

            errors.append(f"{module_name}: SessionLocal not found")

        except Exception as exc:
            errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Could not locate the SQLAlchemy session factory:\n"
        + "\n".join(errors)
    )


def main():
    db = None

    try:
        logger.info("Starting live RAG integration test")
        logger.info(
            "Patient ID: %s | Case ID: %s",
            PATIENT_ID,
            CASE_ID,
        )

        db = create_session()

        result = answer_clinical_question(
            db=db,
            query=QUERY,
            patient_id=PATIENT_ID,
            case_id=CASE_ID,
            candidate_k=CANDIDATE_K,
            rerank_top_k=RERANK_TOP_K,
            context_token_budget=CONTEXT_TOKEN_BUDGET,
        )

        # Validate the orchestration response.
        assert isinstance(result, dict), (
            "Pipeline did not return a dictionary"
        )
        assert result.get("patient_id") == PATIENT_ID, (
            "Unexpected patient ID in result"
        )
        assert result.get("case_id") == CASE_ID, (
            "Unexpected case ID in result"
        )
        assert isinstance(result.get("retrieval_count"), int), (
            "Invalid retrieval count"
        )
        assert isinstance(result.get("reranked_count"), int), (
            "Invalid reranked count"
        )

        context = result.get("context")
        generation = result.get("generation")

        assert isinstance(context, dict), (
            "Pipeline returned invalid assembled context"
        )
        assert isinstance(generation, dict), (
            "Pipeline returned invalid generation result"
        )

        # Check patient/case scope of the assembled sources.
        for source in context.get("sources", []):
            assert source.get("patient_id") == PATIENT_ID, (
                "Cross-patient source detected"
            )
            assert source.get("case_id") == CASE_ID, (
                "Cross-case source detected"
            )

        available_ids = {
            source["chunk_id"]
            for source in context.get("sources", [])
        }

        cited_ids = generation.get("cited_chunk_ids", [])
        assert isinstance(cited_ids, list), (
            "Invalid citation list"
        )

        unknown_citations = set(cited_ids) - available_ids
        assert not unknown_citations, (
            f"Unknown citations returned: {sorted(unknown_citations)}"
        )

        print("\n" + "=" * 55)
        print("LIVE RAG INTEGRATION RESULTS")
        print("=" * 55)
        print(f"Patient ID:          {PATIENT_ID}")
        print(f"Case ID:             {CASE_ID}")
        print(f"Query:               {QUERY}")
        print(f"Retrieved candidates:{result['retrieval_count']:>4}")
        print(f"Reranked candidates: {result['reranked_count']:>4}")
        print(
            "Context chunks:      "
            f"{context.get('included_chunk_count', 0):>4}"
        )
        print(
            f"Gemini model:        {generation.get('model')}"
        )
        print(
            "Insufficient evidence: "
            f"{generation.get('insufficient_evidence')}"
        )
        print(f"Human review needed: {generation.get('needs_human_review')}")
        print(f"Cited source count:  {len(cited_ids)}")

        print("\nGENERATED ANSWER")
        print("-" * 55)
        print(generation.get("answer", "(No answer returned)"))

        print("\nCITED SOURCE IDS")
        if cited_ids:
            for chunk_id in cited_ids:
                print(f"- {chunk_id}")
        else:
            print("(No sources cited)")

        assert generation.get("needs_human_review") is True, (
            "Human review must remain required"
        )

        if result["retrieval_count"] == 0:
            print(
                "\nWARNING: No candidates retrieved. "
                "The pipeline may correctly report insufficient evidence."
            )
        elif generation.get("insufficient_evidence"):
            print(
                "\nWARNING: Retrieval found evidence, but Gemini "
                "reported that it was insufficient."
            )

        print("\nRESULT: LIVE RAG PIPELINE TEST PASSED")

    except Exception as exc:
        logger.error(
            "Live RAG integration test failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        traceback.print_exc()
        print("\nRESULT: LIVE RAG PIPELINE TEST FAILED")
        raise

    finally:
        if db is not None:
            try:
                db.close()
                logger.info("Database session closed")
            except Exception:
                logger.exception("Failed to close database session")


if __name__ == "__main__":
    main()
