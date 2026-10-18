
"""Smoke test for MedFlow AI cross-encoder reranking."""

import logging
import sys
import traceback
from pathlib import Path

from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import SessionLocal
from app.services.retrieval.hybrid_retrieval_service import (
    search_clinical_chunks_hybrid,
)
from app.services.retrieval.reranker_service import (
    rerank_clinical_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

QUERY = "What are the symptoms of gallstones?"
PATIENT_ID = 4
CASE_ID = "C0148"
CANDIDATE_K = 20
TOP_K = 5


def main():
    db = None

    try:
        print("\nMEDFLOW AI — RERANKER SMOKE TEST")
        print(f"Query: {QUERY}")
        print(f"Patient ID: {PATIENT_ID}")
        print(f"Case ID: {CASE_ID}")

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        print("\n[PASS] Database connection")

        candidates = search_clinical_chunks_hybrid(
            db=db,
            query=QUERY,
            patient_id=PATIENT_ID,
            case_id=CASE_ID,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        print(f"[PASS] Hybrid retrieval returned {len(candidates)} candidates")

        if not candidates:
            raise RuntimeError(
                "Hybrid retrieval returned no candidates; "
                "reranking cannot be tested."
            )

        print("\n--- BEFORE RERANKING ---")
        for rank, item in enumerate(candidates[:TOP_K], start=1):
            print(
                f"{rank}. {item['chunk_id']} "
                f"| RRF={item.get('rrf_score')}"
            )

        reranked = rerank_clinical_chunks(
            query=QUERY,
            candidates=candidates,
            top_k=TOP_K,
        )

        if not reranked:
            raise AssertionError("Reranker returned no results")

        if len(reranked) > TOP_K:
            raise AssertionError("Reranker returned too many results")

        chunk_ids = [item["chunk_id"] for item in reranked]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise AssertionError("Reranker returned duplicate chunks")

        scores = [item["reranker_score"] for item in reranked]
        if not all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and __import__("math").isfinite(score)
            for score in scores
        ):
            raise AssertionError("Invalid reranker scores")

        if scores != sorted(scores, reverse=True):
            raise AssertionError("Results are not sorted by reranker score")

        for item in reranked:
            if item.get("patient_id") != PATIENT_ID:
                raise AssertionError("Unexpected patient in reranked results")
            if item.get("case_id") != CASE_ID:
                raise AssertionError("Unexpected case in reranked results")

        print("\n--- AFTER RERANKING ---")
        for rank, item in enumerate(reranked, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk ID: {item['chunk_id']}")
            print(f"Reranker score: {item['reranker_score']}")
            print(f"Original RRF score: {item.get('rrf_score')}")
            print(f"Dense rank: {item.get('dense_rank')}")
            print(f"Sparse rank: {item.get('sparse_rank')}")
            print(
                "Text: "
                + " ".join(item.get("chunk_text", "").split())[:250]
            )

        print("\n[PASS] Reranker returned valid, sorted results")
        print("\nRESULT: RERANKER SMOKE TEST PASSED")

    except Exception as exc:
        print(f"\nRESULT: TEST FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit_code = 1
        raise

    finally:
        if db is not None:
            try:
                db.close()
                print("Database session closed.")
            except Exception:
                logging.exception("Failed to close database session")


if __name__ == "__main__":
    main()
