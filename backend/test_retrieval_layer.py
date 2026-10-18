
"""Global read-only test for the MedFlow AI retrieval layer."""

import importlib
import logging
import math
import sys
import traceback
from pathlib import Path

from sqlalchemy import text

# Ensure the backend root is importable when this file is run directly.
BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.retrieval.dense_retrieval_service import (
    search_clinical_chunks,
)
from app.services.retrieval.sparse_retrieval_service import (
    search_clinical_chunks_sparse,
)
from app.services.retrieval.hybrid_retrieval_service import (
    search_clinical_chunks_hybrid,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retrieval_layer_test")

# Test configuration
PATIENT_ID = 4
CASE_ID = "C0148"
QUERY = "What are the symptoms of gallstones?"
TOP_K = 5
CANDIDATE_K = 20


def create_session():
    """Find the application's SQLAlchemy session factory."""
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

            errors.append(
                f"{module_name}: {attribute_name} not found"
            )
        except Exception as exc:
            errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    details = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(
        "Could not locate the application's SQLAlchemy session factory.\n"
        "Update create_session() to match your actual database module:\n"
        f"{details}"
    )


def validate_results(name, results, expected_patient, expected_case):
    """Check common result properties without modifying database data."""
    assert isinstance(results, list), (
        f"{name}: expected a list, got {type(results).__name__}"
    )

    chunk_ids = []

    for index, item in enumerate(results, start=1):
        assert isinstance(item, dict), (
            f"{name}: result {index} is not a dictionary"
        )

        chunk_id = item.get("chunk_id")
        assert isinstance(chunk_id, str) and chunk_id.strip(), (
            f"{name}: result {index} has an invalid chunk_id"
        )
        chunk_ids.append(chunk_id)

        assert item.get("patient_id") == expected_patient, (
            f"{name}: unexpected patient_id in {chunk_id}"
        )

        assert item.get("case_id") == expected_case, (
            f"{name}: unexpected case_id in {chunk_id}"
        )

    assert len(chunk_ids) == len(set(chunk_ids)), (
        f"{name}: duplicate chunk IDs found"
    )

    return True


def validate_scores(name, results, score_field, descending=False):
    """Check scores are finite and optionally sorted descending."""
    scores = []

    for item in results:
        try:
            score = float(item[score_field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AssertionError(
                f"{name}: invalid {score_field} in "
                f"{item.get('chunk_id', '<unknown chunk>')}"
            ) from exc

        assert math.isfinite(score), (
            f"{name}: non-finite {score_field}"
        )
        scores.append(score)

    if descending:
        assert scores == sorted(scores, reverse=True), (
            f"{name}: results are not sorted by {score_field}"
        )

    return True


def print_results(name, results, score_field):
    print(f"\n{'=' * 72}")
    print(f"{name} — {len(results)} result(s)")
    print("=" * 72)

    if not results:
        print("No matching chunks found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {item.get('chunk_id')}")
        print(f"Patient: {item.get('patient_id')}")
        print(f"Case: {item.get('case_id')}")
        print(f"Score ({score_field}): {item.get(score_field)}")

        if name == "HYBRID RETRIEVAL":
            print(f"Dense rank: {item.get('dense_rank')}")
            print(f"Sparse rank: {item.get('sparse_rank')}")
            print(f"Dense score: {item.get('dense_similarity_score')}")
            print(f"Sparse score: {item.get('sparse_score')}")
            print(f"RRF score: {item.get('rrf_score')}")

        chunk_text = str(item.get("chunk_text", ""))
        preview = " ".join(chunk_text.split())[:250]
        print(f"Text preview: {preview or '[empty]'}")


def main():
    db = None
    passed = []
    failed = []

    try:
        print("\nMEDFLOW AI — GLOBAL RETRIEVAL LAYER TEST")
        print(f"Query: {QUERY}")
        print(f"Patient ID: {PATIENT_ID}")
        print(f"Case ID: {CASE_ID}")
        print(f"Final top_k: {TOP_K}")
        print(f"Candidate pool: {CANDIDATE_K}")

        db = create_session()

        # 1. Verify database connectivity.
        db.execute(text("SELECT 1"))
        db.commit()
        passed.append("Database connectivity")
        print("\n[PASS] Database connectivity")

        # 2. Dense retrieval.
        try:
            dense = search_clinical_chunks(
                db=db,
                query=QUERY,
                patient_id=PATIENT_ID,
                top_k=TOP_K,
                case_id=CASE_ID,
            )

            validate_results(
                "Dense retrieval", dense, PATIENT_ID, CASE_ID
            )
            validate_scores(
                "Dense retrieval",
                dense,
                "similarity_score",
                descending=True,
            )
            print_results("DENSE RETRIEVAL", dense, "similarity_score")
            passed.append("Dense retrieval")
            print("[PASS] Dense retrieval")
        except Exception as exc:
            failed.append("Dense retrieval")
            print(f"\n[FAIL] Dense retrieval: {exc}")
            traceback.print_exc()

        # 3. Sparse retrieval.
        try:
            sparse = search_clinical_chunks_sparse(
                db=db,
                query=QUERY,
                patient_id=PATIENT_ID,
                top_k=TOP_K,
                case_id=CASE_ID,
            )

            validate_results(
                "Sparse retrieval", sparse, PATIENT_ID, CASE_ID
            )
            validate_scores(
                "Sparse retrieval",
                sparse,
                "sparse_score",
                descending=True,
            )
            print_results("SPARSE RETRIEVAL", sparse, "sparse_score")
            passed.append("Sparse retrieval")
            print("[PASS] Sparse retrieval")
        except Exception as exc:
            failed.append("Sparse retrieval")
            print(f"\n[FAIL] Sparse retrieval: {exc}")
            traceback.print_exc()

        # 4. Hybrid retrieval with RRF.
        try:
            hybrid = search_clinical_chunks_hybrid(
                db=db,
                query=QUERY,
                patient_id=PATIENT_ID,
                top_k=TOP_K,
                case_id=CASE_ID,
                candidate_k=CANDIDATE_K,
            )

            validate_results(
                "Hybrid retrieval", hybrid, PATIENT_ID, CASE_ID
            )
            validate_scores(
                "Hybrid retrieval",
                hybrid,
                "rrf_score",
                descending=True,
            )

            # Hybrid results should not exceed the requested final count.
            assert len(hybrid) <= TOP_K, (
                f"Hybrid retrieval returned more than {TOP_K} results"
            )

            print_results("HYBRID RETRIEVAL", hybrid, "rrf_score")
            passed.append("Hybrid retrieval")
            print("[PASS] Hybrid retrieval")
        except Exception as exc:
            failed.append("Hybrid retrieval")
            print(f"\n[FAIL] Hybrid retrieval: {exc}")
            traceback.print_exc()

        # 5. Validate hybrid argument checking without a DB query.
        try:
            invalid_inputs = [
                {"query": "", "patient_id": PATIENT_ID},
                {"query": QUERY, "patient_id": 0},
                {"query": QUERY, "patient_id": PATIENT_ID,
                 "top_k": 0},
                {"query": QUERY, "patient_id": PATIENT_ID,
                 "candidate_k": 0},
            ]

            for params in invalid_inputs:
                arguments = {
                    "db": db,
                    "query": QUERY,
                    "patient_id": PATIENT_ID,
                    "top_k": TOP_K,
                    "case_id": CASE_ID,
                    "candidate_k": CANDIDATE_K,
                }
                arguments.update(params)

                try:
                    search_clinical_chunks_hybrid(**arguments)
                except (TypeError, ValueError):
                    continue
                else:
                    raise AssertionError(
                        f"Invalid input was accepted: {params}"
                    )

            passed.append("Hybrid input validation")
            print("\n[PASS] Hybrid input validation")
        except Exception as exc:
            failed.append("Hybrid input validation")
            print(f"\n[FAIL] Hybrid input validation: {exc}")
            traceback.print_exc()

    except Exception as exc:
        failed.append("Test setup / database connection")
        print(f"\n[FATAL] Test setup failed: {exc}")
        traceback.print_exc()

    finally:
        if db is not None:
            try:
                db.close()
                print("\nDatabase session closed.")
            except Exception:
                logger.exception("Failed to close database session")

    print("\n" + "=" * 72)
    print("FINAL TEST SUMMARY")
    print("=" * 72)
    print(f"Passed ({len(passed)}): {', '.join(passed) or 'None'}")
    print(f"Failed ({len(failed)}): {', '.join(failed) or 'None'}")

    if failed:
        print("\nRESULT: FAILED — inspect the errors above.")
        sys.exit(1)

    print("\nRESULT: ALL TESTS PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
