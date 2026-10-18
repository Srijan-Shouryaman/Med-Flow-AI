
"""Hybrid clinical retrieval using dense and sparse search with RRF."""

import logging
import math
from typing import Any

from sqlalchemy.orm import Session


from app.services.retrieval.dense_retrieval_service import (
    MAX_TOP_K,
    search_clinical_chunks,
)
from app.services.retrieval.sparse_retrieval_service import (
    search_clinical_chunks_sparse,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
RRF_K = 60


def search_clinical_chunks_hybrid(
    db: Session,
    query: str,
    patient_id: int,
    top_k: int = DEFAULT_TOP_K,
    case_id: str | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict[str, Any]]:
    """Combine dense and sparse retrieval using Reciprocal Rank Fusion.

    Retrieves candidate_k chunks from each retrieval method, fuses and
    deduplicates them using RRF, then returns the best top_k candidates.

    Patient filtering is applied by both retrieval services. The API layer
    must independently authorize access to the requested patient and case.

    Args:
        db: Active SQLAlchemy session.
        query: Non-empty natural-language query.
        patient_id: Positive patient identifier.
        top_k: Number of final results to return.
        case_id: Optional case identifier.
        candidate_k: Number of candidates requested from each retrieval
            method. Must not exceed MAX_TOP_K.

    Returns:
        RRF-ranked chunks with dense/sparse ranks and scores.
        rrf_score is a ranking score, not a probability.
    """
    # Validate all arguments before invoking either retrieval service.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string")

    if isinstance(patient_id, bool) or not isinstance(patient_id, int):
        raise TypeError("patient_id must be an integer")

    if patient_id <= 0:
        raise ValueError("patient_id must be greater than zero")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise TypeError("candidate_k must be an integer")

    if not 1 <= candidate_k <= MAX_TOP_K:
        raise ValueError(
            f"candidate_k must be between 1 and {MAX_TOP_K}"
        )

    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string or None")
        case_id = case_id.strip()

    query = query.strip()

    try:
        # Retrieve a wider candidate pool than the final result count.
        dense_results = search_clinical_chunks(
            db=db,
            query=query,
            patient_id=patient_id,
            top_k=candidate_k,
            case_id=case_id,
        )

        sparse_results = search_clinical_chunks_sparse(
            db=db,
            query=query,
            patient_id=patient_id,
            top_k=candidate_k,
            case_id=case_id,
        )

        # Merge chunks by stable chunk_id.
        fused: dict[str, dict[str, Any]] = {}

        def add_ranked_results(
            retrieval_results: list[dict[str, Any]],
            rank_field: str,
            score_field: str,
            source_score_field: str,
        ) -> None:
            for rank, result in enumerate(retrieval_results, start=1):
                chunk_id = result.get("chunk_id")

                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise RuntimeError(
                        f"{rank_field} result has a missing or invalid "
                        "chunk_id"
                    )

                if result.get("patient_id") != patient_id:
                    raise RuntimeError(
                        "Retrieval returned a chunk for an unexpected patient"
                    )

                if (
                    case_id is not None
                    and result.get("case_id") != case_id
                ):
                    raise RuntimeError(
                        "Retrieval returned a chunk for an unexpected case"
                    )

                if chunk_id not in fused:
                    fused[chunk_id] = dict(result)
                    fused[chunk_id]["dense_rank"] = None
                    fused[chunk_id]["sparse_rank"] = None
                    fused[chunk_id]["dense_similarity_score"] = None
                    fused[chunk_id]["sparse_score"] = None
                    fused[chunk_id]["rrf_score"] = 0.0

                item = fused[chunk_id]

                # Preserve the original ranking and score from each source.
                item[rank_field] = rank

                try:
                    source_score = float(result[source_score_field])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"Invalid {source_score_field} for chunk {chunk_id}"
                    ) from exc

                if not math.isfinite(source_score):
                    raise RuntimeError(
                        f"Non-finite {source_score_field} for chunk {chunk_id}"
                    )

                item[score_field] = source_score

                # Reciprocal Rank Fusion contribution.
                item["rrf_score"] += 1.0 / (RRF_K + rank)

        add_ranked_results(
            retrieval_results=dense_results,
            rank_field="dense_rank",
            score_field="dense_similarity_score",
            source_score_field="similarity_score",
        )

        add_ranked_results(
            retrieval_results=sparse_results,
            rank_field="sparse_rank",
            score_field="sparse_score",
            source_score_field="sparse_score",
        )

        candidates = list(fused.values())

        for item in candidates:
            score = float(item["rrf_score"])

            if not math.isfinite(score) or score <= 0:
                raise RuntimeError("Invalid RRF score produced")

        # Deterministic tie-breaking makes results reproducible.
        candidates.sort(
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"],
            )
        )

        final_results = candidates[:top_k]

        logger.info(
            "Hybrid clinical retrieval completed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "candidate_k": candidate_k,
                "dense_result_count": len(dense_results),
                "sparse_result_count": len(sparse_results),
                "unique_candidate_count": len(candidates),
                "final_result_count": len(final_results),
                "top_k": top_k,
            },
        )

        return final_results

    except (TypeError, ValueError):
        raise
    except Exception:
        logger.exception(
            "Hybrid clinical retrieval failed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "candidate_k": candidate_k,
                "top_k": top_k,
            },
        )
        raise
