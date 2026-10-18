
"""Dense clinical retrieval using MedCPT and PostgreSQL pgvector."""

import logging
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.embedding.medcpt_embedding_service import (
    EMBEDDING_DIMENSION,
    embedding_service,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 50

DENSE_SEARCH_SQL = text(
    """
    SELECT
        id,
        chunk_id,
        parent_chunk_id,
        patient_id,
        report_id,
        case_id,
        extraction_id,
        chunk_text,
        metadata,
        1 - (embedding <=> CAST(:query_embedding AS vector))
            AS similarity_score
    FROM public.clinical_chunks
    WHERE patient_id = :patient_id
      AND (
          CAST(:case_id AS varchar) IS NULL
          OR case_id = CAST(:case_id AS varchar)
      )
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_k
    """
)


def search_clinical_chunks(
    db: Session,
    query: str,
    patient_id: int,
    top_k: int = DEFAULT_TOP_K,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most similar clinical chunks for a required patient.

    Args:
        db: Active SQLAlchemy session.
        query: Non-empty natural-language query.
        patient_id: Positive patient identifier.
        top_k: Number of results, from 1 through MAX_TOP_K.
        case_id: Optional case identifier for additional filtering.

    Returns:
        Chunks ordered by pgvector cosine similarity. Each result includes
        similarity_score, which is a similarity value, not a probability.

    Security:
        Patient/case filters are enforced in SQL. The API layer must still
        authorize access to the requested patient and case.
    """
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

    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string or None")
        case_id = case_id.strip()

    query = query.strip()

    try:
        query_embedding = embedding_service.encode_query(query)

        if query_embedding is None:
            raise RuntimeError("MedCPT returned no query embedding")

        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Expected a {EMBEDDING_DIMENSION}-dimensional query "
                f"embedding, got {len(query_embedding)} dimensions"
            )

        # Convert values to floats and reject malformed/non-finite vectors
        # before passing them to PostgreSQL.
        try:
            query_embedding = [float(value) for value in query_embedding]
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "MedCPT returned an invalid query embedding"
            ) from exc

        if not all(math.isfinite(value) for value in query_embedding):
            raise RuntimeError(
                "MedCPT query embedding contains non-finite values"
            )

        result = db.execute(
            DENSE_SEARCH_SQL,
            {
                "query_embedding": str(query_embedding),
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
            },
        )

        rows = result.mappings().all()
        matches: list[dict[str, Any]] = []

        for row in rows:
            item = dict(row)

            if not item.get("chunk_id"):
                raise RuntimeError(
                    "PostgreSQL returned a chunk without a chunk_id"
                )

            try:
                score = float(item["similarity_score"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    "PostgreSQL returned a missing or invalid similarity score"
                ) from exc

            if not math.isfinite(score):
                raise RuntimeError(
                    "Database returned a non-finite similarity score"
                )

            item["similarity_score"] = score
            matches.append(item)

        logger.info(
            "Dense clinical retrieval completed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
                "result_count": len(matches),
                "embedding_dimension": len(query_embedding),
            },
        )

        return matches

    except (TypeError, ValueError):
        raise
    except Exception:
        logger.exception(
            "Dense clinical retrieval failed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
            },
        )
        raise
