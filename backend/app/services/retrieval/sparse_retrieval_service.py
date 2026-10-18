
"""Sparse clinical retrieval using PostgreSQL full-text search.

Questions use OR semantics: a chunk qualifies when it contains any query
lexeme, and chunks matching more lexemes generally rank higher.

The hybrid retrieval service can request a larger candidate pool by passing
top_k=20, while existing callers retain the default top_k=5.

Patient and optional case filters are enforced in SQL. Authorization must
still be enforced by the API layer before results are exposed.
"""

import logging
import math
import re
from collections.abc import Collection
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 50

# ts_rank_cd normalization is a bit mask (0-63).
# 0: no document-length normalization
# 1: divide by 1 + log(document length)
# 2: divide by document length
# 32: rescale the score to 0..1; does not address document length.
# RRF uses ranks rather than raw sparse scores, so normalization does not
# directly change the fused ordering when the candidate order is unchanged.
DEFAULT_RANK_NORMALIZATION = 1
MAX_RANK_NORMALIZATION = 63

# Empty by default. With OR matching, common clinical terms can be useful
# retrieval signals. Add terms only after evaluating them against test queries.
DEFAULT_STOPLIST: frozenset[str] = frozenset()

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

SPARSE_SEARCH_SQL = text(
    """
    WITH q AS (
        SELECT CAST(
            replace(
                CAST(
                    plainto_tsquery('english', CAST(:query AS text))
                    AS text
                ),
                '&',
                '|'
            ) AS tsquery
        ) AS tsq
    )
    SELECT
        c.id,
        c.chunk_id,
        c.parent_chunk_id,
        c.patient_id,
        c.report_id,
        c.case_id,
        c.extraction_id,
        c.chunk_text,
        c.metadata,
        ts_rank_cd(
            c.search_vector,
            q.tsq,
            CAST(:rank_normalization AS integer)
        ) AS sparse_score
    FROM public.clinical_chunks AS c
    CROSS JOIN q
    WHERE numnode(q.tsq) > 0
      AND c.patient_id = :patient_id
      AND (
          CAST(:case_id AS varchar) IS NULL
          OR c.case_id = CAST(:case_id AS varchar)
      )
      AND c.search_vector @@ q.tsq
    ORDER BY sparse_score DESC, c.id ASC
    LIMIT :top_k
    """
)


def apply_stoplist(
    query: str,
    stoplist: Collection[str] = DEFAULT_STOPLIST,
) -> str:
    """Remove configured words from a query.

    Matching is case-insensitive against raw words. If the stoplist is empty,
    the original query is returned unchanged. If every word is removed, an
    empty string is returned.
    """
    if not stoplist:
        return query

    blocked = {word.lower() for word in stoplist}
    kept = [
        token
        for token in _TOKEN_RE.findall(query)
        if token.lower() not in blocked
    ]
    return " ".join(kept)


def search_clinical_chunks_sparse(
    db: Session,
    query: str,
    patient_id: int,
    top_k: int = DEFAULT_TOP_K,
    case_id: str | None = None,
    rank_normalization: int = DEFAULT_RANK_NORMALIZATION,
    stoplist: Collection[str] = DEFAULT_STOPLIST,
) -> list[dict[str, Any]]:
    """Return lexically matching clinical chunks using OR semantics.

    Args:
        db: Active SQLAlchemy session.
        query: Non-empty natural-language search query.
        patient_id: Positive patient identifier.
        top_k: Number of candidates to retrieve, from 1 through 50.
        case_id: Optional case identifier to restrict the search.
        rank_normalization: PostgreSQL ts_rank_cd normalization bit mask.
        stoplist: Optional collection of words to remove from the query.

    Returns:
        Matching chunks containing their database fields and sparse_score.
        Returns an empty list when no searchable lexemes remain or no chunks
        match the query.

    Raises:
        TypeError: When an argument has an invalid type.
        ValueError: When an argument has an invalid value.
        RuntimeError: When PostgreSQL returns an invalid score.
        SQLAlchemy or other database exceptions are logged and re-raised.

    Security:
        SQL filters restrict results to patient_id and optional case_id, but
        they do not authorize the caller to access that patient's data.
        Enforce access control at the API/service authorization boundary.
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

    if isinstance(rank_normalization, bool) or not isinstance(
        rank_normalization, int
    ):
        raise TypeError("rank_normalization must be an integer")

    if not 0 <= rank_normalization <= MAX_RANK_NORMALIZATION:
        raise ValueError(
            "rank_normalization must be between "
            f"0 and {MAX_RANK_NORMALIZATION}"
        )

    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string or None")
        case_id = case_id.strip()

    if isinstance(stoplist, (str, bytes)) or not isinstance(
        stoplist, Collection
    ):
        raise TypeError("stoplist must be a collection of strings")

    if any(not isinstance(word, str) for word in stoplist):
        raise TypeError("every stoplist entry must be a string")

    query = apply_stoplist(query.strip(), stoplist).strip()

    if not query:
        logger.info(
            "Sparse clinical retrieval skipped: query empty after stoplist",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
            },
        )
        return []

    try:
        result = db.execute(
            SPARSE_SEARCH_SQL,
            {
                "query": query,
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
                "rank_normalization": rank_normalization,
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
                score = float(item["sparse_score"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    "PostgreSQL returned a missing or invalid sparse score"
                ) from exc

            if not math.isfinite(score) or score < 0:
                raise RuntimeError(
                    "PostgreSQL returned an invalid sparse search score"
                )

            item["sparse_score"] = score
            matches.append(item)

        logger.info(
            "Sparse clinical retrieval completed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "result_count": len(matches),
                "top_k": top_k,
                "rank_normalization": rank_normalization,
            },
        )

        return matches

    except (TypeError, ValueError):
        raise
    except Exception:
        logger.exception(
            "Sparse clinical retrieval failed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "top_k": top_k,
                "rank_normalization": rank_normalization,
            },
        )
        raise
