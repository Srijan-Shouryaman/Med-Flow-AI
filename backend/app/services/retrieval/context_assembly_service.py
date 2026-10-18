
"""Context assembly for MedFlow AI clinical RAG."""

import hashlib
import logging
import math
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKEN_BUDGET = 6000
MAX_CONTEXT_TOKEN_BUDGET = 20000
MAX_CANDIDATES = 50
MAX_CHUNK_TEXT_LENGTH = 12000

# Conservative fallback estimate, not an exact tokenizer count.
DEFAULT_CHARS_PER_TOKEN = 4.0


def _validate_positive_int(value: int, name: str, maximum: int) -> None:
    """Validate an integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _normalize_text(text: str) -> str:
    """Normalize whitespace for duplicate detection only."""
    return re.sub(r"\s+", " ", text).strip()


def _estimate_tokens(text: str) -> int:
    """Conservative character-based token estimate.

    Replace with Gemini's token-counting API before production use.
    """
    if not text:
        return 0

    return math.ceil(len(text) / DEFAULT_CHARS_PER_TOKEN)


def _get_token_count(
    text: str,
    token_counter: Callable[[str], int] | None,
) -> int:
    """Count tokens using an injected counter or the fallback estimate."""
    if token_counter is None:
        return _estimate_tokens(text)

    count = token_counter(text)

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("token_counter must return an integer")

    if count < 0:
        raise ValueError("token_counter returned a negative count")

    return count


def assemble_clinical_context(
    query: str,
    candidates: list[dict[str, Any]],
    patient_id: int,
    case_id: str | None = None,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Assemble ranked clinical evidence into bounded, traceable context.

    Args:
        query: The clinical question used during retrieval.
        candidates: Output from rerank_clinical_chunks().
        patient_id: Expected patient identifier.
        case_id: Expected case identifier, or None for patient-wide scope.
        token_budget: Maximum budget for assembled evidence and headings.
        token_counter: Optional callable returning an exact token count.

    Returns:
        Structured context, source provenance, and inclusion statistics.

    Security:
        The caller must authorize patient/case access before retrieval.
        This function additionally rejects candidate scope mismatches.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    query = query.strip()

    if isinstance(patient_id, bool) or not isinstance(patient_id, int):
        raise TypeError("patient_id must be an integer")

    if patient_id <= 0:
        raise ValueError("patient_id must be positive")

    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string or None")
        case_id = case_id.strip()

    _validate_positive_int(
        token_budget,
        "token_budget",
        MAX_CONTEXT_TOKEN_BUDGET,
    )

    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")

    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"candidates must not exceed {MAX_CANDIDATES} items"
        )

    if token_counter is not None and not callable(token_counter):
        raise TypeError("token_counter must be callable")

    # Validate all candidates before assembling any context.
    prepared: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    seen_content_hashes: set[str] = set()

    duplicate_count = 0

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"Candidate {index} must be a dictionary")

        chunk_id = candidate.get("chunk_id")
        chunk_text = candidate.get("chunk_text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Candidate {index} has an invalid chunk_id"
            )

        chunk_id = chunk_id.strip()

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")

        seen_chunk_ids.add(chunk_id)

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(
                f"Candidate {chunk_id} has empty or invalid chunk_text"
            )

        chunk_text = chunk_text.strip()

        if len(chunk_text) > MAX_CHUNK_TEXT_LENGTH:
            raise ValueError(
                f"Candidate {chunk_id} exceeds the "
                f"{MAX_CHUNK_TEXT_LENGTH}-character limit"
            )

        candidate_patient_id = candidate.get("patient_id")

        if (
            isinstance(candidate_patient_id, bool)
            or candidate_patient_id != patient_id
        ):
            raise ValueError(
                f"Patient scope mismatch for chunk {chunk_id}"
            )

        candidate_case_id = candidate.get("case_id")

        if case_id is not None and candidate_case_id != case_id:
            raise ValueError(
                f"Case scope mismatch for chunk {chunk_id}"
            )

        report_id = candidate.get("report_id")

        if (
            isinstance(report_id, bool)
            or not isinstance(report_id, int)
            or report_id <= 0
        ):
            raise ValueError(
                f"Candidate {chunk_id} has an invalid report_id"
            )

        reranker_score = candidate.get("reranker_score")

        if (
            isinstance(reranker_score, bool)
            or not isinstance(reranker_score, (int, float))
            or not math.isfinite(float(reranker_score))
        ):
            raise ValueError(
                f"Candidate {chunk_id} has an invalid reranker_score"
            )

        reranker_rank = candidate.get("reranker_rank")

        if (
            isinstance(reranker_rank, bool)
            or not isinstance(reranker_rank, int)
            or reranker_rank < 1
        ):
            raise ValueError(
                f"Candidate {chunk_id} has an invalid reranker_rank"
            )

        # Deduplicate exact normalized content only. Different clinical
        # statements, uncertainty, negation, and contradictions are preserved.
        normalized = _normalize_text(chunk_text)
        content_hash = hashlib.sha256(
            normalized.casefold().encode("utf-8")
        ).hexdigest()

        if content_hash in seen_content_hashes:
            duplicate_count += 1
            continue

        seen_content_hashes.add(content_hash)

        prepared.append(
            {
                "chunk_id": chunk_id,
                "patient_id": patient_id,
                "case_id": candidate_case_id,
                "report_id": report_id,
                "chunk_text": chunk_text,
                "metadata": candidate.get("metadata", {}),
                "reranker_score": float(reranker_score),
                "reranker_rank": reranker_rank,
                "dense_rank": candidate.get("dense_rank"),
                "sparse_rank": candidate.get("sparse_rank"),
                "rrf_score": candidate.get("rrf_score"),
            }
        )

    # Reranker output should already be sorted. Sort defensively so the
    # service remains deterministic when called independently.
    prepared.sort(
        key=lambda item: (
            item["reranker_rank"],
            -item["reranker_score"],
            item["chunk_id"],
        )
    )

    sections: list[str] = []
    sources: list[dict[str, Any]] = []
    used_token_count = 0
    excluded_count = duplicate_count

    for item in prepared:
        heading = (
            f"[Source: {item['chunk_id']} | "
            f"Report: {item['report_id']}]"
        )

        block = f"{heading}\n{item['chunk_text']}"
        block_tokens = _get_token_count(block, token_counter)

        separator_tokens = (
            _get_token_count("\n\n", token_counter) if sections else 0
        )

        required_tokens = block_tokens + separator_tokens

        if used_token_count + required_tokens > token_budget:
            excluded_count += 1
            continue

        sections.append(block)
        used_token_count += required_tokens

        sources.append(
            {
                "chunk_id": item["chunk_id"],
                "patient_id": item["patient_id"],
                "case_id": item["case_id"],
                "report_id": item["report_id"],
                "reranker_rank": item["reranker_rank"],
                "reranker_score": item["reranker_score"],
                "dense_rank": item["dense_rank"],
                "sparse_rank": item["sparse_rank"],
                "rrf_score": item["rrf_score"],
            }
        )

    context = "\n\n".join(sections)

    result = {
        "patient_id": patient_id,
        "case_id": case_id,
        "query": query,
        "context": context,
        "sources": sources,
        "included_chunk_count": len(sources),
        "excluded_chunk_count": excluded_count,
        "duplicate_chunk_count": duplicate_count,
        "token_count": used_token_count,
        "token_budget": token_budget,
        "has_evidence": bool(sources),
    }

    logger.info(
        "Clinical context assembly completed",
        extra={
            "patient_id": patient_id,
            "case_id": case_id,
            "candidate_count": len(candidates),
            "included_chunk_count": len(sources),
            "excluded_chunk_count": excluded_count,
            "token_count": used_token_count,
            "token_budget": token_budget,
        },
    )

    return result
