
"""End-to-end RAG orchestration for MedFlow AI."""

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.services.retrieval.hybrid_retrieval_service import (
    search_clinical_chunks_hybrid,
)
from app.services.retrieval.reranker_service import (
    rerank_clinical_chunks,
)
from app.services.retrieval.context_assembly_service import (
    assemble_clinical_context,
)
from app.services.generation.gemini_generation_service import (
    generate_clinical_answer,
)

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_K = 20
DEFAULT_RERANK_TOP_K = 5
DEFAULT_CONTEXT_TOKEN_BUDGET = 6000


def answer_clinical_question(
    db: Session,
    query: str,
    patient_id: int,
    case_id: str | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Run hybrid retrieval, reranking, context assembly, and generation.

    Authentication, RBAC, and patient/case authorization must be
    enforced by the calling API before this function is invoked.

    The caller owns the database session and is responsible for
    closing it.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if (
        isinstance(patient_id, bool)
        or not isinstance(patient_id, int)
        or patient_id <= 0
    ):
        raise ValueError("patient_id must be a positive integer")

    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string or None")
        case_id = case_id.strip()

    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= 50
    ):
        raise ValueError("candidate_k must be between 1 and 50")

    if (
        isinstance(rerank_top_k, bool)
        or not isinstance(rerank_top_k, int)
        or not 1 <= rerank_top_k <= 50
    ):
        raise ValueError("rerank_top_k must be between 1 and 50")

    if (
        isinstance(context_token_budget, bool)
        or not isinstance(context_token_budget, int)
        or not 1 <= context_token_budget <= 20000
    ):
        raise ValueError(
            "context_token_budget must be between 1 and 20000"
        )

    query = query.strip()

    logger.info(
        "Clinical RAG pipeline started",
        extra={
            "patient_id": patient_id,
            "case_id": case_id,
            "candidate_k": candidate_k,
            "rerank_top_k": rerank_top_k,
        },
    )

    try:
        # 1. Retrieve and fuse dense + sparse results using RRF.
        hybrid_candidates = search_clinical_chunks_hybrid(
            db=db,
            query=query,
            patient_id=patient_id,
            case_id=case_id,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        # 2. Rerank the candidate pool with the cross-encoder.
        # An empty candidate pool is valid and must not load the model.
        if hybrid_candidates:
            reranked_candidates = rerank_clinical_chunks(
                query=query,
                candidates=hybrid_candidates,
                top_k=min(rerank_top_k, len(hybrid_candidates)),
            )
        else:
            reranked_candidates = []

        # 3. Assemble bounded, traceable clinical evidence.
        assembled_context = assemble_clinical_context(
            query=query,
            candidates=reranked_candidates,
            patient_id=patient_id,
            case_id=case_id,
            token_budget=context_token_budget,
        )

        # 4. Generate an answer grounded in the assembled evidence.
        # The generation service handles the no-evidence case without
        # making a Gemini request.
        generated_answer = generate_clinical_answer(
            assembled_context
        )

        result = {
            "patient_id": patient_id,
            "case_id": case_id,
            "query": query,
            "retrieval_count": len(hybrid_candidates),
            "reranked_count": len(reranked_candidates),
            "context": assembled_context,
            "generation": generated_answer,
            "needs_human_review": True,
        }

        logger.info(
            "Clinical RAG pipeline completed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
                "retrieval_count": len(hybrid_candidates),
                "reranked_count": len(reranked_candidates),
                "evidence_count": assembled_context[
                    "included_chunk_count"
                ],
                "insufficient_evidence": generated_answer[
                    "insufficient_evidence"
                ],
            },
        )

        return result

    except Exception:
        logger.exception(
            "Clinical RAG pipeline failed",
            extra={
                "patient_id": patient_id,
                "case_id": case_id,
            },
        )
        raise
