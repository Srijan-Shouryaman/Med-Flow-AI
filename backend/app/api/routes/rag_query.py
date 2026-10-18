
"""API route for patient-scoped clinical RAG queries."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.auth.dependencies import get_current_user
from app.schemas.rag_query import RAGQueryRequest, RAGQueryResponse
from app.services.rag.rag_pipeline_service import answer_clinical_question
from app.services.generation.gemini_generation_service import (
    GenerationServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/rag",
    tags=["Clinical RAG"],
)

ALLOWED_ROLES = {
    "DOCTOR",
    "DEPARTMENT_HEAD",
    "HOSPITAL_ADMIN",
    "SUPER_ADMIN",
}

HOSPITAL_WIDE_ROLES = {
    "HOSPITAL_ADMIN",
    "SUPER_ADMIN",
}


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
)
def query_clinical_records(
    payload: RAGQueryRequest,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Query clinical evidence for a patient and optionally a specific case."""

    user_id = current_user["user_id"]
    role = current_user["role"]

    # 1. Verify that the user is permitted to use the RAG endpoint.
    if role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to use clinical RAG queries.",
        )

    # 2. Validate the question.
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be blank.",
        )

    # 3. Authorization note:
    # Hospital administrators have hospital-wide access.
    # Doctors and department heads must pass a patient-specific
    # department-access check before retrieval is enabled for clinical use.
    #
    # Do not treat a supplied patient_id as proof of authorization.
    # The department-level patient access check must be enforced here
    # using the application's established patient-access rules.

    logger.info(
        "Clinical RAG query received",
        extra={
            "patient_id": payload.patient_id,
            "case_id": payload.case_id,
            "user_id": user_id,
            "role": role,
        },
    )

    try:
        # 4. Run the existing hybrid retrieval, reranking,
        # context assembly, and generation pipeline.
        result = answer_clinical_question(
            db=db,
            query=question,
            patient_id=payload.patient_id,
            case_id=payload.case_id,
        )

        logger.info(
            "Clinical RAG query completed",
            extra={
                "patient_id": payload.patient_id,
                "case_id": payload.case_id,
                "user_id": user_id,
                "retrieval_count": result["retrieval_count"],
                "reranked_count": result["reranked_count"],
            },
        )

        return result

    except HTTPException:
        # Preserve intentional HTTP errors without masking their status.
        raise

    except ValueError as exc:
        logger.warning(
            "Clinical RAG request validation failed",
            extra={
                "patient_id": payload.patient_id,
                "user_id": user_id,
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except GenerationServiceError as exc:
        # For example, Gemini can return 503 when the model is overloaded.
        logger.warning(
            "Clinical RAG generation service failed",
            extra={
                "patient_id": payload.patient_id,
                "case_id": payload.case_id,
                "user_id": user_id,
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The clinical answer generation service is temporarily "
                "unavailable. Please retry shortly."
            ),
            headers={"Retry-After": "30"},
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected clinical RAG query failure",
            extra={
                "patient_id": payload.patient_id,
                "case_id": payload.case_id,
                "user_id": user_id,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clinical query failed. Please try again later.",
        ) from exc
