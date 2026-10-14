import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.auth.dependencies import get_current_user
from app.api.routes.reports import upload_report
from app.api.routes.cases import predict_case
from app.schemas.clinical_entities import CombinedClinicalExtraction
from app.services.clinical_nlp.extraction_service import (
    extract_clinical_entities,
    extract_department_specific_entities,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/clinical-nlp",
    tags=["Clinical NLP"],
)


def _to_jsonable(value: Any) -> Any:
    """Convert Pydantic models and supported values to JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict"):
        return value.dict()

    if isinstance(value, dict):
        return {
            key: _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]

    return value


@router.post("/process")
async def process_clinical_report(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Upload a report, predict its department, and extract clinical entities."""
    try:
        if patient_id <= 0:
            raise HTTPException(
                status_code=422,
                detail="patient_id must be a positive integer",
            )

        if not file.filename:
            raise HTTPException(
                status_code=422,
                detail="A report filename is required",
            )

        user_id = current_user.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Authenticated user ID is missing",
            )

        # Step 1: Reuse the existing report-upload pipeline.
        upload_result = await upload_report(
            patient_id=patient_id,
            file=file,
            db=db,
            current_user=current_user,
        )

        report_id = upload_result.get("report_id")
        case_id = upload_result.get("case_id")

        if report_id is None or not case_id:
            raise HTTPException(
                status_code=500,
                detail="Upload did not return report_id and case_id",
            )

        # Step 2: Check whether an extraction already exists.
        existing = db.execute(
            text("""
                SELECT
                    extraction_id,
                    report_id,
                    case_id,
                    prediction_id,
                    predicted_department,
                    extraction_json,
                    extraction_status,
                    created_at,
                    updated_at
                FROM public.clinical_extractions
                WHERE report_id = :report_id
            """),
            {"report_id": report_id},
        ).mappings().fetchone()

        if existing is not None:
            return {
                "message": "Clinical extraction already exists",
                "report_id": report_id,
                "case_id": case_id,
                "extraction": dict(existing),
            }

        # Step 3: Generate or retrieve the SVM prediction.
        case_result = db.execute(
            text("""
                SELECT status
                FROM public.cases
                WHERE case_id = :case_id
            """),
            {"case_id": case_id},
        ).mappings().fetchone()

        if case_result is None:
            raise HTTPException(
                status_code=404,
                detail="The uploaded case could not be found",
            )

        if case_result["status"] == "Pending Prediction":
            prediction = predict_case(
                case_id=case_id,
                db=db,
            )

        elif case_result["status"] == "Predicted":
            prediction_result = db.execute(
                text("""
                    SELECT
                        p.prediction_id,
                        p.case_id,
                        p.predicted_department_id,
                        d.department_name AS predicted_department,
                        p.confidence_score,
                        p.confidence_level_label,
                        p.decision_function_scores,
                        p.model_version
                    FROM public.predictions AS p
                    JOIN public.departments AS d
                        ON d.department_id = p.predicted_department_id
                    WHERE p.case_id = :case_id
                    ORDER BY p.prediction_id DESC
                    LIMIT 1
                """),
                {"case_id": case_id},
            ).mappings().fetchone()

            if prediction_result is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "The case is marked Predicted, "
                        "but no prediction record exists"
                    ),
                )

            prediction = dict(prediction_result)

        else:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The case cannot be predicted in its current state: "
                    f"{case_result['status']}"
                ),
            )

        prediction_id = prediction.get("prediction_id")
        predicted_department_id = prediction.get("predicted_department_id")

        if prediction_id is None or predicted_department_id is None:
            raise HTTPException(
                status_code=500,
                detail="Prediction is missing required identifiers",
            )

        # Step 4: Read the original extracted report text.
        report_result = db.execute(
            text("""
                SELECT extracted_text
                FROM public.reports
                WHERE report_id = :report_id
            """),
            {"report_id": report_id},
        ).mappings().fetchone()

        if report_result is None:
            raise HTTPException(
                status_code=404,
                detail="The uploaded report could not be found",
            )

        report_text = report_result["extracted_text"]

        if not isinstance(report_text, str) or not report_text.strip():
            raise HTTPException(
                status_code=422,
                detail="The report contains no usable extracted text",
            )

        # Step 5: Extract common and department-specific entities.
        common_result = extract_clinical_entities(report_text)

        department_specific_result = extract_department_specific_entities(
            report_text
        )

        # Convert extraction results into JSON-compatible values.
        common_json = _to_jsonable(common_result)
        department_specific_json = _to_jsonable(
            department_specific_result
        )

        if not isinstance(common_json, dict):
            raise TypeError(
                "Common clinical extractor must return a model or dictionary"
            )

        if not isinstance(department_specific_json, dict):
            raise TypeError(
                "Department-specific extractor must return a dictionary"
            )

        # Validate the combined output using the agreed schema.
        combined_result = CombinedClinicalExtraction(
            common=common_json,
            department_specific=department_specific_json,
        )

        extraction_json = _to_jsonable(combined_result)

        # Step 6: Save the extraction.
        # predicted_department is SMALLINT, so store the department ID.
        saved_result = db.execute(
            text("""
                INSERT INTO public.clinical_extractions (
                    report_id,
                    case_id,
                    prediction_id,
                    predicted_department,
                    extraction_json,
                    extraction_status,
                    extracted_by
                )
                VALUES (
                    :report_id,
                    :case_id,
                    :prediction_id,
                    :predicted_department_id,
                    CAST(:extraction_json AS JSONB),
                    'COMPLETED',
                    :extracted_by
                )
                RETURNING
                    extraction_id,
                    report_id,
                    case_id,
                    prediction_id,
                    predicted_department,
                    extraction_json,
                    extraction_status,
                    created_at,
                    updated_at
            """),
            {
                "report_id": report_id,
                "case_id": case_id,
                "prediction_id": prediction_id,
                "predicted_department_id": predicted_department_id,
                "extraction_json": json.dumps(
                    extraction_json,
                    ensure_ascii=False,
                ),
                "extracted_by": user_id,
            },
        ).mappings().fetchone()

        if saved_result is None:
            raise RuntimeError(
                "Clinical extraction insert returned no saved record"
            )

        db.commit()

        return {
            "message": "Report processed successfully",
            "report_id": report_id,
            "case_id": case_id,
            "prediction": prediction,
            "clinical_extraction": dict(saved_result),
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception:
        db.rollback()
        logger.exception(
            "Clinical NLP processing failed for patient_id=%s, filename=%s",
            patient_id,
            file.filename,
        )
        raise HTTPException(
            status_code=500,
            detail="Clinical NLP processing failed. Check the backend logs.",
        )


@router.get("/reports/{report_id}")
def get_clinical_extraction(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve the saved clinical extraction for a report."""
    try:
        result = db.execute(
            text("""
                SELECT
                    extraction_id,
                    report_id,
                    case_id,
                    prediction_id,
                    predicted_department,
                    extraction_json,
                    extraction_status,
                    extraction_error,
                    extracted_by,
                    created_at,
                    updated_at
                FROM public.clinical_extractions
                WHERE report_id = :report_id
            """),
            {"report_id": report_id},
        ).mappings().fetchone()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Clinical extraction not found for this report",
            )

        return dict(result)

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to retrieve clinical extraction for report_id=%s",
            report_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve clinical extraction",
        )

