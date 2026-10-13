from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.auth.dependencies import get_current_user
from app.schemas.report import ReportResponse
from app.services.report.report_service import (
    save_uploaded_file,
    extract_report_text,
)


router = APIRouter(
    prefix="/api/reports",
    tags=["Reports"]
)


@router.post(
    "/upload",
    response_model=ReportResponse,
    status_code=201
)
async def upload_report(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    patient = db.execute(
        text("""
            SELECT patient_id, patient_ref_id
            FROM patients
            WHERE patient_id = :patient_id
        """),
        {"patient_id": patient_id}
    ).mappings().fetchone()

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    file_data = await save_uploaded_file(file)

    result = db.execute(
        text("""
            INSERT INTO reports (
                patient_id, original_filename, stored_filename,
                file_path, content_type, file_size, sha256_hash,
                extracted_text, extraction_status, extraction_error,
                uploaded_by
            )
            VALUES (
                :patient_id, :original_filename, :stored_filename,
                :file_path, :content_type, :file_size, :sha256_hash,
                NULL, 'PENDING', NULL, :uploaded_by
            )
            RETURNING
                report_id, patient_id, original_filename,
                stored_filename, file_path, content_type,
                file_size, sha256_hash, extracted_text,
                extraction_status, extraction_error,
                uploaded_by, uploaded_at
        """),
        {
            "patient_id": patient_id,
            "original_filename": file_data["original_filename"],
            "stored_filename": file_data["stored_filename"],
            "file_path": file_data["file_path"],
            "content_type": file_data["content_type"],
            "file_size": file_data["file_size"],
            "sha256_hash": file_data["sha256_hash"],
            "uploaded_by": current_user["user_id"],
        }
    )

    report = result.mappings().fetchone()
    report_id = report["report_id"]
    db.commit()

    case_id = None

    try:
        extraction_result = extract_report_text(
            file_data["file_path"]
        )

        extracted_text = extraction_result["extracted_text"]
        ml_text = extraction_result["ml_text"]

        if not extracted_text:
            raise ValueError(
                "OCR completed but no text was extracted"
            )

        if not ml_text:
            raise ValueError(
                "ML text preparation produced no usable text"
            )

        result = db.execute(
            text("""
                UPDATE reports
                SET extracted_text = :extracted_text,
                    extraction_status = 'EXTRACTED',
                    extraction_error = NULL
                WHERE report_id = :report_id
                RETURNING
                    report_id, patient_id, original_filename,
                    stored_filename, file_path, content_type,
                    file_size, sha256_hash, extracted_text,
                    extraction_status, extraction_error,
                    uploaded_by, uploaded_at
            """),
            {
                "extracted_text": extracted_text,
                "report_id": report_id,
            }
        )

        report = result.mappings().fetchone()

        next_case_number = db.execute(
            text("""
                SELECT COALESCE(
                    MAX(
                        CAST(SUBSTRING(case_id FROM 2) AS INTEGER)
                    ),
                    0
                ) + 1
                FROM cases
                WHERE case_id ~ '^C[0-9]+$'
            """)
        ).scalar_one()

        case_id = f"C{next_case_number:04d}"

        db.execute(
            text("""
                INSERT INTO cases (
                    case_id, patient_ref_id, report_summary, report_id
                )
                VALUES (
                    :case_id, :patient_ref_id, :report_summary, :report_id
                )
            """),
            {
                "case_id": case_id,
                "patient_ref_id": patient["patient_ref_id"],
                "report_summary": ml_text,
                "report_id": report_id,
            }
        )

        db.commit()

    except Exception as exc:
        db.rollback()

        import logging
        logger = logging.getLogger(__name__)
        logger.exception(
            "Report processing failed for report_id=%s",
            report_id
        )

        try:
            result = db.execute(
                text("""
                    UPDATE reports
                    SET extraction_status = 'FAILED',
                        extraction_error = :extraction_error
                    WHERE report_id = :report_id
                    RETURNING
                        report_id, patient_id, original_filename,
                        stored_filename, file_path, content_type,
                        file_size, sha256_hash, extracted_text,
                        extraction_status, extraction_error,
                        uploaded_by, uploaded_at
                """),
                {
                    "extraction_error": str(exc)[:2000],
                    "report_id": report_id,
                }
            )

            report = result.mappings().fetchone()
            db.commit()

        except Exception:
            db.rollback()
            logger.exception(
                "Failed to save extraction failure status for report_id=%s",
                report_id
            )
            raise HTTPException(
                status_code=500,
                detail="Report processing failed"
            )

    response = dict(report)
    response["case_id"] = case_id

    return response

@router.get("/{report_id}")
def get_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = db.execute(
        text("""
            SELECT
                r.report_id,
                r.patient_id,
                p.patient_ref_id,
                p.full_name AS patient_name,
                c.case_id,
                r.original_filename,
                r.stored_filename,
                r.file_path,
                r.content_type,
                r.file_size,
                r.sha256_hash,
                r.extracted_text,
                r.extraction_status,
                r.extraction_error,
                r.uploaded_by,
                r.uploaded_at
            FROM reports r
            INNER JOIN patients p
                ON p.patient_id = r.patient_id
            LEFT JOIN cases c
                ON c.report_id = r.report_id
            WHERE r.report_id = :report_id
        """),
        {
            "report_id": report_id
        }
    )

    report = result.mappings().fetchone()

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return dict(report)

@router.get("/")
def get_reports(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = db.execute(
        text("""
            SELECT
                r.report_id,
                r.patient_id,
                p.patient_ref_id,
                p.full_name AS patient_name,
                c.case_id,
                r.original_filename,
                r.stored_filename,
                r.content_type,
                r.file_size,
                r.sha256_hash,
                r.extraction_status,
                r.extraction_error,
                r.uploaded_by,
                r.uploaded_at
            FROM reports r
            INNER JOIN patients p
                ON p.patient_id = r.patient_id
            LEFT JOIN cases c
                ON c.report_id = r.report_id
            ORDER BY
                r.uploaded_at DESC,
                r.report_id DESC
        """)
    )

    return [
        dict(row)
        for row in result.mappings().all()
    ]

