from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.auth.dependencies import require_role

from app.ml.inference.prediction_service import predict_department

import json

from app.schemas.assignment import ApprovalRequest,OverrideRequest
from app.schemas.case import CaseStatusUpdate

from app.services.recommendation.recommendation_service import generate_recommendations
from app.services.assignment.assignment_service import (
    CaseNotFoundError,
    CaseStatusConflictError,
    approve_assignment,
    override_assignment,
    update_case_status,
)


router = APIRouter(
    prefix="/api/cases",
    tags=["Cases"]
)

case_status_user = require_role(
    "SUPER_ADMIN",
    "HOSPITAL_ADMIN",
    "DEPARTMENT_HEAD",
    "DOCTOR",
)


@router.get("/")
def get_cases(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            case_id,
            patient_ref_id,
            report_summary,
            is_emergency,
            priority,
            submitted_at,
            status
        FROM cases
        WHERE status != 'Assigned'
        ORDER BY submitted_at DESC
    """))

    cases = result.mappings().all()

    return cases

@router.get("/history")
def get_case_history(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            case_id,
            patient_ref_id,
            report_summary,
            is_emergency,
            priority,
            submitted_at,
            status
        FROM cases
        WHERE status = 'Assigned'
        ORDER BY submitted_at DESC
    """))

    cases = result.mappings().all()

    return cases

@router.post("/{case_id}/prediction")
def predict_case(
    case_id: str,
    db: Session = Depends(get_db)
):
    case_result = db.execute(
        text("""
            SELECT
                case_id,
                report_id,
                report_summary,
                status
            FROM cases
            WHERE case_id = :case_id
        """),
        {
            "case_id": case_id
        }
    )

    case = case_result.mappings().fetchone()

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    if case["status"] != "Pending Prediction":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Case cannot be predicted because its current "
                f"status is '{case['status']}'"
            )
        )

    prediction_text = case["report_summary"]

    if not prediction_text or not prediction_text.strip():
        raise HTTPException(
            status_code=409,
            detail="Case does not contain usable report text"
        )

    result = predict_department(prediction_text)

    department_result = db.execute(
        text("""
            SELECT department_id
            FROM departments
            WHERE department_name = :department_name
        """),
        {
            "department_name": result["predicted_department"]
        }
    )

    department = department_result.fetchone()

    if department is None:
        raise HTTPException(
            status_code=500,
            detail="Predicted department not found"
        )

    predicted_department_id = department[0]

    prediction_result = db.execute(
        text("""
            INSERT INTO predictions (
                case_id,
                predicted_department_id,
                decision_function_scores,
                confidence_score,
                confidence_level_label,
                model_version
            )
            VALUES (
                :case_id,
                :predicted_department_id,
                CAST(:decision_function_scores AS JSONB),
                :confidence_score,
                :confidence_level_label,
                :model_version
            )
            RETURNING prediction_id
        """),
        {
            "case_id": case_id,
            "predicted_department_id": predicted_department_id,
            "decision_function_scores": json.dumps(
                result["class_probabilities"]
            ),
            "confidence_score": result["confidence_score"],
            "confidence_level_label": result["confidence_level"],
            "model_version": "svm-tfidf-v1.2-calibrated"
        }
    )

    prediction_id = prediction_result.scalar()

    db.execute(
        text("""
            UPDATE cases
            SET status = 'Predicted'
            WHERE case_id = :case_id
        """),
        {
            "case_id": case_id
        }
    )

    db.commit()

    return {
        "prediction_id": prediction_id,
        "case_id": case_id,
        "predicted_department_id": predicted_department_id,
        "predicted_department": result["predicted_department"],
        "confidence_score": result["confidence_score"],
        "confidence_level": result["confidence_level"],
        "class_probabilities": result["class_probabilities"]
    }

@router.get("/{case_id}/prediction")
def get_case_prediction(
    case_id: str,
    db: Session = Depends(get_db)
):
    result = db.execute(text("""
        SELECT
            p.prediction_id,
            p.case_id,
            p.predicted_department_id,
            d.department_name,
            p.decision_function_scores,
            p.confidence_score,
            p.confidence_level_label,
            p.model_version
        FROM predictions p
        JOIN departments d
            ON p.predicted_department_id = d.department_id
        WHERE p.case_id = :case_id
        ORDER BY p.prediction_id DESC
        LIMIT 1
    """), {
        "case_id": case_id
    })

    prediction = result.mappings().fetchone()

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found for this case"
        )

    return prediction

@router.post("/{case_id}/recommendations")
def generate_case_recommendations(
    case_id: str,
    db: Session = Depends(get_db)
):
    try:
        result = generate_recommendations(case_id, db)

        db.commit()

        return result

    except ValueError as e:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to generate recommendations"
        )

@router.get("/{case_id}/recommendations")
def get_case_recommendations(
    case_id: str,
    db: Session = Depends(get_db)
):
    result = db.execute(text("""
        SELECT
            r.recommendation_id,
            r.case_id,
            r.prediction_id,
            r.rank,
            r.team_id,
            ct.team_name,
            ct.specialization,
            r.recommendation_score,
            r.score_breakdown
        FROM recommendations r
        LEFT JOIN clinical_teams ct
            ON r.team_id = ct.team_id
        WHERE r.case_id = :case_id
        ORDER BY r.rank ASC
    """), {
        "case_id": case_id
    })

    recommendations = result.mappings().all()

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="Recommendations not found for this case"
        )

    return recommendations

@router.get("/{case_id}/assignment")
def get_case_assignment(
    case_id: str,
    db: Session = Depends(get_db)
):
    result = db.execute(text("""
        SELECT
            a.assignment_id,
            a.case_id,
            a.ai_recommended_team_id,
            ai_team.team_name AS ai_recommended_team_name,
            a.ai_recommendation_score,
            a.final_assigned_team_id,
            final_team.team_name AS final_assigned_team_name,
            a.human_decision,
            a.override_reason,
            a.approving_user
        FROM assignments a
        LEFT JOIN clinical_teams ai_team
            ON a.ai_recommended_team_id = ai_team.team_id
        LEFT JOIN clinical_teams final_team
            ON a.final_assigned_team_id = final_team.team_id
        WHERE a.case_id = :case_id
        ORDER BY a.assignment_id DESC
        LIMIT 1
    """), {
        "case_id": case_id
    })

    assignment = result.mappings().fetchone()

    if assignment is None:
        raise HTTPException(
            status_code=404,
            detail="Assignment not found for this case"
        )

    return assignment

@router.post("/{case_id}/assignment/approve")
def approve_case_assignment(
    case_id: str,
    request: ApprovalRequest,
    db: Session = Depends(get_db)
):
    try:
        result = approve_assignment(
            case_id,
            request.approving_user,
            db
        )

        db.commit()

        return result

    except ValueError as e:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to approve assignment"
        )

@router.post("/{case_id}/assignment/override")
def override_case_assignment(
    case_id: str,
    request: OverrideRequest,
    db: Session = Depends(get_db)
):
    try:
        result = override_assignment(
            case_id,
            request.selected_team_id,
            request.override_reason,
            request.approving_user,
            db
        )

        db.commit()

        return result

    except ValueError as e:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to override assignment"
        )


@router.get("/{case_id}")
def get_case(case_id: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            case_id,
            patient_ref_id,
            report_summary,
            is_emergency,
            priority,
            submitted_at,
            status
        FROM cases
        WHERE case_id = :case_id
    """), {
        "case_id": case_id
    })

    case = result.mappings().fetchone()

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    return case


@router.patch("/{case_id}/status")
def patch_case_status(
    case_id: str,
    request: CaseStatusUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(case_status_user),
):
    actor = current_user.get("full_name") or current_user.get("email")

    try:
        result = update_case_status(
            case_id,
            request.status,
            actor,
            db,
        )
        db.commit()
        return result
    except CaseNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CaseStatusConflictError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to update case status",
        ) from exc
