import json

from sqlalchemy import text
from sqlalchemy.orm import Session


class CaseNotFoundError(ValueError):
    pass


class CaseStatusConflictError(ValueError):
    pass


def approve_assignment(case_id, approving_user, db):
    case_result = db.execute(
        text("""
            SELECT
                case_id,
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
        raise ValueError("Case not found")

    if case["status"] not in ("Recommended", "Pending Human Review"):
        raise ValueError(
            f"Case cannot be approved because its current status is '{case['status']}'"
        )

    recommendation_result = db.execute(
        text("""
            SELECT
                recommendation_id,
                prediction_id,
                team_id,
                recommendation_score
            FROM recommendations
            WHERE case_id = :case_id
              AND rank = 1
            LIMIT 1
        """),
        {
            "case_id": case_id
        }
    )

    recommendation = recommendation_result.mappings().fetchone()

    if recommendation is None:
        raise ValueError(
            "No rank-1 recommendation found for this case"
        )

    assignment_result = db.execute(
        text("""
            INSERT INTO assignments (
                case_id,
                ai_recommended_team_id,
                ai_recommendation_score,
                final_assigned_team_id,
                human_decision,
                approving_user
            )
            VALUES (
                :case_id,
                :ai_recommended_team_id,
                :ai_recommendation_score,
                :final_assigned_team_id,
                :human_decision,
                :approving_user
            )
            RETURNING assignment_id
        """),
        {
            "case_id": case_id,
            "ai_recommended_team_id": recommendation["team_id"],
            "ai_recommendation_score": recommendation["recommendation_score"],
            "final_assigned_team_id": recommendation["team_id"],
            "human_decision": "Approved AI Recommendation",
            "approving_user": approving_user
        }
    )

    assignment_id = assignment_result.scalar()

    db.execute(
        text("""
            UPDATE cases
            SET status = 'Assigned'
            WHERE case_id = :case_id
        """),
        {
            "case_id": case_id
        }
    )

    db.execute(
        text("""
            INSERT INTO audit_logs (
                case_id,
                event_type,
                actor,
                details
            )
            VALUES (
                :case_id,
                :event_type,
                :actor,
                CAST(:details AS JSONB)
            )
        """),
        {
            "case_id": case_id,
            "event_type": "ASSIGNMENT_APPROVED",
            "actor": approving_user,
            "details": json.dumps({
                "assignment_id": assignment_id,
                "recommendation_id": recommendation["recommendation_id"],
                "team_id": recommendation["team_id"],
                "decision": "Approved AI Recommendation"
            })
        }
    )

    return {
        "assignment_id": assignment_id,
        "case_id": case_id,
        "ai_recommended_team_id": recommendation["team_id"],
        "ai_recommendation_score": recommendation["recommendation_score"],
        "final_assigned_team_id": recommendation["team_id"],
        "human_decision": "Approved AI Recommendation",
        "approving_user": approving_user
    }

def override_assignment(
    case_id,
    selected_team_id,
    override_reason,
    approving_user,
    db
):
    case_result = db.execute(
        text("""
            SELECT
                case_id,
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
        raise ValueError("Case not found")

    if case["status"] not in ("Recommended", "Pending Human Review"):
        raise ValueError(
            f"Case cannot be overridden because its current status is '{case['status']}'"
        )

    if not override_reason or not override_reason.strip():
        raise ValueError("Override reason is required")

    recommendation_result = db.execute(
        text("""
            SELECT
                r.recommendation_id,
                r.prediction_id,
                r.team_id,
                r.recommendation_score,
                p.predicted_department_id
            FROM recommendations r
            JOIN predictions p
                ON r.prediction_id = p.prediction_id
            WHERE r.case_id = :case_id
              AND r.rank = 1
            LIMIT 1
        """),
        {
            "case_id": case_id
        }
    )

    recommendation = recommendation_result.mappings().fetchone()

    if recommendation is None:
        raise ValueError(
            "No rank-1 recommendation found for this case"
        )

    if selected_team_id == recommendation["team_id"]:
        raise ValueError(
            "Selected team is already the AI recommended team"
        )

    team_result = db.execute(
        text("""
            SELECT
                ct.team_id,
                ct.department_id,
                ct.status,
                ct.availability,
                ct.active_cases,
                ct.maximum_capacity
            FROM clinical_teams ct
            WHERE ct.team_id = :team_id
        """),
        {
            "team_id": selected_team_id
        }
    )

    selected_team = team_result.mappings().fetchone()

    if selected_team is None:
        raise ValueError("Selected team not found")

    if selected_team["department_id"] != recommendation["predicted_department_id"]:
        raise ValueError(
            "Selected team does not belong to the predicted department"
        )

    if selected_team["status"] == "Inactive":
        raise ValueError("Selected team is inactive")

    if selected_team["availability"] in ("Unavailable", "On Leave"):
        raise ValueError(
            "Selected team is currently unavailable"
        )

    if selected_team["active_cases"] >= selected_team["maximum_capacity"]:
        raise ValueError(
            "Selected team has reached maximum capacity"
        )

    assignment_result = db.execute(
        text("""
            INSERT INTO assignments (
                case_id,
                ai_recommended_team_id,
                ai_recommendation_score,
                final_assigned_team_id,
                human_decision,
                override_reason,
                approving_user
            )
            VALUES (
                :case_id,
                :ai_recommended_team_id,
                :ai_recommendation_score,
                :final_assigned_team_id,
                :human_decision,
                :override_reason,
                :approving_user
            )
            RETURNING assignment_id
        """),
        {
            "case_id": case_id,
            "ai_recommended_team_id": recommendation["team_id"],
            "ai_recommendation_score": recommendation["recommendation_score"],
            "final_assigned_team_id": selected_team_id,
            "human_decision": "Overridden by Department Head",
            "override_reason": override_reason,
            "approving_user": approving_user
        }
    )

    assignment_id = assignment_result.scalar()

    db.execute(
        text("""
            UPDATE cases
            SET status = 'Assigned'
            WHERE case_id = :case_id
        """),
        {
            "case_id": case_id
        }
    )

    db.execute(
        text("""
            INSERT INTO audit_logs (
                case_id,
                event_type,
                actor,
                details
            )
            VALUES (
                :case_id,
                :event_type,
                :actor,
                CAST(:details AS JSONB)
            )
        """),
        {
            "case_id": case_id,
            "event_type": "ASSIGNMENT_OVERRIDDEN",
            "actor": approving_user,
            "details": json.dumps({
                "assignment_id": assignment_id,
                "recommendation_id": recommendation["recommendation_id"],
                "ai_recommended_team_id": recommendation["team_id"],
                "final_assigned_team_id": selected_team_id,
                "decision": "Overridden by Department Head",
                "override_reason": override_reason
            })
        }
    )

    return {
        "assignment_id": assignment_id,
        "case_id": case_id,
        "ai_recommended_team_id": recommendation["team_id"],
        "ai_recommendation_score": recommendation["recommendation_score"],
        "final_assigned_team_id": selected_team_id,
        "human_decision": "Overridden by Department Head",
        "approving_user": approving_user,
        "override_reason": override_reason
    }
    
def update_case_status(
    case_id,
    new_status,
    actor,
    db
):
    case_result = db.execute(
        text("""
            SELECT
                case_id,
                status
            FROM cases
            WHERE case_id = :case_id
            FOR UPDATE
        """),
        {
            "case_id": case_id
        }
    )

    case = case_result.mappings().fetchone()

    if case is None:
        raise CaseNotFoundError("Case not found")

    current_status = case["status"]

    allowed_transitions = {
        "Assigned": {
            "In Progress"
        },
        "In Progress": {
            "Resolved"
        }
    }

    if current_status not in allowed_transitions:
        raise CaseStatusConflictError(
            f"Case cannot transition from '{current_status}'"
        )

    if new_status not in allowed_transitions[current_status]:
        raise CaseStatusConflictError(
            f"Invalid status transition from "
            f"'{current_status}' to '{new_status}'"
        )

    if current_status == "Assigned" and new_status == "In Progress":
        assignment_result = db.execute(
            text("""
                SELECT assignment_id
                FROM assignments
                WHERE case_id = :case_id
                  AND final_assigned_team_id IS NOT NULL
                ORDER BY assignment_id DESC
                LIMIT 1
            """),
            {"case_id": case_id}
        )

        if assignment_result.mappings().fetchone() is None:
            raise CaseStatusConflictError(
                "A finalized team assignment is required before moving "
                "this case to In Progress"
            )

    update_result = db.execute(
        text("""
            UPDATE cases
            SET status = :new_status
            WHERE case_id = :case_id
              AND status = :current_status
        """),
        {
            "case_id": case_id,
            "new_status": new_status,
            "current_status": current_status,
        }
    )

    if update_result.rowcount != 1:
        raise CaseStatusConflictError(
            "Case status changed during this update. Reload the case and try again."
        )

    db.execute(
        text("""
            INSERT INTO audit_logs (
                case_id,
                event_type,
                actor,
                details
            )
            VALUES (
                :case_id,
                :event_type,
                :actor,
                CAST(:details AS JSONB)
            )
        """),
        {
            "case_id": case_id,
            "event_type": "CASE_STATUS_CHANGED",
            "actor": actor,
            "details": json.dumps({
                "previous_status": current_status,
                "new_status": new_status
            })
        }
    )

    return {
        "case_id": case_id,
        "previous_status": current_status,
        "status": new_status,
        "actor": actor
    }
