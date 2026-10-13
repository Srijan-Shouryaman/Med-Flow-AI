from sqlalchemy import (
    Column,
    CheckConstraint,
    String,
    Text,
    Boolean,
    TIMESTAMP
)

from app.core.database import Base


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (
        CheckConstraint(
            "status IN ("
            "'Pending Prediction', 'Predicted', 'Recommended', "
            "'Pending Human Review', 'Assigned', 'Flagged - No Eligible Team', "
            "'In Progress', 'Resolved'"
            ")",
            name="cases_status_check",
        ),
    )

    case_id = Column(
        String(10),
        primary_key=True
    )

    patient_ref_id = Column(
        String(30),
        nullable=False
    )

    report_summary = Column(
        Text,
        nullable=False
    )

    is_emergency = Column(
        Boolean,
        nullable=False
    )

    priority = Column(
        String(9),
        nullable=False
    )

    submitted_at = Column(
        TIMESTAMP,
        nullable=False
    )

    status = Column(
        String(30),
        nullable=False
    )
