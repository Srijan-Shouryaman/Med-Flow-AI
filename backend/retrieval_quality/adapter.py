
"""Adapter to MedFlow's existing retrieval services. No production files changed."""
from __future__ import annotations

from typing import Any
import re


def _get(obj: Any, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _metadata(row):
    value = _get(row, "metadata", default={})

    if isinstance(value, dict):
        return value

    # Some database drivers or services may return JSON metadata as a string.
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    return {}


def _report_id(row):
    """Extract the source report ID from a result row or its text."""
    md = _metadata(row)

    value = _get(
        row,
        "report_id",
        "source_report_id",
        default=None,
    )

    if value is None:
        value = md.get("report_id") or md.get("source_report_id")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    text = str(
        _get(row, "chunk_text", "text", "content", default="") or ""
    )

    # Supports formats such as "report 63", "report_id: 63",
    # and "report ID 63".
    match = re.search(
        r"\breport(?:\s+id)?\s*[:#]?\s*(\d+)\b",
        text,
        re.IGNORECASE,
    )

    return int(match.group(1)) if match else None


def _load_report_dates(session, rows):
    """
    Resolve source report dates from public.reports.

    This is test-adapter logic only. It does not update database records.
    """
    ids = sorted(
        {
            report_id
            for row in rows
            if (report_id := _report_id(row)) is not None
        }
    )

    if not ids:
        return {}

    from sqlalchemy import text

    bind_names = {
        f"rid_{index}": report_id
        for index, report_id in enumerate(ids)
    }

    placeholders = ", ".join(
        f":{name}" for name in bind_names
    )

    sql = text(
        "SELECT report_id, extracted_text "
        "FROM public.reports "
        f"WHERE report_id IN ({placeholders})"
    )

    result = session.execute(sql, bind_names).mappings().all()

    dates = {}

    for record in result:
        content = str(record.get("extracted_text") or "")

        # Match the report's explicit date, not a patient's birth date
        # or another unrelated date elsewhere in the document.
        match = re.search(
            r"^\s*Report\s+date\s*:\s*(\d{4}-\d{2}-\d{2})",
            content,
            re.IGNORECASE | re.MULTILINE,
        )

        # Some reports use "Report date 2026-06-04" without a colon.
        if not match:
            match = re.search(
                r"^\s*Report\s+date\s+(\d{4}-\d{2}-\d{2})",
                content,
                re.IGNORECASE | re.MULTILINE,
            )

        if match:
            dates[int(record["report_id"])] = match.group(1)

    return dates


def _normalise(row, report_dates=None):
    """Convert a retrieval result into the test suite's common format."""
    md = _metadata(row)

    chunk_text = _get(
        row,
        "chunk_text",
        "text",
        "content",
        default="",
    )

    chunk_type = _get(
        row,
        "chunk_type",
        "type",
        "category",
        default=None,
    )

    if not chunk_type:
        chunk_type = (
            md.get("chunk_type")
            or md.get("type")
            or md.get("category")
            or ""
        )

    report_date = _get(
        row,
        "report_date",
        "source_report_date",
        default=None,
    )

    if not report_date:
        report_date = (
            md.get("report_date")
            or md.get("source_report_date")
            or md.get("date")
            or ""
        )

    # If the chunk doesn't carry a date, resolve it from its report ID.
    if not report_date and report_dates:
        source_id = _report_id(row)
        report_date = report_dates.get(source_id, "")

    # Normalize datetime-like values without assuming every value is a date.
    report_date = str(report_date or "").strip()

    if report_date:
        report_date = report_date[:10]

    return {
        "chunk_id": str(
            _get(row, "chunk_id", "id", default="")
        ),
        "patient_id": _get(row, "patient_id", default=None),
        "report_date": report_date,
        "chunk_type": str(chunk_type).strip().lower(),
        "text": str(chunk_text or ""),
        "assertion": _get(
            row,
            "assertion",
            default=md.get("assertion"),
        ),
        "score": _get(
            row,
            "score",
            "similarity_score",
            "sparse_score",
            "rerank_score",
            default=None,
        ),
        "_raw": row,
    }


def retrieve(
    query: str,
    patient_id: int,
    k: int = 5,
    mode: str = "rerank",
) -> list[dict]:
    """Run an existing retrieval mode and normalize its results."""
    if (
        not isinstance(patient_id, int)
        or isinstance(patient_id, bool)
        or patient_id <= 0
    ):
        raise ValueError("patient_id must be a positive integer")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")

    if mode not in {"dense", "sparse", "hybrid", "rerank"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    # Lazy imports allow offline scoring tests to run without loading
    # the database and retrieval dependencies.
    from app.core.database import SessionLocal
    from app.services.retrieval.dense_retrieval_service import (
        search_clinical_chunks,
    )
    from app.services.retrieval.sparse_retrieval_service import (
        search_clinical_chunks_sparse,
    )
    from app.services.retrieval.hybrid_retrieval_service import (
        search_clinical_chunks_hybrid,
    )
    from app.services.retrieval.reranker_service import (
        rerank_clinical_chunks,
    )

    session = SessionLocal()

    try:
        if mode == "dense":
            raw = search_clinical_chunks(
                session,
                query,
                patient_id=patient_id,
                top_k=k,
            )

        elif mode == "sparse":
            raw = search_clinical_chunks_sparse(
                session,
                query,
                patient_id=patient_id,
                top_k=k,
            )

        else:
            raw = search_clinical_chunks_hybrid(
                session,
                query,
                patient_id=patient_id,
                candidate_k=max(20, k),
            )

            raw = list(raw or [])

            if mode == "rerank":
                raw = rerank_clinical_chunks(
                    query,
                    raw,
                    top_k=k,
                )
            else:
                raw = raw[:k]

        raw = list(raw or [])[:k]

        report_dates = _load_report_dates(session, raw)

        return [
            _normalise(row, report_dates)
            for row in raw
        ]

    finally:
        session.close()


def list_chunks(patient_id: int) -> list[dict]:
    """
    Return up to 50 chunks from the existing dense retrieval path
    for diagnostics. This is not a full-table database scan.
    """
    if (
        not isinstance(patient_id, int)
        or isinstance(patient_id, bool)
        or patient_id <= 0
    ):
        raise ValueError("patient_id must be a positive integer")

    from app.core.database import SessionLocal
    from app.services.retrieval.dense_retrieval_service import (
        search_clinical_chunks,
    )

    session = SessionLocal()

    try:
        raw = search_clinical_chunks(
            session,
            "cardiology clinical report",
            patient_id=patient_id,
            top_k=50,
        )

        raw = list(raw or [])
        report_dates = _load_report_dates(session, raw)

        return [
            _normalise(row, report_dates)
            for row in raw
        ]

    finally:
        session.close()
