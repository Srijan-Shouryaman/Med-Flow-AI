"""MedFlow AI — merged clinical-meaning-based chunker.

Public API:
    chunk_extraction(payload, full_text=None) -> ChunkCollection

Supports legacy and newer extraction response structures.
Does not modify the database, FastAPI routes, or extraction pipeline.

v4 (safety hardening)
---------------------
1. patient_id is REQUIRED. A payload without a valid patient_id raises
   MissingPatientIdError, so no unscoped chunk can reach the index.
2. Conflicting patient_ids (wrapper vs. extraction vs. metadata) raise
   ConflictingPatientIdError instead of silently picking the first.
3. Negation is normalised in one place (_assertion). Negated / denied /
   ruled-out items can never land in an affirmed chunk, and
   `retrieval_polarity` is derived from the assertion, not the chunk type.
4. Allergy status is fail-safe: anything not explicitly "no known
   allergies" resolves to UNKNOWN (or CONFLICTING), never NKDA.
5. Missing information (NOT_PROVIDED / data_gap / "missing" ...) is routed
   to the data_gaps chunk from every block and never appears as a
   performed procedure, test, or symptom.
6. Chunk ids include the assertion group, and id collisions are no longer
   silently dropped: identical duplicates are merged (provenance kept) and
   reported in ChunkCollection.warnings; same-id/different-text raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "clinical-meaning-v4-safety"


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

class PatientScopeError(ValueError):
    """The payload cannot be safely scoped to exactly one patient."""


class MissingPatientIdError(PatientScopeError):
    """No valid patient_id was found anywhere in the payload."""


class ConflictingPatientIdError(PatientScopeError):
    """Different patient_id values were found in the same payload."""


@dataclass
class ClinicalChunk:
    chunk_id: str
    chunk_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkCollection:
    parent: ClinicalChunk
    children: list[ClinicalChunk]
    warnings: list[str] = field(default_factory=list)


def _s(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (str, int, float)):
        return str(value).strip()

    return ""


def _token(value: Any) -> str:
    """Upper-case, underscore-separated token for enum-like strings."""

    return re.sub(r"[\s\-]+", "_", _s(value).upper())


def _is_true(value: Any) -> bool:
    if value is True:
        return True

    return (
        isinstance(value, str)
        and value.strip().lower() in {"true", "yes", "1"}
    )


# ---------------------------------------------------------------------
# Patient identity helpers
# ---------------------------------------------------------------------

_PLACEHOLDER_IDS = {
    "", "none", "null", "nan", "undefined", "unknown", "n/a", "na",
}


def _normalize_patient_id(value: Any) -> int | str | None:
    """Return a consistently typed patient id (int when numeric).

    Returns None when the field is genuinely absent/blank/placeholder.
    Raises PatientScopeError when a value IS present but cannot be a
    valid id (bool, zero/negative, non-integer float, dict/list ...),
    so garbage is never quietly treated as "no conflict".
    """

    if value is None:
        return None

    if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
        raise PatientScopeError(
            f"Invalid patient_id of type {type(value).__name__}"
        )

    if isinstance(value, float):
        if not value.is_integer():
            raise PatientScopeError("Invalid non-integer patient_id")
        value = int(value)

    if isinstance(value, int):
        if value <= 0:
            raise PatientScopeError("Invalid non-positive patient_id")
        return value

    text = _s(value)

    if text.lower() in _PLACEHOLDER_IDS:
        return None

    if text.isascii() and text.isdigit():
        number = int(text)

        if number <= 0:
            raise PatientScopeError("Invalid non-positive patient_id")

        return number

    return text


def _resolve_patient_id(
    sources: Iterable[tuple[str, Any]],
) -> int | str:
    """Find the single patient_id for this payload.

    Raises MissingPatientIdError if none is found and
    ConflictingPatientIdError if more than one distinct id is found.
    """

    found: dict[int | str, list[str]] = {}

    for label, source in sources:
        if not isinstance(source, dict):
            continue

        pid = _normalize_patient_id(source.get("patient_id"))

        if pid is not None:
            found.setdefault(pid, []).append(label)

    if not found:
        raise MissingPatientIdError(
            "patient_id is missing from the extraction payload; "
            "refusing to create unscoped chunks."
        )

    if len(found) > 1:
        detail = "; ".join(
            f"{pid!r} in {', '.join(labels)}"
            for pid, labels in found.items()
        )

        logger.error(
            "Conflicting patient_id values in extraction payload",
            extra={"locations": [l for ls in found.values() for l in ls]},
        )

        raise ConflictingPatientIdError(
            f"Conflicting patient_id values: {detail}"
        )

    return next(iter(found))


def _subject(patient_id: int | str, case_id: str) -> str:
    """Text prefix that identifies the patient inside chunk text."""

    return f"Patient {patient_id} (case {case_id})"


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", _s(value) or "UNKNOWN")


def _stable_id(
    case_id: str,
    chunk_type: str,
    text: str,
    patient_id: int | str | None = None,
    report_id: Any = None,
    group: str = "",
) -> str:
    """Deterministic id, namespaced by patient so identical text from two
    patients can never collide. The assertion group is part of the digest
    so the same wording under different assertions stays distinct."""

    digest = hashlib.sha256(
        f"{patient_id}|{report_id}|{group}|{text}".encode("utf-8")
    ).hexdigest()[:20]

    prefix = f"P{_safe(patient_id)}:" if patient_id is not None else ""

    return f"{prefix}{_safe(case_id)}:{chunk_type}:{digest}"


def _unique(items: Iterable[Any]) -> list[Any]:
    output = []
    seen = set()

    for item in items:
        if item is None:
            continue

        marker = json.dumps(
            item,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        if marker not in seen:
            seen.add(marker)
            output.append(item)

    return output


def _unwrap(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support legacy API responses and direct extraction JSON."""

    outer = payload

    nested = payload.get("clinical_extraction")

    if isinstance(nested, dict):
        outer = {**payload, **nested}

    extraction_json = outer.get("extraction_json")

    if not isinstance(extraction_json, dict):
        extraction_json = outer

    if not isinstance(extraction_json.get("common"), dict):
        raise ValueError(
            "Expected extraction_json.common to be a JSON object."
        )

    return extraction_json, outer


# ---------------------------------------------------------------------
# Assertion / negation / missing-data normalisation
# ---------------------------------------------------------------------

_ABSENT_TOKENS = {
    "ABSENT",
    "NEGATED",
    "DENIED",
    "DENIES",
    "NOT_PRESENT",
    "NO",
    "RULED_OUT",
}

# "Information was not supplied" -- NOT the same as "not done".
_GAP_TOKENS = {
    "NOT_PROVIDED",
    "MISSING",
    "NOT_REPORTED",
    "NOT_DOCUMENTED",
    "NOT_AVAILABLE",
    "UNAVAILABLE",
    "NOT_SUPPLIED",
    "DATA_GAP",
}


def _assertion(
    entity: dict[str, Any],
    default: str = "UNKNOWN",
) -> str:
    """Single source of truth for an entity's assertion.

    Precedence (safety first):
      1. data-gap markers  -> NOT_PROVIDED (missing info, never a fact)
      2. explicit negation -> ABSENT (a `negated` flag beats a stray
         PRESENT assertion; a negated item must never read as positive)
      3. explicit assertion, then status, then default
    """

    if _token(entity.get("evidence_role")) == "DATA_GAP":
        return "NOT_PROVIDED"

    assertion = _token(entity.get("assertion"))
    status = _token(entity.get("status"))

    if assertion in _GAP_TOKENS or (not assertion and status in _GAP_TOKENS):
        return "NOT_PROVIDED"

    if _is_true(entity.get("negated")):
        if assertion and assertion not in _ABSENT_TOKENS:
            logger.warning(
                "Entity has negated=true but assertion=%s; treating as ABSENT",
                assertion,
            )

        return "ABSENT"

    if assertion:
        return "ABSENT" if assertion in _ABSENT_TOKENS else assertion

    if status in _ABSENT_TOKENS:
        return "ABSENT"

    if status in {"PRESENT", "MENTIONED", "ACTIVE"}:
        return "PRESENT"

    if status in {"UNCERTAIN", "UNKNOWN"}:
        return "UNKNOWN"

    return default


def _is_gap(entity: dict[str, Any]) -> bool:
    return _assertion(entity) == "NOT_PROVIDED"


def _assertion_suffix(assertion: str) -> str:
    """Make non-affirmed groups explicit in the chunk text itself."""

    if assertion in {"", "PRESENT"}:
        return ""

    return f" [assertion {assertion.replace('_', ' ').lower()}]"


def _polarity(chunk_type: str, assertion_group: Any) -> str:
    if chunk_type == "family_history":
        return "family"

    if chunk_type == "pertinent_negatives":
        return "negated"

    if chunk_type == "data_gaps":
        return "not_documented"

    group = _s(assertion_group).upper()

    if group in {"", "PRESENT"}:
        return "affirmed"

    if group == "ABSENT":
        return "negated"

    if group == "NOT_PROVIDED":
        return "not_documented"

    if group in {"UNKNOWN", "UNCERTAIN"}:
        return "uncertain"

    return "qualified"


def _is_family(entity: dict[str, Any]) -> bool:
    relation = _s(entity.get("relation")).lower()
    experiencer = _s(entity.get("experiencer")).lower()
    history_type = _s(entity.get("history_type")).lower()

    return (
        history_type == "family"
        or experiencer in {
            "family",
            "family_member",
            "relative",
        }
        or relation in {
            "mother",
            "father",
            "parent",
            "sibling",
            "brother",
            "sister",
            "grandmother",
            "grandfather",
            "family",
            "aunt",
            "uncle",
        }
        or _assertion(entity) == "FAMILY"
    )


def _name(entity: dict[str, Any]) -> str:
    return (
        _s(entity.get("name"))
        or _s(entity.get("measurement"))
        or _s(entity.get("substance"))
        or _s(entity.get("description"))
        or _s(entity.get("type"))
        or "clinical item"
    )


def _entity_text(
    entity: dict[str, Any],
    include_duration: bool = True,
) -> str:
    name = _name(entity)
    result = _s(entity.get("result"))
    value = _s(entity.get("value"))
    unit = _s(entity.get("unit"))

    if result:
        phrase = (
            f"{name}: {result}"
            f"{(' ' + unit) if unit else ''}"
        )
    elif value:
        phrase = (
            f"{name}: {value}"
            f"{(' ' + unit) if unit else ''}"
        )
    elif _s(entity.get("description")):
        phrase = _s(entity.get("description"))
    else:
        phrase = name

    if include_duration and _s(entity.get("duration")):
        phrase += f" for {_s(entity.get('duration'))}"

    for key, label in (
        ("severity", "severity"),
        ("laterality", "laterality"),
        ("qualifier", "qualifier"),
    ):
        value_text = _s(entity.get(key))

        if value_text:
            phrase += f" ({label}: {value_text})"

    return phrase


def _evidence(entity: dict[str, Any]) -> str:
    return _s(entity.get("evidence"))


def _entity_spans(entity: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect evidence spans from both `evidence_spans` (common block)
    and the bare `span` list (department-specific block)."""

    spans: list[dict[str, Any]] = []

    for span in entity.get("evidence_spans") or []:
        if isinstance(span, dict):
            spans.append(span)

    if isinstance(entity.get("span"), list):
        spans.append({
            "span": entity["span"],
            "sentence_index": entity.get("sentence_index"),
        })

    return spans


def _make_chunk(
    case_id: str,
    report_id: Any,
    chunk_type: str,
    text: str,
    entities: list[dict[str, Any]],
    parent_id: str,
    outer: dict[str, Any],
    extra: dict[str, Any] | None = None,
    patient_id: int | str | None = None,
) -> ClinicalChunk:
    if patient_id is None:
        # Defence in depth: chunk_extraction already guarantees this.
        raise MissingPatientIdError(
            "Refusing to build a chunk without a patient_id."
        )

    text = re.sub(r"\s+", " ", text).strip()

    prediction = (
        outer.get("prediction")
        if isinstance(outer.get("prediction"), dict)
        else {}
    )

    fact_ids = _unique([
        entity.get("fact_id")
        for entity in entities
        if entity.get("fact_id")
    ])

    entity_ids = _unique([
        entity.get("entity_id")
        for entity in entities
        if entity.get("entity_id")
    ])

    spans = []

    for entity in entities:
        spans.extend(_entity_spans(entity))

    outer_metadata = outer.get("metadata")
    if not isinstance(outer_metadata, dict):
        outer_metadata = {}

    metadata = {
        "patient_id": patient_id,
        "case_id": case_id,
        "report_id": report_id,
        "chunk_type": chunk_type,
        "parent_id": parent_id,
        "is_parent": False,
        "fact_ids": fact_ids,
        "entity_ids": entity_ids,
        "evidence_spans": _unique(spans),
        "extraction_schema_version": (
            outer.get("extraction_schema_version")
            or outer_metadata.get("extraction_schema_version")
            or outer_metadata.get("schema_version")
        ),
        "requires_human_review": True,
        "chunker_version": CHUNKER_VERSION,
        # Prediction is metadata only, never clinical text.
        "predicted_department": prediction.get(
            "predicted_department"
        ),
        "prediction_confidence": prediction.get(
            "confidence_score"
        ),
    }

    if extra:
        metadata.update(extra)

    if "retrieval_polarity" not in (extra or {}):
        metadata["retrieval_polarity"] = _polarity(
            chunk_type, metadata.get("assertion_group")
        )

    return ClinicalChunk(
        chunk_id=_stable_id(
            case_id,
            chunk_type,
            text,
            patient_id,
            report_id,
            _s(metadata.get("assertion_group")),
        ),
        chunk_type=chunk_type,
        text=text,
        metadata=metadata,
    )


def _all_entities(
    common: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    """Entities of one block, EXCLUDING data-gap entities.

    Gaps are collected separately (see _common_gap_entities) so that
    missing information can never be rendered as a clinical finding."""

    values = common.get(key, [])

    if not isinstance(values, list):
        return []

    return [
        item
        for item in values
        if isinstance(item, dict) and not _is_gap(item)
    ]


def _common_gap_entities(common: dict[str, Any]) -> list[dict[str, Any]]:
    gaps = []

    for key, values in common.items():
        if not isinstance(values, list):
            continue

        for item in values:
            if isinstance(item, dict) and _is_gap(item):
                gaps.append({**item, "_category": key})

    return gaps


def _dedup_by_fact(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop true duplicates only. The assertion is part of the key, so a
    present and an absent mention sharing one fact_id are both kept
    (dropping either one would silently flip or erase polarity)."""

    output = []
    seen = set()

    for entity in entities:
        fact_id = _s(entity.get("fact_id"))

        if fact_id:
            key = ("fact", fact_id, _assertion(entity))
        else:
            key = (
                "entity",
                _s(entity.get("entity_id")),
                _name(entity).lower(),
                _assertion(entity),
            )

        if key in seen:
            continue

        seen.add(key)
        output.append(entity)

    return output


# ---------------------------------------------------------------------
# Allergy status (fail-safe)
# ---------------------------------------------------------------------

_ALLERGY_NONE = {
    "NKDA", "NKA", "NONE", "NONE_KNOWN",
    "NO_KNOWN_ALLERGIES", "NO_KNOWN_DRUG_ALLERGIES",
}

_ALLERGY_UNKNOWN = {
    "UNKNOWN", "UNCERTAIN", "NOT_ADDRESSED", "UNKNOWN_DOCUMENTED",
    "NOT_PROVIDED", "NOT_DOCUMENTED", "UNSPECIFIED",
}

_ALLERGY_PRESENT = {
    "PRESENT", "ALLERGIC", "ALLERGY_PRESENT",
    "HAS_ALLERGY", "HAS_ALLERGIES", "KNOWN_ALLERGIES",
}


def _resolve_allergy(
    meta: Any,
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return {status, allergens, evidence, documented}.

    status is one of NONE_KNOWN, PRESENT, CONFLICTING, UNKNOWN.
    NONE_KNOWN is only produced by an EXPLICIT no-known-allergies value.
    Anything unrecognised resolves to UNKNOWN, never NONE_KNOWN."""

    raw = ""
    evidence = ""

    if isinstance(meta, dict):
        raw = _token(
            meta.get("status")
            or meta.get("assertion")
            or meta.get("allergy_status")
        )
        evidence = _s(meta.get("evidence"))
    elif isinstance(meta, str):
        raw = _token(meta)
    elif meta is not None:
        logger.warning(
            "Unrecognised allergy_status type %s; treating as UNKNOWN",
            type(meta).__name__,
        )
        raw = "UNKNOWN"

    if raw in _ALLERGY_NONE:
        raw_class = "NONE_KNOWN"
    elif raw in _ALLERGY_PRESENT:
        raw_class = "PRESENT"
    elif raw in _ALLERGY_UNKNOWN:
        raw_class = "UNKNOWN"
    elif raw:
        logger.warning(
            "Unrecognised allergy status %r; treating as UNKNOWN", raw
        )
        raw_class = "UNKNOWN"
    else:
        raw_class = ""

    allergens: list[str] = []
    entity_none = False

    for entity in entities:
        tok = _token(entity.get("assertion") or entity.get("status"))
        name = _s(entity.get("name")) or _s(entity.get("substance"))
        name_tok = _token(name)

        if not evidence:
            evidence = _evidence(entity)

        if _assertion(entity) == "ABSENT" and name_tok not in _ALLERGY_NONE:
            continue  # "not allergic to X" is NOT "no known allergies"

        if tok in _ALLERGY_NONE or name_tok in _ALLERGY_NONE:
            entity_none = True
        elif tok in _ALLERGY_UNKNOWN:
            continue
        elif name and (
            tok == "" or tok in _ALLERGY_PRESENT
            or _assertion(entity) == "PRESENT"
        ):
            allergens.append(name)

    allergens = _unique(allergens)

    if allergens:
        contradicted = raw_class == "NONE_KNOWN" or (
            not raw_class and entity_none
        )
        status = "CONFLICTING" if contradicted else "PRESENT"
    elif raw_class:
        status = raw_class
    elif entity_none:
        status = "NONE_KNOWN"
    else:
        status = "UNKNOWN"

    return {
        "status": status,
        "allergens": allergens,
        "evidence": evidence,
        "documented": bool(raw or entities),
    }


# ---------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------

def _resolve_collisions(
    children: list[ClinicalChunk],
) -> tuple[list[ClinicalChunk], list[str]]:
    """Never silently drop a chunk.

    Same id + same text  -> true duplicate: merge provenance into the
                            first chunk and report it.
    Same id + other text -> id collision: raise, so it is fixed before
                            anything is indexed.
    """

    kept: dict[str, ClinicalChunk] = {}
    ordered: list[ClinicalChunk] = []
    warnings: list[str] = []

    for child in children:
        prior = kept.get(child.chunk_id)

        if prior is None:
            kept[child.chunk_id] = child
            ordered.append(child)
            continue

        if prior.text != child.text:
            raise RuntimeError(
                f"chunk_id collision with different text: {child.chunk_id}"
            )

        for key in ("fact_ids", "entity_ids", "evidence_spans"):
            prior.metadata[key] = _unique(
                list(prior.metadata.get(key, []))
                + list(child.metadata.get(key, []))
            )

        message = (
            f"Duplicate chunk_id {child.chunk_id} merged "
            f"({child.chunk_type})"
        )

        warnings.append(message)
        logger.warning(message)

    return ordered, warnings


def chunk_extraction(
    payload: dict[str, Any],
    full_text: str | None = None,
) -> ChunkCollection:
    """Convert a clinical extraction response into semantic chunks.

    Raises:
        MissingPatientIdError: no valid patient_id in the payload.
        ConflictingPatientIdError: more than one distinct patient_id.
    """

    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object")

    extraction, outer = _unwrap(payload)

    common = extraction["common"]

    patient_id = _resolve_patient_id([
        ("payload", payload),
        ("clinical_extraction", payload.get("clinical_extraction")),
        ("outer", outer),
        ("extraction_json", extraction),
        ("payload.metadata", payload.get("metadata")),
        ("outer.metadata", outer.get("metadata")),
        ("extraction_json.metadata", extraction.get("metadata")),
    ])

    department_specific = (
        extraction.get("department_specific") or {}
    )

    if not isinstance(department_specific, dict):
        department_specific = {}

    case_id = (
        _s(outer.get("case_id") or payload.get("case_id"))
        or "UNKNOWN_CASE"
    )

    report_id = outer.get(
        "report_id",
        payload.get("report_id"),
    )

    who = _subject(patient_id, case_id)

    prediction = (
        outer.get("prediction")
        if isinstance(outer.get("prediction"), dict)
        else {}
    )

    # Never pretend a placeholder is the original report text.
    parent_text = (
        full_text.strip()
        if isinstance(full_text, str) and full_text.strip()
        else (
            f"Clinical report {report_id} for {who}. "
            "Original report text was not supplied "
            "to the chunker."
        )
    )

    parent = ClinicalChunk(
        chunk_id=_stable_id(
            case_id,
            "parent_report",
            parent_text,
            patient_id,
            report_id,
        ),
        chunk_type="parent_report",
        text=parent_text,
        metadata={
            "patient_id": patient_id,
            "case_id": case_id,
            "report_id": report_id,
            "chunk_type": "parent_report",
            "is_parent": True,
            "parent_id": None,
            "chunker_version": CHUNKER_VERSION,
            "predicted_department": prediction.get(
                "predicted_department"
            ),
            "prediction_confidence": prediction.get(
                "confidence_score"
            ),
            "requires_human_review": True,
            "original_text_supplied": bool(
                isinstance(full_text, str) and full_text.strip()
            ),
        },
    )

    children: list[ClinicalChunk] = []

    def mk(
        chunk_type: str,
        text: str,
        entities: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> ClinicalChunk:
        return _make_chunk(
            case_id,
            report_id,
            chunk_type,
            text,
            entities,
            parent.chunk_id,
            outer,
            extra,
            patient_id=patient_id,
        )

    def merge_into(
        existing: ClinicalChunk,
        addition: str,
        entities: list[dict[str, Any]],
    ) -> None:
        """Append negatives to an existing negatives chunk and keep its
        ids/spans consistent."""

        existing.text = re.sub(
            r"\s+",
            " ",
            existing.text.rstrip(".") + addition + ".",
        ).strip()

        existing.chunk_id = _stable_id(
            case_id,
            "pertinent_negatives",
            existing.text,
            patient_id,
            report_id,
            _s(existing.metadata.get("assertion_group")),
        )

        existing.metadata["fact_ids"] = _unique(
            existing.metadata.get("fact_ids", [])
            + [
                e.get("fact_id")
                for e in entities
                if e.get("fact_id")
            ]
        )

        existing.metadata["entity_ids"] = _unique(
            existing.metadata.get("entity_ids", [])
            + [
                e.get("entity_id")
                for e in entities
                if e.get("entity_id")
            ]
        )

        existing.metadata["evidence_spans"] = _unique(
            existing.metadata.get("evidence_spans", [])
            + [
                span
                for e in entities
                for span in _entity_spans(e)
            ]
        )

    def add_negative(
        label: str,
        descriptions: list[str],
        entities: list[dict[str, Any]],
    ) -> None:
        """Route ANY negated item into the pertinent-negatives chunk."""

        descriptions = _unique(descriptions)

        if not descriptions:
            return

        existing = next(
            (
                chunk for chunk in children
                if chunk.chunk_type == "pertinent_negatives"
            ),
            None,
        )

        if existing:
            merge_into(
                existing,
                f"; {label}: " + ", ".join(descriptions),
                entities,
            )
            return

        evidence = _unique([
            _evidence(entity)
            for entity in entities
            if _evidence(entity)
        ])

        text = (
            f"{who} — Pertinent negatives "
            f"(report {report_id}): "
            f"{label[:1].upper() + label[1:]}: "
            + ", ".join(descriptions)
            + "."
        )

        if evidence:
            text += " Source: " + " ".join(evidence[:1])

        children.append(
            mk(
                "pertinent_negatives",
                text,
                entities,
                {
                    "assertion_group": "ABSENT",
                    "retrieval_polarity": "negated",
                },
            )
        )

    # ---------------------------------------------------------
    # 1. Symptoms: positive, absent and uncertain are separate.
    # ---------------------------------------------------------

    symptoms = _dedup_by_fact(
        _all_entities(common, "symptoms")
    )

    present = [
        entity for entity in symptoms
        if _assertion(entity) == "PRESENT"
    ]

    absent = [
        entity for entity in symptoms
        if _assertion(entity) == "ABSENT"
    ]

    uncertain = [
        entity for entity in symptoms
        if _assertion(entity) not in {"PRESENT", "ABSENT"}
    ]

    if present:
        by_name: dict[str, dict[str, Any]] = {}

        for entity in present:
            key = _name(entity).lower()

            if (
                key not in by_name
                or (
                    not _s(by_name[key].get("duration"))
                    and _s(entity.get("duration"))
                )
            ):
                by_name[key] = entity

        phrases = _unique([
            _entity_text(entity)
            for entity in by_name.values()
        ])

        evidence = _unique([
            _evidence(entity)
            for entity in present
            if _evidence(entity)
        ])

        text = (
            f"{who} — Present symptoms "
            f"(report {report_id}): "
            + "; ".join(phrases)
            + "."
        )

        if evidence and not any(
            phrase.lower() in text.lower()
            for phrase in evidence
        ):
            text += " Evidence: " + " ".join(evidence[:2])

        children.append(
            mk(
                "symptoms_present",
                text,
                present,
                {"assertion_group": "PRESENT"},
            )
        )

    if absent:
        names = _unique([
            _name(entity)
            for entity in absent
        ])

        evidence = _unique([
            _evidence(entity)
            for entity in absent
            if _evidence(entity)
        ])

        text = (
            f"{who} — Pertinent negatives "
            f"(report {report_id}): Patient denies "
            + ", ".join(names)
            + "."
        )

        if evidence:
            text += " Source: " + " ".join(evidence[:1])

        children.append(
            mk(
                "pertinent_negatives",
                text,
                absent,
                {
                    "assertion_group": "ABSENT",
                    "retrieval_polarity": "negated",
                },
            )
        )

    if uncertain:
        text = (
            f"{who} — Uncertain symptoms "
            f"(report {report_id}): "
            + "; ".join(_unique([
                _entity_text(entity)
                for entity in uncertain
            ]))
            + "."
        )

        children.append(
            mk(
                "symptoms_uncertain",
                text,
                uncertain,
                {"assertion_group": "UNCERTAIN"},
            )
        )

    # ---------------------------------------------------------
    # 2. Diagnoses: family-history facts excluded; ruled-out
    #    diagnoses go to pertinent negatives (never affirmed).
    # ---------------------------------------------------------

    all_diagnoses = [
        entity
        for entity in _dedup_by_fact(
            _all_entities(common, "diagnoses")
        )
        if not _is_family(entity)
    ]

    diagnoses = [
        e for e in all_diagnoses if _assertion(e) != "ABSENT"
    ]

    ruled_out = [
        e for e in all_diagnoses if _assertion(e) == "ABSENT"
    ]

    if diagnoses:
        groups: dict[str, list[dict[str, Any]]] = {}

        for entity in diagnoses:
            groups.setdefault(
                _assertion(entity),
                [],
            ).append(entity)

        for assertion, group in groups.items():
            text = (
                f"{who} — Diagnoses{_assertion_suffix(assertion)} "
                f"(report {report_id}): "
                + "; ".join(_unique([
                    _entity_text(entity)
                    for entity in group
                ]))
                + "."
            )

            children.append(
                mk(
                    "diagnoses",
                    text,
                    group,
                    {"assertion_group": assertion},
                )
            )

    if ruled_out:
        add_negative(
            "diagnoses ruled out",
            [_name(entity) for entity in ruled_out],
            ruled_out,
        )

    # ---------------------------------------------------------
    # 3. History: family kept separate; negated history -> negatives.
    # ---------------------------------------------------------

    history = _dedup_by_fact(
        _all_entities(common, "clinical_history")
    )

    family = [
        entity for entity in history
        if _is_family(entity)
    ]

    if family:
        phrases = []

        for entity in family:
            relation = (
                _s(entity.get("relation"))
                or "family member"
            )

            detail = (
                _s(entity.get("description"))
                or _s(entity.get("name"))
                or _evidence(entity)
            )

            if _assertion(entity) == "ABSENT":
                phrases.append(
                    f"{relation}: NO history of "
                    f"{detail or 'the stated condition'}"
                )
            elif (
                detail
                and not detail.lower().startswith(
                    relation.lower() + ":"
                )
            ):
                phrases.append(f"{relation}: {detail}")
            elif detail:
                phrases.append(detail)

        text = (
            f"{who} — Family history "
            f"(report {report_id}): "
            + "; ".join(_unique(phrases))
            + "."
        )

        children.append(
            mk(
                "family_history",
                text,
                family,
                {"retrieval_polarity": "family"},
            )
        )

    nonfamily_history = [
        entity for entity in history
        if not _is_family(entity)
    ]

    history_absent = [
        e for e in nonfamily_history if _assertion(e) == "ABSENT"
    ]

    history_other = [
        e for e in nonfamily_history if _assertion(e) != "ABSENT"
    ]

    if history_other:
        text = (
            f"{who} — Clinical history "
            f"(report {report_id}): "
            + "; ".join(_unique([
                _entity_text(entity)
                for entity in history_other
            ]))
            + "."
        )

        children.append(
            mk("clinical_history", text, history_other)
        )

    if history_absent:
        add_negative(
            "history denied",
            [_name(entity) for entity in history_absent],
            history_absent,
        )

    # ---------------------------------------------------------
    # 4. Vitals: canonical source; avoid duplicate investigations.
    # ---------------------------------------------------------

    vitals = _dedup_by_fact(
        _all_entities(common, "vital_signs")
    )

    valid_vitals = [
        entity for entity in vitals
        if (
            _s(entity.get("value"))
            or _s(entity.get("result"))
        )
    ]

    if valid_vitals:
        phrases = []

        for entity in valid_vitals:
            measurement = (
                _s(entity.get("measurement"))
                or _name(entity)
            )

            if measurement:
                measurement = (
                    measurement[0].upper()
                    + measurement[1:]
                )

            value = (
                _s(entity.get("value"))
                or _s(entity.get("result"))
            )

            unit = _s(entity.get("unit"))

            phrases.append(
                f"{measurement}: {value}"
                f"{(' ' + unit) if unit else ''}"
            )

        text = (
            f"{who} — Vitals "
            f"(report {report_id}): "
            + "; ".join(_unique(phrases))
            + "."
        )

        children.append(mk("vitals", text, valid_vitals))

    # ---------------------------------------------------------
    # 5. Investigations: keep only entities with actual results.
    #    (Data-gap entities were already removed by _all_entities.)
    # ---------------------------------------------------------

    investigations = _dedup_by_fact(
        _all_entities(common, "investigations")
    )

    tests = []

    for entity in investigations:
        if not (
            _s(entity.get("result"))
            or _s(entity.get("value"))
        ):
            continue

        tests.append(entity)

    if tests:
        groups = {}

        for entity in tests:
            groups.setdefault(
                _assertion(entity),
                [],
            ).append(entity)

        for assertion, group in groups.items():
            text = (
                f"{who} — Investigations{_assertion_suffix(assertion)} "
                f"(report {report_id}): "
                + "; ".join(_unique([
                    _entity_text(
                        entity,
                        include_duration=False,
                    )
                    for entity in group
                ]))
                + "."
            )

            children.append(
                mk(
                    "investigations",
                    text,
                    group,
                    {"assertion_group": assertion},
                )
            )

    # ---------------------------------------------------------
    # 6. Medications and procedures.
    # ---------------------------------------------------------

    medications = [
        entity
        for entity in _dedup_by_fact(
            _all_entities(common, "medications")
        )
        if _assertion(entity) != "ABSENT"
    ]

    if medications:
        text = (
            f"{who} — Medications "
            f"(report {report_id}): "
            + "; ".join(_unique([
                _entity_text(entity)
                for entity in medications
            ]))
            + "."
        )

        children.append(mk("medications", text, medications))

    procedures = [
        entity
        for entity in _dedup_by_fact(
            _all_entities(common, "procedures")
        )
        if _assertion(entity) != "ABSENT"
    ]

    if procedures:
        groups = {}

        for entity in procedures:
            groups.setdefault(
                _assertion(entity),
                [],
            ).append(entity)

        for assertion, group in groups.items():
            # A procedure whose assertion is not affirmed is only a
            # MENTION; the text must not read as "was performed".
            if assertion == "PRESENT":
                header = "Procedures and tests"
            else:
                header = (
                    "Procedures and tests mentioned "
                    "(performance and results not confirmed)"
                )

            text = (
                f"{who} — {header}{_assertion_suffix(assertion)} "
                f"(report {report_id}): "
                + "; ".join(_unique([
                    _entity_text(entity)
                    for entity in group
                ]))
                + "."
            )

            children.append(
                mk(
                    "procedures_tests",
                    text,
                    group,
                    {"assertion_group": assertion},
                )
            )

    # ---------------------------------------------------------
    # 7. Department-specific findings and documented data gaps.
    # Do not re-embed facts already represented in common.
    # ---------------------------------------------------------

    dept_negatives = []
    gap_entities = _common_gap_entities(common)

    common_facts = {
        _s(entity.get("fact_id"))
        for values in common.values()
        if isinstance(values, list)
        for entity in values
        if isinstance(entity, dict)
        and _s(entity.get("fact_id"))
    }

    for department, block in department_specific.items():
        if not isinstance(block, dict):
            continue

        for category, items in block.items():
            if not isinstance(items, list):
                continue

            for entity in items:
                if not isinstance(entity, dict):
                    continue

                assertion = _assertion(entity)

                if assertion == "NOT_PROVIDED":
                    gap_entities.append({
                        **entity,
                        "_department": department,
                        "_category": category,
                    })

                elif (
                    assertion == "ABSENT"
                    and category in {"findings", "symptoms"}
                ):
                    fact_id = _s(entity.get("fact_id"))

                    if not fact_id or fact_id not in common_facts:
                        dept_negatives.append({
                            **entity,
                            "_department": department,
                        })

    if dept_negatives:
        add_negative(
            "examination",
            [_name(entity) for entity in dept_negatives],
            dept_negatives,
        )

    # ---------------------------------------------------------
    # 8. Data gaps: explicit missing-information only. A gap means
    #    "not supplied", never "not done".
    # ---------------------------------------------------------

    gap_map: dict[str, dict[str, Any]] = {}

    for entity in gap_entities:
        name = _name(entity)
        key = name.lower()

        if key not in gap_map:
            gap_map[key] = entity

    extraction_metadata = extraction.get("metadata")

    if not isinstance(extraction_metadata, dict):
        extraction_metadata = {}

    explicit_gaps = extraction_metadata.get("data_gaps", [])

    if isinstance(explicit_gaps, list):
        for item in explicit_gaps:
            if isinstance(item, dict):
                name = _s(
                    item.get("name")
                    or item.get("item")
                    or item.get("description")
                    or item.get("text")
                )

                gap_entity = {
                    **item,
                    "name": name or "documented information gap",
                    "evidence_role": "data_gap",
                    "assertion": "NOT_PROVIDED",
                }

            elif isinstance(item, str) and item.strip():
                gap_entity = {
                    "name": item.strip(),
                    "evidence": item.strip(),
                    "evidence_role": "data_gap",
                    "assertion": "NOT_PROVIDED",
                }

            else:
                continue

            key = _name(gap_entity).lower()
            gap_map.setdefault(key, gap_entity)

    if gap_map:
        names = _unique([
            _name(entity)
            for entity in gap_map.values()
        ])

        text = (
            f"{who} — Data gaps "
            f"(report {report_id}): "
            + ", ".join(names)
            + " were not supplied in the available note. "
            "This does NOT establish that these procedures "
            "or tests were not performed."
        )

        children.append(
            mk(
                "data_gaps",
                text,
                list(gap_map.values()),
                {
                    "data_gap_status": "NOT_PROVIDED",
                    "assertion_group": "NOT_PROVIDED",
                },
            )
        )

    # ---------------------------------------------------------
    # 9. Allergy status: fail-safe; only an EXPLICIT no-known-
    #    allergies value may become NKDA.
    # ---------------------------------------------------------

    allergy_entities = _all_entities(common, "allergies")

    allergy = _resolve_allergy(
        extraction_metadata.get("allergy_status"),
        allergy_entities,
    )

    display_status = allergy["status"]
    allergy_evidence = allergy["evidence"]

    if display_status == "NONE_KNOWN":
        allergy_text = (
            f"{who} — Allergy status "
            f"(report {report_id}): "
            "NKDA is documented (no known drug allergies)."
        )
        allergy_polarity = "negated"

    elif display_status == "PRESENT":
        if allergy["allergens"]:
            allergy_text = (
                f"{who} — Allergy status "
                f"(report {report_id}): allergies documented: "
                + ", ".join(allergy["allergens"])
                + "."
            )
        else:
            allergy_text = (
                f"{who} — Allergy status "
                f"(report {report_id}): an allergy is documented "
                "but the allergen is not specified."
            )
        allergy_polarity = "affirmed"

    elif display_status == "CONFLICTING":
        allergy_text = (
            f"{who} — Allergy status "
            f"(report {report_id}): CONFLICTING documentation. "
            "The record states no known allergies but also lists: "
            + ", ".join(allergy["allergens"])
            + ". Verify with the patient before prescribing."
        )
        allergy_polarity = "uncertain"

    else:
        display_status = "UNKNOWN"

        if not allergy["documented"]:
            allergy_text = (
                f"{who} — Allergy status "
                f"(report {report_id}): NOT DOCUMENTED. "
                "The available note does not establish "
                "confirmed allergy status; this does NOT "
                "mean the patient has no allergies."
            )
        else:
            allergy_text = (
                f"{who} — Allergy status "
                f"(report {report_id}): UNKNOWN. "
                "The available note does not establish "
                "confirmed allergy status; this does NOT "
                "mean the patient has no allergies."
            )

        if allergy_evidence:
            allergy_text += f" Evidence: {allergy_evidence}"

        allergy_polarity = "uncertain"

    children.append(
        mk(
            "allergy_status",
            allergy_text,
            allergy_entities,
            {
                "allergy_status": display_status,
                "allergy_documented": allergy["documented"],
                "retrieval_polarity": allergy_polarity,
            },
        )
    )

    # ---------------------------------------------------------
    # 10. Other findings: preserve explicit assertion polarity.
    # ---------------------------------------------------------

    other_findings = _dedup_by_fact(
        _all_entities(common, "other_findings")
    )

    if other_findings:
        groups = {}

        for entity in other_findings:
            groups.setdefault(
                _assertion(entity),
                [],
            ).append(entity)

        for assertion, group in groups.items():
            descriptions = _unique([
                _entity_text(entity)
                for entity in group
            ])

            if assertion == "ABSENT":
                add_negative("examination", descriptions, group)

            else:
                text = (
                    f"{who} — Examination findings"
                    f"{_assertion_suffix(assertion)} "
                    f"(report {report_id}): "
                    + "; ".join(descriptions)
                    + "."
                )

                children.append(
                    mk(
                        "exam_findings",
                        text,
                        group,
                        {"assertion_group": assertion},
                    )
                )

    # ---------------------------------------------------------
    # 11. Deterministic output; collisions merged or raised,
    #     never silently dropped.
    # ---------------------------------------------------------

    children, warnings = _resolve_collisions(children)

    logger.info(
        "Clinical extraction chunked",
        extra={
            "patient_id": patient_id,
            "case_id": case_id,
            "report_id": report_id,
            "chunk_count": len(children),
            "duplicate_merges": len(warnings),
            "chunker_version": CHUNKER_VERSION,
        },
    )

    return ChunkCollection(
        parent=parent,
        children=children,
        warnings=warnings,
    )


def chunk_clinical_extraction(
    payload: dict[str, Any],
) -> list[ClinicalChunk]:
    """Compatibility wrapper for callers expecting a list."""
    return chunk_extraction(payload).children