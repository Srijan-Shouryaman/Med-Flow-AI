from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# Base clinical entity
# ============================================================

class ClinicalEntity(BaseModel):
    entity_id: str
    evidence: str
    extraction: str


# ============================================================
# Common clinical entities
# ============================================================

class SymptomEntity(ClinicalEntity):
    name: str
    status: str | None = None
    temporality: str | None = None
    negated: bool | None = None
    onset: str | None = None
    duration: str | None = None
    severity: str | None = None
    experiencer: str | None = None


class DiagnosisEntity(ClinicalEntity):
    name: str
    status: str | None = None
    negated: bool | None = None
    clinical_code: str | None = None
    code_system: str | None = None
    temporality: str | None = None


class MedicationEntity(ClinicalEntity):
    name: str
    status: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class AllergyEntity(ClinicalEntity):
    substance: str
    status: str | None = None
    reaction: str | None = None
    severity: str | None = None
    temporality: str | None = None


class ProcedureEntity(ClinicalEntity):
    name: str
    status: str | None = None
    procedure_date: str | None = None
    outcome: str | None = None


class InvestigationEntity(ClinicalEntity):
    name: str
    result: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    date: str | None = None


class ClinicalHistoryEntity(ClinicalEntity):
    history_type: str
    description: str
    date: str | None = None
    status: str | None = None
    temporality: str | None = None


class VitalSignEntity(ClinicalEntity):
    measurement: str
    value: str
    unit: str
    measurement_date: str | None = None
    reference_range: str | None = None


class OtherFindingEntity(ClinicalEntity):
    name: str
    description: str
    status: str | None = None
    temporality: str | None = None


# ============================================================
# Common clinical information
# ============================================================

class CommonClinicalData(BaseModel):
    symptoms: list[SymptomEntity] = Field(default_factory=list)
    diagnoses: list[DiagnosisEntity] = Field(default_factory=list)
    medications: list[MedicationEntity] = Field(default_factory=list)
    allergies: list[AllergyEntity] = Field(default_factory=list)
    procedures: list[ProcedureEntity] = Field(default_factory=list)
    investigations: list[InvestigationEntity] = Field(default_factory=list)
    clinical_history: list[ClinicalHistoryEntity] = Field(
        default_factory=list
    )
    vital_signs: list[VitalSignEntity] = Field(default_factory=list)
    other_findings: list[OtherFindingEntity] = Field(default_factory=list)


# ============================================================
# Department-specific clinical details
#
# These are structured details, not new department predictions.
# The existing SVM remains responsible for routing.
# ============================================================

class CardiovascularPulmonaryDetails(BaseModel):
    cardiovascular_findings: list[OtherFindingEntity] = Field(
        default_factory=list
    )
    pulmonary_findings: list[OtherFindingEntity] = Field(
        default_factory=list
    )
    cardiac_investigations: list[InvestigationEntity] = Field(
        default_factory=list
    )
    respiratory_investigations: list[InvestigationEntity] = Field(
        default_factory=list
    )


class ConsultHistoryPhysicalDetails(BaseModel):
    presenting_complaint: str | None = None
    relevant_history: list[ClinicalHistoryEntity] = Field(
        default_factory=list
    )
    physical_findings: list[OtherFindingEntity] = Field(
        default_factory=list
    )


class GastroenterologyDetails(BaseModel):
    gastrointestinal_symptoms: list[SymptomEntity] = Field(
        default_factory=list
    )
    gastrointestinal_diagnoses: list[DiagnosisEntity] = Field(
        default_factory=list
    )
    gastrointestinal_investigations: list[InvestigationEntity] = Field(
        default_factory=list
    )
    gastrointestinal_procedures: list[ProcedureEntity] = Field(
        default_factory=list
    )


class OrthopedicDetails(BaseModel):
    musculoskeletal_symptoms: list[SymptomEntity] = Field(
        default_factory=list
    )
    musculoskeletal_diagnoses: list[DiagnosisEntity] = Field(
        default_factory=list
    )
    musculoskeletal_findings: list[OtherFindingEntity] = Field(
        default_factory=list
    )
    orthopedic_procedures: list[ProcedureEntity] = Field(
        default_factory=list
    )


class RadiologyDetails(BaseModel):
    imaging_investigations: list[InvestigationEntity] = Field(
        default_factory=list
    )
    imaging_procedures: list[ProcedureEntity] = Field(
        default_factory=list
    )
    imaging_findings: list[OtherFindingEntity] = Field(
        default_factory=list
    )


class DepartmentClinicalData(BaseModel):
    cardiovascular_pulmonary: CardiovascularPulmonaryDetails = Field(
        default_factory=CardiovascularPulmonaryDetails
    )
    consult_history_physical: ConsultHistoryPhysicalDetails = Field(
        default_factory=ConsultHistoryPhysicalDetails
    )
    gastroenterology: GastroenterologyDetails = Field(
        default_factory=GastroenterologyDetails
    )
    orthopedic: OrthopedicDetails = Field(
        default_factory=OrthopedicDetails
    )
    radiology: RadiologyDetails = Field(
        default_factory=RadiologyDetails
    )


# ============================================================
# Extraction metadata
# ============================================================

class ClinicalExtractionMetadata(BaseModel):
    extraction_method: str = "rule_based"
    requires_human_review: bool = True


# ============================================================
# Version 2.0 output contract
# ============================================================

class ClinicalExtractionV2(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    common: CommonClinicalData = Field(
        default_factory=CommonClinicalData
    )
    departments: DepartmentClinicalData = Field(
        default_factory=DepartmentClinicalData
    )
    metadata: ClinicalExtractionMetadata = Field(
        default_factory=ClinicalExtractionMetadata
    )


# ============================================================
# Backward-compatible legacy extraction model
#
# Keep existing extraction_service.py imports working while
# the service migrates to the versioned output.
# ============================================================

class ClinicalExtraction(CommonClinicalData):
    pass


# ============================================================
# Combined extraction contract for API persistence
#
# This matches the agreed JSON structure:
# {
#     "common": {...},
#     "department_specific": {
#         "cardiovascular_pulmonary": {...},
#         "consult_history_physical": {...},
#         "gastroenterology": {...},
#         "orthopedic": {...},
#         "radiology": {...}
#     }
# }
# ============================================================

class CombinedClinicalExtraction(BaseModel):
    common: CommonClinicalData = Field(
        default_factory=CommonClinicalData
    )
    department_specific: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
