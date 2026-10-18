"""Rule-based clinical extraction, schema 2.1 (additive over 2.0).

What changed versus the previous module (see the review notes for the full list):
  * Sentence spans instead of OCR line fragments (line-wrap rejoin, abbreviation-safe
    splitting, character offsets kept with every evidence snippet).
  * Local assertion engine instead of "any negation word anywhere in the sentence".
    Assertions: PRESENT, ABSENT, UNCERTAIN, UNKNOWN, HISTORICAL, FAMILY, FAMILY_ABSENT,
    PLANNED, POLARITY_UNKNOWN, NOT_PROVIDED.
  * Sentence roles: finding, indication, plan, data_gap, boilerplate, heading.
    Only `finding` sentences produce clinical entities.
  * Allergy rules output NKDA / ALLERGIC / UNKNOWN instead of treating prose as a substance,
    plus a document-level `metadata["allergy_status"]` so "unknown", "none" and "not
    addressed" are no longer all an empty list.
  * Structured values: labs, vitals, spirometry, LVEF, medication dose/route/frequency,
    durations, severity, laterality, dates.
  * `common` is a superset of the department blocks; cross-block duplicates share `fact_id`.

Fixes in this revision:
  1. Family history is detected before negation, in both directions (before and after the
     term), and routed OUT of the patient lists. Family conditions only appear as
     `clinical_history` entries with history_type="family" and a `relation`.
  2. Document-level allergy status (`metadata["allergy_status"]`).
  3a. Severity / laterality only attach when adjacent to the term within the same clause.
  3b. Vitals that were also picked up as valueless investigations (e.g. oxygen saturation)
      are de-duplicated; the department block fills the value from the vital instead.
  4.  "Not provided" data gaps (e.g. "No endoscopy result is supplied") no longer become
      negated procedures / investigations in `common`; they remain in department blocks
      with evidence_role="data_gap" and in metadata["data_gaps"].

New fields are additive and optional. The legacy fields (`status`, `negated`, `temporality`,
`evidence` with " | " joins) are still filled, derived from the new assertion, and only use
status values the old code already emitted (MENTIONED, NEGATED, UNCERTAIN, PLANNED, HISTORICAL).
Entity models are built through `_model()`, which drops keys the pydantic class does not
declare, so this module works before and after the schema additions.

NOTE: add these optional fields to the pydantic entity models, otherwise `_model()` drops them
and they never reach anything reading `common`: assertion, fact_id, evidence_role,
evidence_spans, experiencer, relation, laterality, qualifier, unit_source, value_missing,
date_raw (and AllergyEntity.assertion).
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from app.schemas.clinical_entities import (
    AllergyEntity,
    ClinicalExtraction,
    ClinicalExtractionV2,
    ClinicalHistoryEntity,
    DiagnosisEntity,
    InvestigationEntity,
    MedicationEntity,
    ProcedureEntity,
    SymptomEntity,
    VitalSignEntity,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.1"

# ==================================================================
# Clinical terminology
# Heading-like terms ("history of", "past medical history", "current medication",
# "presenting complaint", "physical examination") were removed: they matched section
# titles, not patient facts. History is now derived from the assertion engine.
# ==================================================================

SYMPTOMS = [
    "chest pain", "shortness of breath", "dyspnea", "cough", "fever", "fatigue",
    "headache", "dizziness", "nausea", "vomiting", "abdominal pain", "wheezing",
    "palpitations", "swelling", "weakness", "breathlessness", "hemoptysis",
    "constipation", "diarrhea", "diarrhoea", "bloating", "heartburn", "indigestion",
    "abdominal cramps", "stomach pain", "epigastric pain", "right upper quadrant pain",
    "left lower quadrant pain", "loss of appetite", "reduced appetite",
    "unintentional weight loss", "weight loss", "weight gain", "jaundice",
    "difficulty swallowing", "dysphagia", "painful swallowing", "odynophagia",
    "blood in stool", "rectal bleeding", "black stools", "melena", "vomiting blood",
    "hematemesis", "blood in vomit", "excessive gas", "belching", "acid reflux",
    # added after reviewing real records
    "chest discomfort", "upper abdominal discomfort", "epigastric discomfort",
    "abdominal discomfort", "increased thirst", "frequent urination", "polyuria",
    "polydipsia", "blurred vision", "visual loss", "fainting", "open wound",
    "inability to bear weight", "knee locking", "locking", "night sweats", "rash",
]

DIAGNOSES = [
    "hypertension", "diabetes mellitus", "type 2 diabetes mellitus", "type 2 diabetes",
    "type 1 diabetes", "diabetes", "pneumonia", "asthma", "copd",
    "coronary artery disease", "heart failure", "bronchitis", "anemia",
    "pulmonary embolism", "myocardial infarction",
    "non-alcoholic fatty liver disease", "gastroesophageal reflux disease",
    "inflammatory bowel disease", "peptic ulcer disease", "acute pancreatitis",
    "chronic pancreatitis", "fatty liver disease", "colorectal carcinoma",
    "colorectal cancer", "colon cancer", "intestinal obstruction", "bowel obstruction",
    "irritable bowel syndrome", "ulcerative colitis", "crohn's disease",
    "celiac disease", "coeliac disease", "diverticular disease", "diverticulitis",
    "gastroenteritis", "pancreatitis", "gastritis", "duodenal ulcer", "gastric ulcer",
    "peptic ulcer", "hepatitis", "cirrhosis", "liver failure", "cholecystitis",
    "cholelithiasis", "gallstones", "appendicitis", "gerd", "ibs", "hemorrhoids",
    "haemorrhoids", "anal fissure",
]

MEDICATIONS = [
    "aspirin", "metformin", "paracetamol", "acetaminophen", "amoxicillin",
    "azithromycin", "atorvastatin", "amlodipine", "losartan", "ibuprofen",
    "salbutamol", "omeprazole", "insulin", "prednisone", "prednisolone",
]

PROCEDURES = [
    "surgery", "biopsy", "angiography", "angioplasty", "echocardiogram", "ecg", "ekg",
    "ct scan", "mri", "x-ray", "endoscopy", "bronchoscopy", "ultrasound",
    "upper gastrointestinal endoscopy", "colonoscopy", "sigmoidoscopy",
]

INVESTIGATIONS = [
    "white blood cell count", "white blood cell", "platelet count", "haemoglobin",
    "hemoglobin", "chest x-ray", "creatinine", "troponin", "d-dimer", "glucose", "wbc",
    "alanine aminotransferase", "aspartate aminotransferase", "gamma-glutamyl transferase",
    "liver function tests", "liver function test", "fecal occult blood",
    "faecal occult blood", "stool calprotectin", "fecal calprotectin",
    "faecal calprotectin", "helicobacter pylori", "upper gastrointestinal endoscopy",
    "abdominal ultrasound", "abdominal ct", "abdominal mri", "serum amylase",
    "serum lipase", "alkaline phosphatase", "bilirubin", "amylase", "lipase", "albumin",
    "alt", "ast", "lft", "alp", "ggt", "stool culture", "stool examination", "stool test",
    "h. pylori", "colonoscopy", "sigmoidoscopy",
    # added
    "hba1c", "sodium", "potassium", "platelets", "total bilirubin",
]

DEPARTMENT_LEXICONS: dict[str, dict[str, list[str]]] = {
    "cardiovascular_pulmonary": {
        "symptoms": [
            "chest tightness", "chest discomfort", "exertional chest pain", "orthopnea",
            "paroxysmal nocturnal dyspnea", "syncope", "near syncope", "leg swelling",
            "peripheral edema", "sputum production", "productive cough", "dry cough",
            "pleuritic chest pain", "tachypnea", "hypoxia", "cyanosis", "hemoptysis",
            "shortness of breath", "dyspnea", "wheezing", "palpitations", "chest pain",
            "cough", "breathlessness",
        ],
        "diagnoses": [
            "chronic obstructive pulmonary disease", "coronary artery disease",
            "ischemic heart disease", "ischaemic heart disease", "atrial fibrillation",
            "atrial flutter", "arrhythmia", "angina", "myocardial infarction",
            "cardiomyopathy", "pericarditis", "endocarditis", "aortic stenosis",
            "deep vein thrombosis", "peripheral arterial disease", "pulmonary hypertension",
            "pleural effusion", "pneumothorax", "respiratory failure", "pulmonary fibrosis",
            "interstitial lung disease", "pulmonary embolism", "heart failure",
            "hypertension", "pneumonia", "asthma", "bronchitis", "copd",
        ],
        "investigations": [
            "b-type natriuretic peptide", "brain natriuretic peptide", "nt-probnp",
            "high-sensitivity troponin", "troponin", "ck-mb", "creatine kinase",
            "electrocardiogram", "echocardiography", "holter monitor", "arterial blood gas",
            "pulmonary function test", "spirometry", "peak expiratory flow",
            "oxygen saturation", "d-dimer", "chest x-ray", "ecg", "ekg",
        ],
        "procedures": [
            "cardiac catheterization", "coronary angiography", "coronary angioplasty",
            "stress test", "holter monitoring", "thoracentesis", "pleural biopsy",
            "echocardiogram", "angiography", "angioplasty", "ecg", "ekg",
        ],
        "findings": [
            "heart murmur", "gallop rhythm", "irregular rhythm", "jugular venous distension",
            "raised jugular venous pressure", "delayed capillary refill",
            "reduced breath sounds", "decreased breath sounds", "crackles", "rales",
            "wheezes", "rhonchi", "stridor", "pleural rub", "pursed-lip breathing",
            "cardiomegaly", "pulmonary infiltrate", "consolidation", "atelectasis",
            "pneumothorax", "pulmonary edema", "pulmonary oedema", "lung nodule",
            "lung mass", "hyperinflation",
        ],
    },
    "consult_history_physical": {
        "symptoms": [
            "abdominal pain", "chest pain", "shortness of breath", "fatigue", "dizziness",
            "weakness", "fever", "nausea", "vomiting", "cough", "headache",
        ],
        "diagnoses": [
            "hypertension", "diabetes mellitus", "type 2 diabetes", "asthma", "pneumonia",
            "coronary artery disease", "heart failure",
        ],
        "medications": list(MEDICATIONS),
        "findings": [
            "abdominal tenderness", "joint tenderness", "bone tenderness", "heart murmur",
            "crackles", "wheezes", "reduced range of motion", "normal examination",
            "normal breath sounds", "soft abdomen", "no acute distress",
        ],
        "history": [],  # derived from HISTORICAL / FAMILY assertions, not from a lexicon
    },
    "gastroenterology": {
        "symptoms": [
            "right upper quadrant pain", "left lower quadrant pain", "abdominal distension",
            "abdominal cramps", "epigastric pain", "difficulty swallowing",
            "painful swallowing", "blood in stool", "rectal bleeding", "black stools",
            "vomiting blood", "loss of appetite", "unintentional weight loss",
            "abdominal pain", "constipation", "diarrhea", "diarrhoea", "bloating",
            "heartburn", "indigestion", "dysphagia", "odynophagia", "melena",
            "hematemesis", "nausea", "vomiting", "jaundice", "acid reflux", "belching",
            "upper abdominal discomfort", "epigastric discomfort",
        ],
        "diagnoses": [
            "non-alcoholic fatty liver disease", "gastroesophageal reflux disease",
            "inflammatory bowel disease", "peptic ulcer disease", "acute pancreatitis",
            "chronic pancreatitis", "colorectal carcinoma", "colorectal cancer",
            "intestinal obstruction", "irritable bowel syndrome", "ulcerative colitis",
            "crohn's disease", "celiac disease", "coeliac disease", "diverticulitis",
            "gastroenteritis", "pancreatitis", "gastritis", "hepatitis", "cirrhosis",
            "liver failure", "cholecystitis", "cholelithiasis", "gallstones",
            "appendicitis", "hemorrhoids", "haemorrhoids", "anal fissure", "gerd", "ibs",
        ],
        "investigations": [
            "alanine aminotransferase", "aspartate aminotransferase",
            "gamma-glutamyl transferase", "liver function tests", "fecal occult blood",
            "faecal occult blood", "stool calprotectin", "fecal calprotectin",
            "faecal calprotectin", "helicobacter pylori", "upper gastrointestinal endoscopy",
            "abdominal ultrasound", "abdominal ct", "abdominal mri", "serum amylase",
            "serum lipase", "alkaline phosphatase", "total bilirubin", "bilirubin",
            "amylase", "lipase", "albumin", "stool culture", "stool examination",
            "stool test", "h. pylori", "colonoscopy", "sigmoidoscopy", "alt", "ast",
            "lft", "alp", "ggt",
        ],
        "procedures": [
            "endoscopic retrograde cholangiopancreatography", "esophagogastroduodenoscopy",
            "upper gastrointestinal endoscopy", "capsule endoscopy", "liver biopsy",
            "colonoscopy", "sigmoidoscopy", "endoscopy",
        ],
        "findings": [
            "abdominal guarding", "guarding", "rebound tenderness", "abdominal tenderness",
            "hepatomegaly", "splenomegaly", "ascites", "abdominal distension",
            "abdominal rigidity", "bowel sounds",
        ],
    },
    "orthopedic": {
        "symptoms": [
            "reduced range of motion", "difficulty bearing weight", "difficulty walking",
            "inability to bear weight", "joint stiffness", "joint pain", "knee pain",
            "hip pain", "shoulder pain", "back pain", "neck pain", "limb pain",
            "bone pain", "muscle pain", "muscle weakness", "myalgia", "arthralgia",
            "limited mobility", "limping", "numbness", "tingling", "swelling", "weakness",
            "knee locking", "locking",
        ],
        "diagnoses": [
            "intervertebral disc prolapse", "chronic osteomyelitis", "osteoarthritis",
            "rheumatoid arthritis", "osteoporosis", "osteomyelitis", "scoliosis",
            "spinal stenosis", "herniated disc", "fracture", "dislocation", "tendinitis",
            "bursitis", "ligament injury", "meniscal tear",
        ],
        "investigations": [
            "bone mineral density", "dual-energy x-ray absorptiometry", "joint x-ray",
            "bone scan", "mri scan", "computed tomography", "magnetic resonance imaging",
        ],
        "procedures": [
            "joint replacement", "arthroscopy", "joint aspiration", "internal fixation",
            "spinal fusion", "joint x-ray", "bone scan", "surgery",
        ],
        "findings": [
            "joint tenderness", "bone tenderness", "crepitus", "joint deformity",
            "limited range of motion", "reduced range of motion", "decreased range of motion",
            "muscle atrophy", "muscle weakness", "antalgic gait", "abnormal gait",
            "joint instability", "spinal tenderness", "soft tissue swelling",
            "joint effusion", "effusion",
        ],
    },
    "radiology": {
        "investigations": [
            "computed tomography", "magnetic resonance imaging", "ct angiography",
            "mri scan", "chest radiograph", "chest x-ray", "chest ct", "abdominal ultrasound",
            "abdominal ct", "abdominal mri", "doppler ultrasound", "ultrasound", "x-ray",
            "mri", "ct scan", "radiograph",
        ],
        "procedures": [
            "computed tomography", "magnetic resonance imaging", "ct angiography",
            "chest radiograph", "chest x-ray", "doppler ultrasound", "ultrasound", "x-ray",
            "mri", "ct scan", "chest ct", "radiograph",
        ],
        "findings": [
            "pericardial effusion", "pleural effusion", "pulmonary infiltrate",
            "consolidation", "atelectasis", "pneumothorax", "pulmonary edema",
            "pulmonary oedema", "lung nodule", "lung mass", "cardiomegaly", "fracture",
            "dislocation", "joint effusion", "degenerative changes", "osteophyte",
            "disc herniation", "spinal stenosis", "soft tissue swelling", "opacity",
            "hyperinflation", "compression deformity",
            "no acute cardiopulmonary abnormality",
        ],
    },
}

_CAT_FROM_KEY = {
    "symptoms": "symptom", "diagnoses": "diagnosis", "medications": "medication",
    "procedures": "procedure", "investigations": "investigation", "findings": "finding",
}


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# `common` is built from the union of the global list and every department list,
# so it is always a superset of the department blocks.
_COMMON_TERMS: dict[str, tuple[str, ...]] = {}
for _cat, _glob, _key in (
    ("symptom", SYMPTOMS, "symptoms"),
    ("diagnosis", DIAGNOSES, "diagnoses"),
    ("medication", MEDICATIONS, "medications"),
    ("procedure", PROCEDURES, "procedures"),
    ("investigation", INVESTIGATIONS, "investigations"),
):
    _all = list(_glob)
    for _d in DEPARTMENT_LEXICONS.values():
        _all.extend(_d.get(_key, []))
    _COMMON_TERMS[_cat] = tuple(sorted(_dedupe(_all), key=len, reverse=True))

_DEPT_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    dept: {k: tuple(sorted(_dedupe(v), key=len, reverse=True)) for k, v in cats.items()}
    for dept, cats in DEPARTMENT_LEXICONS.items()
}

# ==================================================================
# Assertion vocabulary
# ==================================================================

STATUS_FROM_ASSERTION = {
    "PRESENT": "MENTIONED", "ABSENT": "NEGATED", "UNCERTAIN": "UNCERTAIN",
    "POLARITY_UNKNOWN": "UNCERTAIN", "UNKNOWN": "UNCERTAIN", "PLANNED": "PLANNED",
    "HISTORICAL": "HISTORICAL", "FAMILY": "MENTIONED", "NOT_PROVIDED": "NEGATED",
}
NEGATED_FROM_ASSERTION = {
    "PRESENT": False, "ABSENT": True, "NOT_PROVIDED": True, "HISTORICAL": False,
    "FAMILY": False, "PLANNED": False, "UNCERTAIN": None, "POLARITY_UNKNOWN": None,
    "UNKNOWN": None,
}
# "No family history of diabetes" is about the family, never a patient-level negation.
FAMILY_ASSERTIONS = {"FAMILY", "FAMILY_ABSENT"}
STATUS_FROM_ASSERTION["FAMILY_ABSENT"] = "NEGATED"
NEGATED_FROM_ASSERTION["FAMILY_ABSENT"] = True

_I = re.IGNORECASE

SCOPE_END = re.compile(
    r";|\.(?=\s|$)|\b(?:but|however|although|though|except|whereas|while|which|because|"
    r"since|yet)\b|\bapart from\b|\baside from\b", _I)
FINITE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|remains?|reports?|complains?|presents?|shows?|"
    r"states?|endorses?|developed|began|started)\b", _I)
_NEG_TAIL = re.compile(
    r"^\s*(?:(?:have|has|had|been|report(?:s|ed)?|complain(?:s|ed)?(?: of)?|show(?:s|n)?|"
    r"experienc(?:e|es|ing)|tak(?:e|es|ing)|us(?:e|es|ing)|any|a|an|known|currently|"
    r"presently|having|a history of)\b\s*)+", _I)

NEG_PRE = re.compile(
    r"\b(?:no|not|never|none|nil|denies|denied|denying|deny|without|negative for|"
    r"absence of|absent|free of|lack of|lacks|lacking|ruled out|"
    r"(?:does|do|did|has|have|is|are|was|were)n['’]t)\b", _I)
PSEUDO_NEG = re.compile(
    r"\b(?:no|not)\s+(?:change|longer|increase|decrease|further|only|necessarily|"
    r"limited to|certain|sure|clear|just|worse|better)\b|"
    r"\b(?:do|does|did|can|could|may)(?:\s+not|n['’]t)\s+(?:assess|evaluate|exclude|"
    r"rule out|reveal|detect|rule)\b|\bcannot\s+(?:be\s+)?(?:exclude[d]?|rule[d]? out)\b|"
    r"\bnot\s+(?:be\s+)?excluded\b|\bnot\s+ruled\s+out\b|\bwithout\s+(?:difficulty|change)\b", _I)
POST_NEG = re.compile(
    r"^\s*(?:[:\-–]\s*)?(?:(?:is|are|was|were|has been|have been|being)\s+)?"
    r"(?:(?:not|never)\s+(?:seen|present|identified|observed|noted|evident|visuali[sz]ed|"
    r"demonstrated|detected|found|elicited|appreciated)|negative\b|absent\b|excluded\b|"
    r"ruled out\b|none\b|nil\b|no\b)", _I)
POST_UNKNOWN = re.compile(
    r"^\s*(?:is|are|was|were)?\s*not\s+(?:documented|recorded|reported|mentioned|"
    r"available|provided|verified|confirmed|assessed|known)\b", _I)
UNC_PRE = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspected|suspect|suspicious for|"
    r"questionable|likely|may have|might have|may represent|could be|"
    r"cannot (?:be )?(?:excluded|exclude|rule out)|concern(?:ing)? for|"
    r"differential(?: diagnosis)? includes?|r/o|rule[s]? out|query)\b|\?", _I)
UNC_POST = re.compile(
    r"^\s*(?:is|are|was|were|cannot be|can't be)\s+(?:possible|suspected|likely|probable|"
    r"questionable|not excluded|not ruled out|excluded)\b|^\s*cannot be excluded\b", _I)

# ---- family history -------------------------------------------------
_REL = (r"(?:mother|father|sister|brother|siblings?|parents?|grandmother|grandfather|"
        r"grandparents?|aunt|uncle|cousins?|son|daughter|twin|maternal|paternal|relatives|"
        r"family members?|(?:first|second)[- ]degree relatives?)")
# deliberately NOT included: "child/children" (paediatric patient) and bare "relative" ("relative bradycardia")
FAMILY_PRE = re.compile(rf"\b(?:family history of|fh of|fhx of|{_REL}(?:['’]s)?)\b", _I)
FAMILY_POST = re.compile(
    rf"^\s*(?:(?:in|of|among)\s+(?:(?:the|his|her|their|a|patient['’]s)\s+)?{_REL}\b"
    rf"|[(:\-–]\s*{_REL}\b"
    rf"|(?:runs?|running)\s+in\s+(?:the\s+|his\s+|her\s+)?family\b"
    rf"|in\s+(?:the\s+)?family\b)", _I)
# "Mother reports the child has fever" must stay a patient symptom
_REPORTER = re.compile(
    r"\b(?:reports?|reported|states?|stated|says?|said|notes?|noted|complain\w*|brought|concern\w*)\b", _I)
_PATIENT_CUE = re.compile(
    r"\b(?:patient|pt|currently|denies|presents?|presented|complains?|today|on examination)\b", _I)

PLAN_PRE = re.compile(
    r"\b(?:if|should|plan(?:ned|s)?\s+(?:to|for)|scheduled\s+for|will\s+(?:undergo|have|need)|"
    r"recommend(?:ed)?|consider(?:ing)?|awaiting|pending|referred\s+for|referral\s+for|"
    r"to\s+rule\s+out|in\s+case\s+of|return\s+if|to\s+assess\s+for|to\s+evaluate)\b", _I)
HIST_PRE = re.compile(
    r"\b(?:history of|hx of|h/o|previous(?:ly)?|prior(?!\s+to\b)|formerly|remote|"
    r"in the past|status post|s/p|past medical history|past surgical history)\b", _I)
HIST_POST = re.compile(r"\b(?:resolved|in remission|\d+\s+(?:years?|months?)\s+ago|last year)\b", _I)

SEVERITY_RE = re.compile(
    r"\b(mild|moderate|severe|marked|slight|minimal|significant|profound)\b", _I)
LATERALITY_RE = re.compile(r"\b(right|left|bilateral(?:ly)?)\b", _I)
_NUM_WORD = r"(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|few)"
DURATION_RE = re.compile(
    rf"\b(?:for|over|x)\s+(?:the\s+(?:past|last)\s+)?"
    rf"((?:approximately|about|around|roughly|nearly|~)?\s*{_NUM_WORD}"
    rf"(?:\s*[-–]\s*{_NUM_WORD})?\s*(?:day|week|month|year|hour)s?)\b", _I)

# ==================================================================
# Sentence segmentation
# ==================================================================

_FUNC_WORDS = {
    "or", "and", "of", "to", "with", "for", "the", "a", "an", "in", "on", "at", "by",
    "than", "via", "including", "nor", "is", "are", "was", "were", "not", "require",
    "requires", "oxygen", "from",
}
_SMALL = {"of", "and", "the", "with", "for", "to", "in", "on", "a", "an", "or", "at"}
BULLET_RE = re.compile(r"^\s*(?:[•●▪◦*]|[-–]\s|\d{1,2}[.)]\s)\s*")
_ABBREV = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|vs|approx|Fig|cf|ca|incl|appt)\.|\be\.g\.|\bi\.e\.|"
    r"\bH\.(?=\s*pylori)|\b[A-Z]\.(?=\s+[a-z])", _I)
_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9•●▪◦*\"(\[-])|\n+|\s+(?=[•●▪◦]\s)")

_SECTION_DEFS: list[tuple[str, str]] = [
    ("hpi", r"history of present illness|hpi|present illness|presenting complaints?|chief complaints?|reason for visit"),
    ("family_history", r"family history"),
    ("social_history", r"social history"),
    ("past_history", r"past medical history|past surgical history|surgical history|medical history|clinical history|pmh|history"),
    ("allergies", r"allerg(?:y|ies)(?: information| history| and reactions)?"),
    ("medications", r"current medications?(?: information)?|medications?(?: list| information| reconciliation)?|home medications?|medication reconciliation"),
    ("vitals", r"vital signs|vitals"),
    ("exam", r"physical examination|examination|systems? review|review of systems|exam"),
    ("indication", r"clinical question|clinical indication|indications?|reason for (?:exam(?:ination)?|study|referral|consultation)|referral reason"),
    ("comparison", r"comparisons?"),
    ("technique", r"technique|protocol"),
    ("plan", r"assessment and plan|plan|recommendations?|next steps|follow[- ]up|advice|referrals?|management|instructions"),
    ("impression", r"impression|assessment|diagnos[ie]s|summary|conclusion"),
    ("results", r"investigations?|results?|laboratory(?: results| findings)?|lab results|labs|imaging|findings|radiology findings|diagnostic (?:tests|results)"),
]
_SECTION_HEAD = [(k, re.compile(rf"^(?:{p})\s*:?\s*$", _I)) for k, p in _SECTION_DEFS]
_SECTION_PREFIX = [(k, re.compile(rf"^(?:{p})\s*[:\-–]\s*(?=\S)", _I)) for k, p in _SECTION_DEFS]
_TRANSIENT_SECTIONS = {"indication", "comparison", "technique"}

PLAN_START = re.compile(
    r"^(?:review|confirm|escalate|assess|consider|document|ensure|monitor|obtain|arrange|"
    r"refer|repeat|recommend|advise|counsel|check|evaluate|order|perform|schedule|"
    r"follow[- ]up|discuss|verify|correlate|clarify|continue|start|initiate|stop|avoid|"
    r"return|seek|screen|educate|encourage|investigate|plan|exclude|determine|update)\b",
    _I)
PLAN_MODAL = re.compile(r"\b(?:should|must|recommended|is advised|are advised|advised to)\b|\brequires?\s*$", _I)
REFERRAL_RE = re.compile(
    r"\b(?:cardiology|pulmonology|gastroenterology|orthopedics?|orthopaedics?|radiology|"
    r"neurology|endocrinology|nephrology|oncology|surgical)\s+"
    r"(?:assessment|evaluation|review|referral|follow[- ]up)\b", _I)
INDICATION_START = re.compile(
    r"^(?:clinical question|clinical indication|indications?|reason for (?:exam|examination|study|referral)|referral reason)\b", _I)
BOILERPLATE_RE = re.compile(
    r"\b(?:radiographs?|x-?rays?|imaging|scans?|ct|mri|ultrasound)\b[^.]{0,40}"
    r"\b(?:do(?:es)? not|cannot|may not|have limited)\b[^.]{0,30}"
    r"\b(?:assess|evaluate|exclude|detect|visuali[sz]e)\b[^.]{0,20}"
    r"\b(?:all|every|fully|entire|soft[- ]tissue|ligament\w*)\b|"
    r"\bfor (?:educational|illustrative|demonstration) purposes\b|"
    r"\bdoes not constitute\b|\bthis (?:synthetic|sample|illustrative) (?:report|document)\b", _I)

NKDA_RE = re.compile(
    r"\bnkda\b|\bnka\b|\bno\s+(?:known|reported|documented)\s+(?:(?:drug|medication|food|environmental)\s+)*allerg|"
    r"\bno\s+(?:drug|medication|food)\s+allerg|\bdenies\s+(?:any\s+)?(?:(?:drug|medication|food)\s+)*allerg|"
    r"\ballerg(?:y|ies)\s*[:\-]\s*(?:none|nil|nka|no)\b", _I)
_DATA_NOUN = re.compile(
    r"\b(?:list|histor(?:y|ies)|status|readings?|results?|information|records?|details|data|"
    r"imaging|stud(?:y|ies)|measurements?|investigations?|tests?)\b", _I)
_G1 = re.compile(
    r"\b(?:unconfirmed|unverified|undocumented|not\s+(?:yet\s+)?(?:been\s+)?(?:provided|available|"
    r"verified|documented|confirmed|supplied|obtained|submitted|specified|on file|assessed|recorded))\b", _I)
_G2 = re.compile(r"^(?:no|not)\b.{0,80}?\b(?:provided|available|supplied|submitted|attached|on file)\b", _I)
_G3 = re.compile(r"\b(?:is|are|was|were)\s+not\s*$", _I)
_G4 = re.compile(
    r"^no\b.{0,60}\b(?:readings?|results?|list|stud(?:y|ies)|records?|measurements?)\b.*\b(?:are|is|were|was)\s*$", _I)
_PROC_NOUN = re.compile(
    r"\b(?:imaging|endoscopy|colonoscopy|biops(?:y|ies)|scan|mri|ct|x-?ray|radiograph|"
    r"ultrasound|testing|tests?|studies|study|workup|investigations?|labs?)\b", _I)
_G5_VERB = re.compile(r"\b(?:documented|recorded|reported|performed|done|obtained)\b", _I)
ALLERGY_WORD = re.compile(r"\ballerg", _I)


def _heading_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith(":"):
        return True
    if re.search(r"[.!?]$", s):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/&-]*", s)
    if not words or len(words) > 6 or re.search(r"\d", s):
        return False
    return all(w[0].isupper() or w.lower() in _SMALL for w in words)


def _flatten(text: str) -> tuple[str, str]:
    """Rejoin OCR line wraps (length-preserving, so offsets stay valid).

    Returns (flat, protected): `protected` additionally hides abbreviation dots so the
    sentence splitter does not cut at 'H. pylori' or 'Dr. Smith'.
    """
    lines = text.split("\n")
    chars = list(text)
    offset = 0
    for i, line in enumerate(lines[:-1]):
        prev, nxt = line.strip(), lines[i + 1].strip()
        end = offset + len(line)
        offset = end + 1
        if not prev or not nxt or prev[-1] in ".!?:;":
            continue
        if BULLET_RE.match(nxt) or _heading_like(prev) or _heading_like(nxt):
            continue
        last_word = re.findall(r"[A-Za-z]+", prev[-20:])
        if (nxt[0].islower() or nxt[0].isdigit() or prev[-1] in ",-(/"
                or (last_word and last_word[-1].lower() in _FUNC_WORDS)):
            chars[end] = " "
    flat = "".join(chars)
    prot = list(flat)
    for m in _ABBREV.finditer(flat):
        for k in range(m.start(), m.end()):
            if prot[k] == ".":
                prot[k] = "\u2024"
    return flat, "".join(prot)


@dataclass(frozen=True)
class Sent:
    idx: int
    text: str
    start: int
    end: int
    role: str
    subtype: str
    section: str | None
    flags: tuple[str, ...]

    @property
    def evidence(self) -> str:
        return re.sub(r"\s+", " ", self.text).strip()

    def span(self) -> list[int]:
        return [self.start, self.end]


def _gap_subtype(text: str, section: str | None) -> str | None:
    if NKDA_RE.search(text):
        return None
    gap = bool(
        _G1.search(text) or _G2.match(text) or _G4.match(text)
        or (_DATA_NOUN.search(text) and _G3.search(text))
        or (re.match(r"^(?:no|not)\b", text, _I) and _PROC_NOUN.search(text) and _G5_VERB.search(text))
    )
    if not gap:
        return None
    if ALLERGY_WORD.search(text):
        return "allergy"
    if section == "comparison" or re.search(r"\bprior (?:study|studies|imaging|exam)\b|\bcomparison\b", text, _I):
        return "comparison"
    if re.search(r"\breadings?\b|\bresults?\b|\bimaging\b|\bendoscopy\b|\bstud(?:y|ies)\b|\btests?\b|\bmri\b|\bscan\b|\blabs?\b", text, _I):
        return "investigation"
    if re.search(r"\bhistor", text, _I):
        return "history"
    if re.search(r"\bmedication|prescription", text, _I):
        return "medication"
    return "other"


def _classify(text: str, section: str | None) -> tuple[str, str]:
    if BOILERPLATE_RE.search(text) or section == "technique":
        return "boilerplate", ""
    if section == "indication" or INDICATION_START.match(text):
        return "indication", ""
    if section == "plan" or PLAN_START.match(text) or REFERRAL_RE.search(text) or PLAN_MODAL.search(text):
        return "plan", ""
    gap = _gap_subtype(text, section)
    if gap:
        return "data_gap", gap
    return "finding", ""


def _segment(text: str) -> tuple[Sent, ...]:
    t = text.replace("\r\n", " \n").replace("\r", "\n")
    flat, prot = _flatten(t)
    sents: list[Sent] = []
    persistent: str | None = None
    pos = 0
    bounds = [(m.start(), m.end()) for m in _SPLIT.finditer(prot)]
    segments, cursor = [], 0
    for s, e in bounds:
        segments.append((cursor, s))
        cursor = e
    segments.append((cursor, len(flat)))

    for a, b in segments:
        seg = flat[a:b]
        lead = len(seg) - len(seg.lstrip())
        a += lead
        seg = flat[a:b].rstrip()
        b = a + len(seg)
        if not seg:
            continue
        bm = BULLET_RE.match(seg)
        if bm:
            a += bm.end()
            seg = flat[a:b]
            if not seg.strip():
                continue
        had_prefix = False
        transient: str | None = None
        # bare heading line
        head = None
        for key, rx in _SECTION_HEAD:
            if rx.match(seg.strip()):
                head = key
                break
        if head:
            if head in _TRANSIENT_SECTIONS:
                transient = head
            else:
                persistent = head
            sents.append(Sent(len(sents), seg.strip(), a, b, "heading", "", head, ()))
            continue
        for key, rx in _SECTION_PREFIX:
            m = rx.match(seg)
            if m:
                a += m.end()
                seg = flat[a:b]
                had_prefix = True
                if key in _TRANSIENT_SECTIONS:
                    transient = key
                else:
                    persistent = key
                break
        section = transient or persistent
        role, subtype = _classify(seg, section)
        words = seg.split()
        flags = []
        if len(words) <= 8:
            flags.append("list_like")
        if (not had_prefix and len(words) >= 3
                and (seg[0].islower() or re.match(r"(?:or|and|nor|but)\b", seg))):
            flags.append("fragment_start")
        last = re.findall(r"[A-Za-z]+", seg[-20:])
        if seg[-1] not in ".!?:" and last and (last[-1].lower() in _FUNC_WORDS or seg[-1] == ","):
            flags.append("truncated_end")
        sents.append(Sent(len(sents), seg, a, b, role, subtype, section, tuple(flags)))
        pos = b
    return tuple(sents)


def _sentences(text: str) -> list[str]:
    """Backward-compatible helper: sentence strings only."""
    return [s.evidence for s in _segment(text) if s.role != "heading"]


# ==================================================================
# Term matching and assertion engine
# ==================================================================

@lru_cache(maxsize=4096)
def _term_re(term: str) -> re.Pattern:
    body = r"\s+".join(re.escape(w) for w in term.split())
    return re.compile(rf"(?<![\w]){body}(?:s|es)?(?!\w)", _I)


def _find_terms(text: str, terms: tuple[str, ...]) -> list[tuple[str, int, int]]:
    taken: list[tuple[int, int]] = []
    hits: list[tuple[str, int, int]] = []
    for term in terms:  # already sorted longest first
        for m in _term_re(term).finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            hits.append((term, s, e))
    hits.sort(key=lambda h: h[1])
    return hits


def _cue_before(pre: str, rx: re.Pattern, max_tokens: int) -> bool:
    ms = list(rx.finditer(pre))
    if not ms:
        return False
    between = pre[ms[-1].end():]
    return len(between.split()) <= max_tokens and not SCOPE_END.search(between)


def _cue_after(post: str, rx: re.Pattern, max_pre_tokens: int = 4) -> bool:
    m = rx.search(post[:80])
    if not m:
        return False
    gap = post[:m.start()]
    return len(gap.split()) <= max_pre_tokens and not SCOPE_END.search(gap)


def _family_before(pre: str) -> bool:
    """Family cue before the term, unless a reporter verb shows the relative is only the source."""
    ms = list(FAMILY_PRE.finditer(pre))
    if not ms:
        return False
    between = pre[ms[-1].end():]
    return (len(between.split()) <= 8 and not SCOPE_END.search(between)
            and not _REPORTER.search(between))


def _neg_before(pre: str) -> bool:
    cues = list(NEG_PRE.finditer(pre))
    if not cues:
        return False
    pseudo = [m.span() for m in PSEUDO_NEG.finditer(pre)]
    for m in reversed(cues):
        if any(ps <= m.start() < pe for ps, pe in pseudo):
            continue
        between = _NEG_TAIL.sub(" ", pre[m.end():], count=1)
        if SCOPE_END.search(between) or FINITE.search(between):
            return False
        return len(between.split()) <= 20
    return False


def _assertion(sent: Sent, s: int, e: int, category: str) -> str:
    text = sent.text
    pre, post = text[:s], text[e:]
    # Family first, so "no family history of X" is not read as a patient-level negation.
    if (_family_before(pre) or FAMILY_POST.match(post)
            or (sent.section == "family_history" and not _PATIENT_CUE.search(text))):
        return "FAMILY_ABSENT" if (_neg_before(pre) or POST_NEG.match(post)) else "FAMILY"
    if _neg_before(pre):
        return "ABSENT"
    if POST_NEG.match(post) and not SCOPE_END.search(post[:0]):
        return "ABSENT"
    if POST_UNKNOWN.match(post):
        return "UNKNOWN"
    if _cue_before(pre, UNC_PRE, 8) or UNC_POST.match(post):
        return "UNCERTAIN"
    if _cue_before(pre, PLAN_PRE, 8):
        return "PLANNED"
    if (_cue_before(pre, HIST_PRE, 6) or _cue_after(post, HIST_POST)
            or (sent.section == "past_history" and "list_like" in sent.flags)):
        return "HISTORICAL"
    if "fragment_start" in sent.flags and category in ("symptom", "diagnosis", "finding", "procedure"):
        return "POLARITY_UNKNOWN"
    return "PRESENT"


# ==================================================================
# Attributes: dates, duration, severity, laterality, medication details
# ==================================================================

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
DATE_RE = re.compile(
    rf"\b(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|"
    rf"\d{{1,2}}\s+(?:{_MONTHS})[a-z]*\.?,?\s+\d{{4}}|(?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s+\d{{4}})\b", _I)


def _parse_date(raw: str) -> str | None:
    raw = raw.strip().rstrip(",")
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw.replace(".", ""), fmt).date().isoformat()
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        y += 2000 if y < 100 else 0
        try:
            if a > 12 >= b:
                return datetime(y, b, a).date().isoformat()
            if b > 12 >= a:
                return datetime(y, a, b).date().isoformat()
        except ValueError:
            return None
    return None  # ambiguous or unparseable: caller keeps the raw string


def _sentence_date(sent: Sent) -> tuple[str | None, str | None]:
    found = DATE_RE.findall(sent.text)
    if len(found) != 1:
        return None, None
    return _parse_date(found[0]), found[0]


# A modifier only belongs to a term if it sits in the same clause, right before the term.
_CLAUSE_BREAK = re.compile(
    r"[,;:()]|\b(?:and|or|with|but|plus|then|while|also|including|as well as)\b", _I)


def _adjacent_modifier(pre: str, rx: re.Pattern, max_words: int = 3) -> str | None:
    seg = _CLAUSE_BREAK.split(pre)[-1]  # text after the last clause break
    hits = rx.findall(" ".join(seg.split()[-max_words:]))
    return hits[-1].lower() if hits else None


def _clinical_attrs(sent: Sent, s: int, e: int) -> dict:
    pre, post = sent.text[:s], sent.text[e:]
    out: dict = {}
    sev = _adjacent_modifier(pre, SEVERITY_RE, 3)
    if sev:
        out["severity"] = sev
    lat = _adjacent_modifier(pre, LATERALITY_RE, 2)
    if lat:
        out["laterality"] = lat
    m = DURATION_RE.search(post[:60])
    if m and not re.search(r"[,;]|\band\b", post[:m.start()]):
        out["duration"] = re.sub(r"\s+", " ", m.group(1)).strip()
    if sent.section == "family_history" or _family_before(pre):
        out["experiencer"] = "family"
    return out


_DOSE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|iu|units?|puffs?)\b", _I)
_ROUTE = re.compile(r"\b(oral(?:ly)?|po|iv|im|sc|subcutaneous(?:ly)?|inhaled|inhaler|topical|sublingual)\b", _I)
_FREQ = re.compile(
    r"\b(?:once|twice|three times|four times)\s+(?:a|per)\s+day\b|\b(?:once|twice)\s+daily\b|"
    r"\b(?:daily|nightly|weekly|bid|tid|qid|prn|as needed|at night|in the morning)\b|"
    r"\bevery\s+\d+\s*(?:hours?|h)\b|\b(?:intermittent(?:ly)?|occasional(?:ly)?)\b", _I)
_ADHERENCE = re.compile(
    r"\b(inconsistent(?:ly)?\s+(?:adherence|use)|poor\s+adherence|non-?adherence|"
    r"non-?complian\w+|missed\s+(?:doses|dosing)|stopped\s+taking|stopped|discontinued)"
    r"(?:\s+(?:to|with|of))?\s*(?:the\s+)?", _I)
_ROUTE_NORM = {"orally": "oral", "po": "oral", "inhaler": "inhaled", "subcutaneously": "subcutaneous", "sc": "subcutaneous"}


def _med_attrs(sent: Sent, mentions: list["Mention"]) -> None:
    mentions.sort(key=lambda m: m.s)
    for i, m in enumerate(mentions):
        limit = mentions[i + 1].s if i + 1 < len(mentions) else len(sent.text)
        window = sent.text[m.e:min(limit, m.e + 90)]
        window = re.split(r"[.;,]|\band\b", window)[0]
        d = _DOSE.search(window)
        if d:
            m.extra["dose"] = f"{d.group(1)} {d.group(2).lower()}"
        r = _ROUTE.search(window)
        if r:
            rv = r.group(1).lower()
            m.extra["route"] = _ROUTE_NORM.get(rv, rv)
        f = _FREQ.search(window)
        if f:
            m.extra["frequency"] = f.group(0).lower()
        else:
            pf = re.search(r"\b(intermittent|occasional)(?:ly)?\s+(?:use\s+of\s+)?(?:the\s+)?$",
                           sent.text[max(0, m.s - 30):m.s], _I)
            if pf:
                m.extra["frequency"] = pf.group(1).lower()
    for cue in _ADHERENCE.finditer(sent.text):
        for m in mentions:
            if 0 <= m.s - cue.end() <= 25:
                label = cue.group(1).lower()
                m.extra["adherence"] = (
                    "inconsistent" if "inconsistent" in label else
                    "stopped" if ("stopped" in label or "discontinued" in label) else
                    "missed doses" if "missed" in label else "poor")
                break


# ==================================================================
# Structured values: labs, vitals, spirometry, LVEF
# ==================================================================

# surface form -> (canonical name, default unit or None, qualifier or None)
_LAB_SURFACES: dict[str, tuple[str, str | None, str | None]] = {}


def _lab(canon: str, surfaces: list[str], default_unit: str | None = None):
    for s in surfaces:
        q = s.replace(canon, "").strip() or None if canon in s else None
        _LAB_SURFACES[s] = (canon, default_unit, q)


_lab("white blood cell count", ["white blood cell count", "white blood cells", "white blood cell", "wbc", "leukocyte count"])
_lab("hemoglobin", ["hemoglobin", "haemoglobin", "hb"])
_lab("platelet count", ["platelet count", "platelets", "platelet", "plt"])
_lab("hba1c", ["hba1c", "hemoglobin a1c", "haemoglobin a1c", "glycated hemoglobin", "a1c"], "%")
_lab("sodium", ["sodium", "serum sodium"], "mmol/L")
_lab("potassium", ["potassium", "serum potassium"], "mmol/L")
_lab("creatinine", ["creatinine", "serum creatinine"])
_lab("urea", ["blood urea nitrogen", "bun", "urea"])
_lab("glucose", ["random plasma glucose", "fasting plasma glucose", "fasting glucose",
                 "random glucose", "blood glucose", "plasma glucose", "serum glucose", "glucose"])
_lab("high-sensitivity troponin", ["high-sensitivity troponin", "hs-troponin", "hs troponin"])
_lab("troponin", ["troponin"])
_lab("d-dimer", ["d-dimer"])
_lab("alt", ["alanine aminotransferase", "alt"], "U/L")
_lab("ast", ["aspartate aminotransferase", "ast"], "U/L")
_lab("alkaline phosphatase", ["alkaline phosphatase", "alp"], "U/L")
_lab("ggt", ["gamma-glutamyl transferase", "ggt"], "U/L")
_lab("lipase", ["serum lipase", "lipase"], "U/L")
_lab("amylase", ["serum amylase", "amylase"], "U/L")
_lab("total bilirubin", ["total bilirubin"])
_lab("bilirubin", ["bilirubin"])
_lab("albumin", ["albumin"])
_lab("crp", ["c-reactive protein", "crp"])
_lab("esr", ["esr"])
_lab("tsh", ["tsh"])
_lab("inr", ["inr"])
_lab("egfr", ["egfr"])
_lab("total cholesterol", ["total cholesterol"])
_lab("ldl cholesterol", ["ldl cholesterol", "ldl"])
_lab("hdl cholesterol", ["hdl cholesterol", "hdl"])
_lab("triglycerides", ["triglycerides"])

_LAB_NAME_RE = "|".join(
    r"\s+".join(re.escape(w) for w in s.split()) for s in sorted(_LAB_SURFACES, key=len, reverse=True))
_LAB_UNIT = (r"mg/dl|g/dl|mmol/l|mmol/mol|meq/l|ng/l|ng/ml|pg/ml|u/l|iu/l|µmol/l|umol/l|µg/l|ug/l|mg/l|"
             r"cells?\s*/\s*[µu]l|10\^?[369]\s*/\s*[µu]?l|x\s*10\^?\d+\s*/\s*l|%|ml/min(?:/1\.73\s*m2)?|seconds|sec")
LAB_RE = re.compile(
    rf"(?<![\w-])(?P<name>{_LAB_NAME_RE})\s*(?:[:=]|\bof\b|\bis\b|\bwas\b)?\s*"
    rf"(?P<val>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)\s*(?P<unit>{_LAB_UNIT})?"
    rf"(?:\s*\(\s*(?:ref(?:erence)?(?:\s*range)?\s*[:=]?\s*)?(?P<ref>\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)[^)]*\))?", _I)
_UNIT_CANON = {
    "mg/dl": "mg/dL", "g/dl": "g/dL", "mmol/l": "mmol/L", "mmol/mol": "mmol/mol", "meq/l": "mEq/L",
    "ng/l": "ng/L", "ng/ml": "ng/mL", "pg/ml": "pg/mL", "u/l": "U/L", "iu/l": "IU/L",
    "µmol/l": "µmol/L", "umol/l": "µmol/L", "µg/l": "µg/L", "ug/l": "µg/L", "mg/l": "mg/L",
}


def _canon_unit(u: str | None) -> str | None:
    if not u:
        return None
    k = re.sub(r"\s+", "", u).lower()
    if k.startswith("cell"):
        return "cells/µL"
    return _UNIT_CANON.get(k, u.strip())


def _canon_lab(term: str) -> str:
    hit = _LAB_SURFACES.get(term.lower())
    return hit[0] if hit else term.lower()


_VITAL_NAMES = {
    "blood pressure": "blood pressure", "bp": "blood pressure", "heart rate": "heart rate",
    "pulse rate": "heart rate", "pulse": "heart rate", "temperature": "temperature",
    "temp": "temperature", "respiratory rate": "respiratory rate", "resp rate": "respiratory rate",
    "rr": "respiratory rate", "spo2": "oxygen saturation", "oxygen saturation": "oxygen saturation",
    "o2 saturation": "oxygen saturation", "weight": "weight", "height": "height", "bmi": "bmi",
}
_VITAL_DEFAULT_UNIT = {"blood pressure": "mmHg", "heart rate": "bpm", "respiratory rate": "breaths/min",
                       "oxygen saturation": "%", "temperature": "°C", "weight": "kg", "height": "cm"}
VITAL_RE = re.compile(
    r"(?<![\w-])(?P<name>" + "|".join(re.escape(k) for k in sorted(_VITAL_NAMES, key=len, reverse=True)) +
    r")\s*(?:[:=]|\bof\b|\bis\b|\bwas\b)?\s*(?P<val>\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)\s*"
    r"(?P<unit>mmhg|beats\s*/\s*min|beats\s+per\s+minute|bpm|breaths\s*/\s*min|breaths\s+per\s+minute|"
    r"/min|°\s*[cf]|degrees?\s*[cf]|%|kg/m2|kg/m²|kg|cm|lbs?)?", _I)
_ECG_RATE = re.compile(r"\b(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:beats\s*/\s*min|beats\s+per\s+minute)\b", _I)


def _canon_vital_unit(name: str, unit: str | None) -> str:
    if not unit:
        return _VITAL_DEFAULT_UNIT.get(name, "")
    k = re.sub(r"\s+", "", unit).lower()
    if k in ("beats/min", "beatsperminute", "bpm", "/min") and name == "heart rate":
        return "bpm"
    if k in ("breaths/min", "breathsperminute", "/min"):
        return "breaths/min"
    if k == "mmhg":
        return "mmHg"
    if k.startswith(("°", "degree")):
        return "°" + k[-1].upper()
    return unit.strip()


_BRONCHO = re.compile(r"\b(before|pre-?)\s*bronchodilator|\b(after|post-?)\s*bronchodilator", _I)
_NUMERIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("fev1/fvc ratio", re.compile(r"\bFEV1\s*/\s*FVC\b(?:\s*ratio)?\s*(?:of|:|=|is|was)?\s*(\d*\.\d+|\d+(?:\.\d+)?)\s*(%)?", _I)),
    ("fev1", re.compile(r"(?<![/\w])FEV1\b(?!\s*/)\s*(?:of|:|=|is|was)?\s*(\d+(?:\.\d+)?)\s*(%|l\b)?\s*(predicted)?", _I)),
    ("fvc", re.compile(r"(?<![/\w])FVC\b\s*(?:of|:|=|is|was)?\s*(\d+(?:\.\d+)?)\s*(%|l\b)?\s*(predicted)?", _I)),
    ("left ventricular ejection fraction", re.compile(
        r"\b(?:left ventricular ejection fraction|lvef|ejection fraction)\b\s*(?:\(\s*(?:lvef|ef)\s*\))?\s*"
        r"(?:of|:|=|is|was)?\s*(\d+(?:\.\d+)?)\s*(%)", _I)),
]


def _numeric_records(sent: Sent) -> list[dict]:
    out = []
    taken: list[tuple[int, int]] = []
    for name, rx in _NUMERIC_PATTERNS:
        for m in rx.finditer(sent.text):
            if any(m.start() < te and ts < m.end() for ts, te in taken):
                continue
            taken.append(m.span())
            unit = m.group(2)
            if name == "fev1/fvc ratio":
                unit = "%" if unit else "ratio"
            elif unit and unit.lower() == "l":
                unit = "L"
            elif unit == "%" and len(m.groups()) >= 3 and m.group(3):
                unit = "% predicted"
            qual = None
            b = _BRONCHO.search(sent.text)
            if b:
                qual = "pre-bronchodilator" if (b.group(1) or "").lower() else "post-bronchodilator"
            out.append({"name": name, "result": m.group(1), "unit": unit, "qualifier": qual,
                        "s": m.start(), "e": m.end(), "extraction": "numeric_pattern"})
    return out


# ==================================================================
# Allergy rules
# ==================================================================

_ALLERGEN_AFTER = re.compile(r"\ballerg(?:y|ies|ic|ic reactions?)\s*(?:to|:|-|–)\s*(?P<list>[^.;]+)", _I)
_ALLERGEN_BEFORE = re.compile(r"\b(?P<sub>[A-Za-z][\w-]{2,})\s+allergy\b", _I)
_ALLERGY_STOP = {
    "drug", "medication", "food", "known", "no", "any", "reported", "documented", "confirmed",
    "unconfirmed", "status", "of", "with", "environmental", "seasonal", "history", "information",
    "details", "none", "unknown", "unspecified", "not", "is", "are", "was", "were", "available",
    "verified", "histories", "the", "and", "or", "to", "a", "an", "this", "that", "mild",
}
_REACTION_VOCAB = re.compile(
    r"\b(rash|hives|urticaria|anaphylaxis|angioedema|swelling|wheez\w*|itch\w*|vomiting|"
    r"diarrh\w+|shortness of breath|nausea)\b", _I)
_ALLERGY_UNKNOWN = re.compile(r"\b(?:unknown|unclear|not known|unconfirmed|unverified|pending)\b", _I)


def _allergies(sent: Sent) -> list[dict]:
    text = sent.text
    if not (ALLERGY_WORD.search(text) or sent.section == "allergies"):
        return []
    if re.fullmatch(r"\s*allerg\w*(?:\s+\w+){0,2}\s*[:\-]?\s*", text, _I):
        return []
    base = {"sent": sent, "s": 0, "e": len(text)}
    if NKDA_RE.search(text) or (sent.section == "allergies" and re.fullmatch(r"\s*(?:none|nil|nka|nkda|no)\.?\s*", text, _I)):
        return [{**base, "assertion": "NKDA", "substance": "unspecified", "status": "NEGATED"}]
    if (sent.role == "data_gap" and sent.subtype == "allergy") or _ALLERGY_UNKNOWN.search(text) or _G1.search(text):
        return [{**base, "assertion": "UNKNOWN", "substance": "unspecified", "status": "UNCERTAIN"}]
    if sent.role != "finding":
        return []
    items: list[str] = []
    m = _ALLERGEN_AFTER.search(text)
    negated = False
    if m:
        items = re.split(r",|\band\b|\bor\b|/", m.group("list"))
        negated = _neg_before(text[:m.start()]) or bool(re.search(r"\bnot\s+$", text[:m.start()], _I))
    elif sent.section == "allergies":
        items = re.split(r",|;|\band\b", text)
    else:
        b = _ALLERGEN_BEFORE.search(text)
        if b and b.group("sub").lower() not in _ALLERGY_STOP:
            items = [b.group("sub")]
    out = []
    for raw in items:
        reaction = None
        paren = re.search(r"\(([^)]*)\)", raw)
        rm = _REACTION_VOCAB.search(paren.group(1) if paren else raw)
        if rm:
            reaction = rm.group(1).lower()
        cleaned = re.sub(r"\([^)]*\)", " ", raw)
        cleaned = re.split(r"\b(?:causing|resulting|with|manifest\w*|reaction|[-–])\b|[-–]", cleaned, maxsplit=1)[0]
        words = [w for w in re.findall(r"[A-Za-z][\w-]*", cleaned)]
        if not words or len(words) > 4 or any(w.lower() in _ALLERGY_STOP for w in words):
            continue
        out.append({**base, "assertion": "NOT_ALLERGIC" if negated else "ALLERGIC",
                    "substance": " ".join(words).lower(),
                    "status": "NEGATED" if negated else "MENTIONED", "reaction": reaction})
    return out


# ==================================================================
# Mentions and collection
# ==================================================================

@dataclass
class Mention:
    category: str
    term: str
    sent: Sent
    s: int
    e: int
    assertion: str
    extra: dict = field(default_factory=dict)

    @property
    def abs_span(self) -> list[int]:
        return [self.sent.start + self.s, self.sent.start + self.e]

    @property
    def norm(self) -> str:
        return _canon_lab(self.term) if self.category == "investigation" else self.term.lower()

    @property
    def fact_id(self) -> str:
        raw = f"{self.norm}|{self.sent.idx}"
        return "f_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


@dataclass
class _Collected:
    sents: tuple[Sent, ...]
    common: dict[str, list[Mention]] = field(default_factory=lambda: {c: [] for c in _COMMON_TERMS})
    dept: dict[str, dict[str, list[Mention]]] = field(default_factory=dict)
    family: list[Mention] = field(default_factory=list)  # FAMILY / FAMILY_ABSENT, kept out of patient lists
    allergies: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)
    labs: list[dict] = field(default_factory=list)


def _mk(sent: Sent, cat: str, term: str, s: int, e: int, forced: str | None = None) -> Mention:
    a = forced or _assertion(sent, s, e, cat)
    m = Mention(cat, term, sent, s, e, a)
    if cat in ("symptom", "diagnosis", "finding"):
        m.extra.update(_clinical_attrs(sent, s, e))
    if cat in ("procedure", "investigation"):
        iso, raw = _sentence_date(sent)
        if iso or raw:
            m.extra["date"] = iso
            m.extra["date_raw"] = raw
    if a in FAMILY_ASSERTIONS:
        # tag who it applies to so history entities carry the relation
        m.extra["experiencer"] = "family"
        rel = re.search(rf"\b{_REL}\b", sent.text, _I)
        if rel:
            m.extra["relation"] = rel.group(0).lower()
    return m


def _split_family(ms: list[Mention]) -> tuple[list[Mention], list[Mention]]:
    keep, fam = [], []
    for m in ms:
        (fam if m.assertion in FAMILY_ASSERTIONS else keep).append(m)
    return keep, fam


@lru_cache(maxsize=8)
def _collect(text: str) -> _Collected:
    sents = _segment(text)
    col = _Collected(sents)
    col.dept = {d: {k: [] for k in cats} for d, cats in DEPARTMENT_LEXICONS.items()}
    for sent in sents:
        if sent.role not in ("finding", "data_gap"):
            continue
        col.allergies.extend(_allergies(sent))
        allergy_sentence = bool(ALLERGY_WORD.search(sent.text)) or sent.section == "allergies"
        gap = sent.role == "data_gap"

        for cat, terms in _COMMON_TERMS.items():
            if gap and cat not in ("procedure", "investigation"):
                continue
            if cat == "medication" and allergy_sentence:
                continue
            found = [_mk(sent, cat, t, s, e, "NOT_PROVIDED" if gap else None)
                     for t, s, e in _find_terms(sent.text, terms)]
            if cat == "medication" and found:
                _med_attrs(sent, found)
            found, fam = _split_family(found)
            col.family.extend(fam)
            col.common[cat].extend(found)

        for dept, cats in _DEPT_TERMS.items():
            for key, terms in cats.items():
                cat = _CAT_FROM_KEY.get(key)
                if cat is None or (gap and cat not in ("procedure", "investigation")):
                    continue
                if cat == "medication" and allergy_sentence:
                    continue
                found = [_mk(sent, cat, t, s, e, "NOT_PROVIDED" if gap else None)
                         for t, s, e in _find_terms(sent.text, terms)]
                if cat == "medication" and found:
                    _med_attrs(sent, found)
                found, _ = _split_family(found)  # family mentions live only in clinical_history
                col.dept[dept][key].extend(found)

        if gap:
            continue

        iso, raw = _sentence_date(sent)
        for m in VITAL_RE.finditer(sent.text):
            name = _VITAL_NAMES[m.group("name").lower()]
            col.vitals.append({
                "measurement": name, "value": re.sub(r"\s+", "", m.group("val")),
                "unit": _canon_vital_unit(name, m.group("unit")), "sent": sent,
                "extraction": "vital_pattern", "date": iso, "span": [sent.start + m.start(), sent.start + m.end()],
                "unit_source": "text" if m.group("unit") else "default"})
        if re.search(r"\b(?:ecg|ekg|electrocardiogram)\b", sent.text, _I):
            r = _ECG_RATE.search(sent.text)
            if r and not any(v["measurement"] == "heart rate" and v["value"] == r.group(1)
                             and v["sent"].idx == sent.idx for v in col.vitals):
                col.vitals.append({
                    "measurement": "heart rate", "value": r.group(1), "unit": "bpm", "sent": sent,
                    "extraction": "ecg_rate_pattern", "date": iso,
                    "span": [sent.start + r.start(), sent.start + r.end()], "unit_source": "text"})
        for m in LAB_RE.finditer(sent.text):
            canon, default_unit, qual = _LAB_SURFACES[re.sub(r"\s+", " ", m.group("name").lower())]
            unit = _canon_unit(m.group("unit"))
            src = "text"
            if not unit and default_unit:
                unit, src = default_unit, "default"
            col.labs.append({
                "name": canon, "result": m.group("val"), "unit": unit, "qualifier": qual,
                "reference_range": re.sub(r"\s+", "", m.group("ref")) if m.group("ref") else None,
                "sent": sent, "extraction": "lab_pattern", "date": iso, "unit_source": src,
                "span": [sent.start + m.start(), sent.start + m.end()]})
        for rec in _numeric_records(sent):
            col.labs.append({**rec, "reference_range": None, "sent": sent, "date": iso, "unit_source": "text",
                             "span": [sent.start + rec["s"], sent.start + rec["e"]]})
    return col


# ==================================================================
# Model builders
# ==================================================================

def _model(cls, **kw):
    fields = getattr(cls, "model_fields", None)
    if fields:
        kw = {k: v for k, v in kw.items() if k in fields}
    return cls(**kw)


def _eid() -> str:
    return str(uuid.uuid4())


def _group(mentions: list[Mention], key=lambda m: (m.norm, m.assertion)) -> list[list[Mention]]:
    groups: dict = {}
    for m in mentions:
        groups.setdefault(key(m), []).append(m)
    return list(groups.values())


def _evidence(group: list[Mention]) -> str:
    return " | ".join(_dedupe(m.sent.evidence for m in group))


def _spans(group: list[Mention]) -> list[dict]:
    seen, out = set(), []
    for m in group:
        k = (m.sent.idx, m.abs_span[0])
        if k not in seen:
            seen.add(k)
            out.append({"sentence_index": m.sent.idx, "span": m.abs_span, "sentence_span": m.sent.span(),
                        "role": m.sent.role, "text": m.sent.evidence})
    return out


def _first(group: list[Mention], key: str):
    for m in group:
        if m.extra.get(key):
            return m.extra[key]
    return None


def _common_kwargs(group: list[Mention]) -> dict:
    a = group[0].assertion
    return {
        "entity_id": _eid(), "evidence": _evidence(group), "assertion": a,
        "evidence_spans": _spans(group), "fact_id": group[0].fact_id,
        "evidence_role": group[0].sent.role,
        "status": STATUS_FROM_ASSERTION[a],
    }


def _merge_subsumed(groups: list[list[Mention]]) -> list[list[Mention]]:
    """Merge 'cough' into 'dry cough' (same assertion) so one specific entity remains."""
    groups = sorted(groups, key=lambda g: -len(g[0].norm))
    kept: list[list[Mention]] = []
    for g in groups:
        host = next((k for k in kept if k[0].assertion == g[0].assertion
                     and k[0].norm.endswith(" " + g[0].norm)), None)
        if host is not None:
            host.extend(g)
        else:
            kept.append(g)
    return sorted(kept, key=lambda g: (g[0].sent.idx, g[0].s))


def _build_symptoms(ms: list[Mention]):
    out = []
    for g in _merge_subsumed(_group(ms)):
        kw = _common_kwargs(g)
        a = kw["assertion"]
        out.append(_model(
            SymptomEntity, name=g[0].term, negated=NEGATED_FROM_ASSERTION[a],
            temporality=kw["status"].lower(), extraction="lexicon_rule", onset=None,
            duration=_first(g, "duration"), severity=_first(g, "severity"),
            experiencer=_first(g, "experiencer"), laterality=_first(g, "laterality"), **kw))
    return out


def _build_diagnoses(ms: list[Mention]):
    out = []
    for g in _merge_subsumed(_group(ms)):
        kw = _common_kwargs(g)
        a = kw["assertion"]
        out.append(_model(
            DiagnosisEntity, name=g[0].term, negated=NEGATED_FROM_ASSERTION[a],
            temporality=kw["status"].lower(), extraction="lexicon_rule",
            experiencer=_first(g, "experiencer"), **kw))
    return out


def _build_medications(ms: list[Mention]):
    out = []
    for g in _group(ms):
        kw = _common_kwargs(g)
        out.append(_model(
            MedicationEntity, name=g[0].term, extraction="lexicon_rule",
            dose=_first(g, "dose"), route=_first(g, "route"), frequency=_first(g, "frequency"),
            adherence=_first(g, "adherence"), start_date=None, end_date=None, **kw))
    return out


def _build_procedures(ms: list[Mention]):
    # "No endoscopy result is supplied" is a data gap, not a negated procedure.
    # It stays in metadata["data_gaps"] and in the department blocks (evidence_role="data_gap").
    ms = [m for m in ms if m.assertion != "NOT_PROVIDED"]
    out = []
    for g in _group(ms):
        kw = _common_kwargs(g)
        out.append(_model(
            ProcedureEntity, name=g[0].term, extraction="lexicon_rule", outcome=None,
            procedure_date=_first(g, "date"), date_raw=_first(g, "date_raw"), **kw))
    return out


def _vital_fact_id(measurement: str, sent_idx: int) -> str:
    return "f_" + hashlib.sha1(f"{measurement}|{sent_idx}".encode()).hexdigest()[:10]


def _build_investigations(col: _Collected):
    valued = col.labs
    valued_names = {r["name"] for r in valued}
    out, seen = [], set()
    for r in valued:
        key = (r["name"], r.get("qualifier"), r["result"], r["sent"].idx)
        if key in seen:
            continue
        seen.add(key)
        sent = r["sent"]
        out.append(_model(
            InvestigationEntity, entity_id=_eid(), name=r["name"], result=r["result"], unit=r["unit"],
            reference_range=r["reference_range"], date=r["date"], evidence=sent.evidence,
            extraction=r["extraction"], assertion="PRESENT", status="MENTIONED",
            qualifier=r.get("qualifier"), unit_source=r.get("unit_source"),
            evidence_spans=[{"sentence_index": sent.idx, "span": r["span"],
                             "sentence_span": sent.span(), "role": sent.role, "text": sent.evidence}],
            fact_id=_vital_fact_id(r["name"], sent.idx),
            evidence_role=sent.role))
    # Skip mentions already captured as a vital sign (e.g. oxygen saturation) and data gaps.
    vital_ids = {_vital_fact_id(v["measurement"], v["sent"].idx) for v in col.vitals}
    mentions = [m for m in col.common["investigation"]
                if m.norm not in valued_names
                and m.fact_id not in vital_ids
                and m.assertion != "NOT_PROVIDED"]
    for g in _group(mentions):
        kw = _common_kwargs(g)
        out.append(_model(
            InvestigationEntity, name=g[0].norm, result=None, unit=None, reference_range=None,
            date=_first(g, "date"), extraction="lexicon_rule", value_missing=True, **kw))
    return out


def _build_vitals(col: _Collected):
    out, seen = [], set()
    for v in col.vitals:
        key = (v["measurement"], v["value"], v["sent"].idx)
        if key in seen:
            continue
        seen.add(key)
        s = v["sent"]
        out.append(_model(
            VitalSignEntity, entity_id=_eid(), measurement=v["measurement"], value=v["value"],
            unit=v["unit"], evidence=s.evidence, extraction=v["extraction"], reference_range=None,
            measurement_date=v["date"], unit_source=v["unit_source"], assertion="PRESENT",
            evidence_spans=[{"sentence_index": s.idx, "span": v["span"], "sentence_span": s.span(),
                             "role": s.role, "text": s.evidence}],
            fact_id=_vital_fact_id(v["measurement"], s.idx)))
    return out


def _build_allergies(col: _Collected):
    out, seen = [], set()
    for a in col.allergies:
        key = (a["assertion"], a["substance"])
        if key in seen:
            continue
        seen.add(key)
        s = a["sent"]
        out.append(_model(
            AllergyEntity, entity_id=_eid(), substance=a["substance"], status=a["status"],
            reaction=a.get("reaction"), severity=None, evidence=s.evidence, extraction="allergy_rule",
            temporality=None, assertion=a["assertion"], evidence_role=s.role,
            evidence_spans=[{"sentence_index": s.idx, "span": s.span(), "sentence_span": s.span(),
                             "role": s.role, "text": s.evidence}]))
    return out


def _allergy_status(col: _Collected) -> dict:
    """Document-level allergy status, so an empty list is never ambiguous."""
    kinds = {a["assertion"] for a in col.allergies}
    if "ALLERGIC" in kinds:
        status = "CONFLICTING" if "NKDA" in kinds else "ALLERGY_REPORTED"
    elif "NKDA" in kinds:
        status = "NONE_KNOWN"                # explicit NKDA / "no known allergies"
    elif "UNKNOWN" in kinds:
        status = "UNKNOWN_DOCUMENTED"        # note says status is unconfirmed or unavailable
    elif "NOT_ALLERGIC" in kinds:
        status = "SPECIFIC_ALLERGEN_DENIED"  # e.g. "no penicillin allergy": not an NKDA
    else:
        status = "NOT_ADDRESSED"             # note never mentions allergies
    return {
        "status": status,
        "substances": sorted({a["substance"] for a in col.allergies if a["assertion"] == "ALLERGIC"}),
        "evidence": _dedupe(a["sent"].evidence for a in col.allergies),
    }


def _history_type(m: Mention) -> str:
    if m.assertion in FAMILY_ASSERTIONS:
        return "family"
    return {"procedure": "surgical", "diagnosis": "past_medical"}.get(m.category, "past_symptom")


def _history_mentions(col: _Collected) -> list[Mention]:
    out = [m for cat in ("symptom", "diagnosis", "procedure", "medication")
           for m in col.common[cat] if m.assertion == "HISTORICAL"]
    # Only affirmed family conditions become history; FAMILY_ABSENT ("no family history of X") is dropped.
    out += [m for m in col.family if m.assertion == "FAMILY"]
    return sorted(out, key=lambda m: (m.sent.idx, m.s))


def _build_history(col: _Collected):
    out, seen = [], set()
    for m in _history_mentions(col):
        if m.sent.idx in seen:
            continue
        seen.add(m.sent.idx)
        iso, raw = _sentence_date(m.sent)
        out.append(_model(
            ClinicalHistoryEntity, entity_id=_eid(), history_type=_history_type(m),
            description=m.sent.evidence, status="HISTORICAL", temporality="historical",
            evidence=m.sent.evidence, extraction="history_rule", date=iso, date_raw=raw,
            assertion=m.assertion, evidence_spans=_spans([m]), fact_id=m.fact_id,
            relation=_first([m], "relation"), experiencer=_first([m], "experiencer")))
    return out


# ==================================================================
# Public API
# ==================================================================

def extract_clinical_entities(text: str) -> ClinicalExtraction:
    """Extract candidate clinical entities from OCR text (preliminary; needs human review)."""
    result = ClinicalExtraction()
    if not isinstance(text, str) or not text.strip():
        return result
    col = _collect(text)
    result.symptoms = _build_symptoms(col.common["symptom"])
    result.diagnoses = _build_diagnoses(col.common["diagnosis"])
    result.medications = _build_medications(col.common["medication"])
    result.procedures = _build_procedures(col.common["procedure"])
    result.investigations = _build_investigations(col)
    result.vital_signs = _build_vitals(col)
    result.allergies = _build_allergies(col)
    result.clinical_history = _build_history(col)
    return result


def _dept_item(m: Mention, category: str, col: _Collected) -> dict:
    a = m.assertion
    st = STATUS_FROM_ASSERTION[a]
    item = {
        "name": m.term, "category": category, "status": st, "negated": NEGATED_FROM_ASSERTION[a],
        "temporality": st.lower(), "evidence": m.sent.evidence, "extraction": "department_lexicon_rule",
        "assertion": a, "sentence_index": m.sent.idx, "span": m.abs_span,
        "evidence_role": m.sent.role, "fact_id": m.fact_id,
    }
    item.update({k: v for k, v in m.extra.items() if v})
    if m.category == "investigation":
        for r in col.labs:
            if r["sent"].idx == m.sent.idx and r["name"] == m.norm:
                item.update({"result": r["result"], "unit": r["unit"]})
                break
        else:
            v = next((v for v in col.vitals
                      if v["sent"].idx == m.sent.idx and v["measurement"] == m.norm), None)
            if v:
                # e.g. oxygen saturation: already a complete vital sign, not a missing value
                item.update({"result": v["value"], "unit": v["unit"], "also_vital_sign": True})
            else:
                item["value_missing"] = True
    return item


def extract_department_specific_entities(text: str) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Specialty-specific mentions. Same output shape as before, plus assertion/span/fact_id."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    output: dict[str, dict[str, list[dict[str, object]]]] = {}
    if not text.strip():
        return {d: {k: [] for k in cats} for d, cats in DEPARTMENT_LEXICONS.items()}
    col = _collect(text)
    for dept, cats in DEPARTMENT_LEXICONS.items():
        block: dict[str, list[dict[str, object]]] = {}
        for key in cats:
            if key == "history":
                continue
            seen: set = set()
            items = []
            for m in col.dept[dept][key]:
                k = (m.norm, m.sent.idx)
                if k in seen:
                    continue
                seen.add(k)
                items.append(_dept_item(m, key, col))
            block[key] = items
        if "history" in cats:
            hist, seen = [], set()
            for m in _history_mentions(col):
                if (m.norm, m.sent.idx) in seen:
                    continue
                seen.add((m.norm, m.sent.idx))
                item = _dept_item(m, "history", col)
                item["history_type"] = _history_type(m)
                hist.append(item)
            block["history"] = hist
        output[dept] = block
    return output


def _document_notes(text: str) -> dict:
    col = _collect(text)
    gaps, quarantined = [], []
    for s in col.sents:
        rec = {"sentence_index": s.idx, "span": s.span(), "text": s.evidence}
        if s.role == "data_gap":
            gaps.append({**rec, "subtype": s.subtype})
        elif s.role in ("plan", "indication", "boilerplate"):
            quarantined.append({**rec, "role": s.role})
    return {"data_gaps": gaps, "quarantined_text": quarantined}


def extract_clinical_entities_v2(text: str) -> ClinicalExtractionV2:
    """Schema 2.0 envelope. `common` is a superset of every department block."""
    try:
        legacy_result = extract_clinical_entities(text)
        metadata = {
            "extraction_method": "rule_based",
            "requires_human_review": True,
            "extraction_schema_version": SCHEMA_VERSION,
        }
        if isinstance(text, str) and text.strip():
            metadata.update(_document_notes(text))
            metadata["allergy_status"] = _allergy_status(_collect(text))
        return ClinicalExtractionV2(
            schema_version="2.0", common=legacy_result.model_dump(), metadata=metadata)
    except Exception:
        logger.exception("Failed to convert clinical extraction into schema version 2.0")
        raise