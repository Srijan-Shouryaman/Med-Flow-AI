import logging

import re

import uuid

from app.schemas.clinical_entities import (

    ClinicalExtraction,

    ClinicalExtractionV2,

    SymptomEntity,

    DiagnosisEntity,

    MedicationEntity,

    AllergyEntity,

    ProcedureEntity,

    InvestigationEntity,

    ClinicalHistoryEntity,

    VitalSignEntity,

)

logger = logging.getLogger(__name__)

# --------------------------------------------------

# Clinical terminology

# --------------------------------------------------

SYMPTOMS = [

    # Existing cardiovascular / pulmonary / general symptoms

    "chest pain", "shortness of breath", "dyspnea", "cough",

    "fever", "fatigue", "headache", "dizziness", "nausea",

    "vomiting", "abdominal pain", "wheezing", "palpitations",

    "swelling", "weakness",

    # Gastroenterology symptoms

    "breathlessness", "hemoptysis",

    "constipation", "diarrhea", "diarrhoea", "bloating",

    "heartburn", "indigestion", "abdominal cramps", "stomach pain",

    "epigastric pain", "right upper quadrant pain",

    "left lower quadrant pain", "loss of appetite", "reduced appetite",

    "unintentional weight loss", "weight loss", "weight gain", "jaundice",

    "difficulty swallowing", "dysphagia", "painful swallowing",

    "odynophagia", "blood in stool", "rectal bleeding", "black stools",

    "melena", "vomiting blood", "hematemesis", "blood in vomit",

    "excessive gas", "belching", "acid reflux",

]

DIAGNOSES = [

    # Existing diagnoses

    "hypertension", "diabetes mellitus", "pneumonia",

    "asthma", "copd", "coronary artery disease",

    "heart failure", "bronchitis", "anemia",

    "pulmonary embolism", "myocardial infarction",

    # Gastroenterology diagnoses

    "non-alcoholic fatty liver disease",

    "gastroesophageal reflux disease",

    "inflammatory bowel disease",

    "peptic ulcer disease",

    "acute pancreatitis",

    "chronic pancreatitis",

    "fatty liver disease",

    "colorectal carcinoma",

    "colorectal cancer",

    "colon cancer",

    "intestinal obstruction",

    "bowel obstruction",

    "irritable bowel syndrome",

    "ulcerative colitis",

    "crohn's disease",

    "celiac disease",

    "coeliac disease",

    "diverticular disease",

    "diverticulitis",

    "gastroenteritis",

    "pancreatitis",

    "gastritis",

    "duodenal ulcer",

    "gastric ulcer",

    "peptic ulcer",

    "hepatitis",

    "cirrhosis",

    "liver failure",

    "cholecystitis",

    "cholelithiasis",

    "gallstones",

    "appendicitis",

    "gerd",

    "ibs",

    "hemorrhoids",

    "haemorrhoids",

    "anal fissure",

]

MEDICATIONS = [

    "aspirin", "metformin", "paracetamol", "acetaminophen",

    "amoxicillin", "azithromycin", "atorvastatin", "amlodipine",

    "losartan", "ibuprofen", "salbutamol", "omeprazole",

    "insulin", "prednisone", "prednisolone",

]

PROCEDURES = [

    "surgery", "biopsy", "angiography", "angioplasty",

    "echocardiogram", "ecg", "ekg", "ct scan", "mri",

    "x-ray", "endoscopy", "bronchoscopy", "ultrasound",

    "upper gastrointestinal endoscopy", "colonoscopy", "sigmoidoscopy",

]

# Longer terms are listed before their shorter overlapping variants.

INVESTIGATIONS = [

    "white blood cell count", "white blood cell", "platelet count",

    "haemoglobin", "hemoglobin", "chest x-ray", "creatinine",

    "troponin", "d-dimer", "glucose", "wbc",

    # Gastroenterology investigations

    "alanine aminotransferase",

    "aspartate aminotransferase",

    "gamma-glutamyl transferase",

    "liver function tests",

    "liver function test",

    "fecal occult blood",

    "faecal occult blood",

    "stool calprotectin",

    "fecal calprotectin",

    "faecal calprotectin",

    "helicobacter pylori",

    "upper gastrointestinal endoscopy",

    "abdominal ultrasound",

    "abdominal ct",

    "abdominal mri",

    "serum amylase",

    "serum lipase",

    "alkaline phosphatase",

    "bilirubin",

    "amylase",

    "lipase",

    "albumin",

    "alt",

    "ast",

    "lft",

    "alp",

    "ggt",

    "stool culture",

    "stool examination",

    "stool test",

    "h. pylori",

    "colonoscopy",

    "sigmoidoscopy",

]

# --------------------------------------------------

# Department-specific lexicons. These are intentionally separate from the
# legacy output schema: they provide extraction capability before we agree
# on the final department-specific JSON contract.
DEPARTMENT_LEXICONS = {
    "cardiovascular_pulmonary": {
        "symptoms": [
            "chest tightness", "chest discomfort", "exertional chest pain",
            "orthopnea", "paroxysmal nocturnal dyspnea", "syncope",
            "near syncope", "leg swelling", "peripheral edema",
            "sputum production", "productive cough", "dry cough",
            "pleuritic chest pain", "tachypnea", "hypoxia", "cyanosis",
            "hemoptysis", "shortness of breath", "dyspnea", "wheezing",
            "palpitations", "chest pain", "cough",
        ],
        "diagnoses": [
            "chronic obstructive pulmonary disease", "coronary artery disease",
            "ischemic heart disease", "ischaemic heart disease",
            "atrial fibrillation", "atrial flutter", "arrhythmia", "angina",
            "myocardial infarction", "cardiomyopathy", "pericarditis",
            "endocarditis", "aortic stenosis", "deep vein thrombosis",
            "peripheral arterial disease", "pulmonary hypertension",
            "pleural effusion", "pneumothorax", "respiratory failure",
            "pulmonary fibrosis", "interstitial lung disease",
            "pulmonary embolism", "heart failure", "hypertension",
            "pneumonia", "asthma", "bronchitis", "copd",
        ],
        "investigations": [
            "b-type natriuretic peptide", "brain natriuretic peptide",
            "nt-probnp", "high-sensitivity troponin", "troponin", "ck-mb",
            "creatine kinase", "electrocardiogram", "echocardiography",
            "holter monitor", "arterial blood gas", "pulmonary function test",
            "spirometry", "peak expiratory flow", "oxygen saturation",
            "d-dimer", "chest x-ray", "ecg", "ekg",
        ],
        "procedures": [
            "cardiac catheterization", "coronary angiography",
            "coronary angioplasty", "stress test", "holter monitoring",
            "thoracentesis", "pleural biopsy", "echocardiogram",
            "angiography", "angioplasty", "ecg", "ekg",
        ],
        "findings": [
            "heart murmur", "gallop rhythm", "irregular rhythm",
            "jugular venous distension", "raised jugular venous pressure",
            "delayed capillary refill", "reduced breath sounds",
            "decreased breath sounds", "crackles", "rales", "wheezes",
            "rhonchi", "stridor", "pleural rub", "pursed-lip breathing",
            "cardiomegaly", "pulmonary infiltrate", "consolidation",
            "atelectasis", "pneumothorax", "pulmonary edema",
            "pulmonary oedema", "lung nodule", "lung mass",
        ],
    },
    "consult_history_physical": {
        "symptoms": [
            "presenting complaint", "abdominal pain", "chest pain",
            "shortness of breath", "fatigue", "dizziness", "weakness",
            "fever", "nausea", "vomiting", "cough", "headache",
        ],
        "diagnoses": [
            "past medical history", "hypertension", "diabetes mellitus",
            "asthma", "pneumonia", "coronary artery disease", "heart failure",
        ],
        "medications": list(MEDICATIONS),
        "findings": [
            "physical examination", "abdominal tenderness", "joint tenderness",
            "bone tenderness", "heart murmur", "crackles", "wheezes",
            "reduced range of motion", "normal examination", "normal breath sounds",
            "soft abdomen", "no acute distress",
        ],
        "history": [
            "history of", "past medical history", "family history",
            "social history", "surgical history", "previously", "prior",
            "in the past", "current medication", "current medications",
        ],
    },
    "gastroenterology": {
        "symptoms": [
            "right upper quadrant pain", "left lower quadrant pain",
            "abdominal distension", "abdominal cramps", "epigastric pain",
            "difficulty swallowing", "painful swallowing", "blood in stool",
            "rectal bleeding", "black stools", "vomiting blood",
            "loss of appetite", "unintentional weight loss", "abdominal pain",
            "constipation", "diarrhea", "diarrhoea", "bloating", "heartburn",
            "indigestion", "dysphagia", "odynophagia", "melena", "hematemesis",
            "nausea", "vomiting", "jaundice", "acid reflux", "belching",
        ],
        "diagnoses": [
            "non-alcoholic fatty liver disease", "gastroesophageal reflux disease",
            "inflammatory bowel disease", "peptic ulcer disease",
            "acute pancreatitis", "chronic pancreatitis", "colorectal carcinoma",
            "colorectal cancer", "intestinal obstruction", "irritable bowel syndrome",
            "ulcerative colitis", "crohn's disease", "celiac disease",
            "coeliac disease", "diverticulitis", "gastroenteritis", "pancreatitis",
            "gastritis", "hepatitis", "cirrhosis", "liver failure",
            "cholecystitis", "cholelithiasis", "gallstones", "appendicitis",
            "hemorrhoids", "haemorrhoids", "anal fissure", "gerd", "ibs",
        ],
        "investigations": [
            "alanine aminotransferase", "aspartate aminotransferase",
            "gamma-glutamyl transferase", "liver function tests",
            "fecal occult blood", "faecal occult blood", "stool calprotectin",
            "fecal calprotectin", "faecal calprotectin", "helicobacter pylori",
            "upper gastrointestinal endoscopy", "abdominal ultrasound",
            "abdominal ct", "abdominal mri", "serum amylase", "serum lipase",
            "alkaline phosphatase", "bilirubin", "amylase", "lipase", "albumin",
            "stool culture", "stool examination", "stool test", "h. pylori",
            "colonoscopy", "sigmoidoscopy", "alt", "ast", "lft", "alp", "ggt",
        ],
        "procedures": [
            "endoscopic retrograde cholangiopancreatography", "esophagogastroduodenoscopy",
            "upper gastrointestinal endoscopy", "capsule endoscopy", "liver biopsy",
            "colonoscopy", "sigmoidoscopy", "endoscopy",
        ],
        "findings": [
            "abdominal guarding", "rebound tenderness", "abdominal tenderness",
            "hepatomegaly", "splenomegaly", "ascites", "abdominal distension",
            "abdominal rigidity", "bowel sounds", "jaundice",
        ],
    },
    "orthopedic": {
        "symptoms": [
            "reduced range of motion", "difficulty bearing weight", "difficulty walking",
            "joint stiffness", "joint pain", "knee pain", "hip pain", "shoulder pain",
            "back pain", "neck pain", "limb pain", "bone pain", "muscle pain",
            "muscle weakness", "myalgia", "arthralgia", "limited mobility", "limping",
            "numbness", "tingling", "swelling", "weakness",
        ],
        "diagnoses": [
            "intervertebral disc prolapse", "chronic osteomyelitis", "osteoarthritis",
            "rheumatoid arthritis", "osteoporosis", "osteomyelitis", "scoliosis",
            "spinal stenosis", "herniated disc", "fracture", "dislocation",
            "tendinitis", "bursitis", "ligament injury", "meniscal tear",
        ],
        "investigations": [
            "bone mineral density", "dual-energy x-ray absorptiometry", "joint x-ray",
            "bone scan", "mri scan", "computed tomography", "magnetic resonance imaging",
        ],
        "procedures": [
            "joint replacement", "arthroscopy", "joint aspiration", "internal fixation",
            "spinal fusion", "joint x-ray", "bone scan",
        ],
        "findings": [
            "joint tenderness", "bone tenderness", "crepitus", "joint deformity",
            "limited range of motion", "reduced range of motion",
            "decreased range of motion", "muscle atrophy", "muscle weakness",
            "antalgic gait", "abnormal gait", "joint instability", "spinal tenderness",
            "soft tissue swelling",
        ],
    },
    "radiology": {
        
        "investigations": [
            "computed tomography",
            "magnetic resonance imaging",
            "ct angiography",
            "mri scan",
            "chest radiograph",
            "chest x-ray",
            "chest ct",
            "abdominal ultrasound",
            "abdominal ct",
            "abdominal mri",
            "doppler ultrasound",
            "ultrasound",
            "x-ray",
            "mri",
            "ct scan",
        ],

        "procedures": [
            "computed tomography", "magnetic resonance imaging", "ct angiography",
            "chest radiograph", "chest x-ray", "doppler ultrasound", "ultrasound",
            "x-ray", "mri", "ct scan","chest ct",
        ],
        "findings": [
            "pericardial effusion", "pleural effusion", "pulmonary infiltrate",
            "consolidation", "atelectasis", "pneumothorax", "pulmonary edema",
            "pulmonary oedema", "lung nodule", "lung mass", "cardiomegaly",
            "fracture", "dislocation", "joint effusion", "degenerative changes",
            "osteophyte", "disc herniation", "spinal stenosis", "soft tissue swelling",
            "opacity", "no acute cardiopulmonary abnormality",
        ],
    },
}

# Context patterns

# --------------------------------------------------

NEGATION_PATTERN = re.compile(

    r"\b(no|not|denies|denied|without|negative for|"

    r"absence of|rules out|ruled out)\b",

    re.IGNORECASE,

)

UNCERTAINTY_PATTERN = re.compile(

    r"\b(possible|possibly|probable|suspected|suspect|"

    r"may have|might have|cannot exclude|rule out|"

    r"questionable|likely)\b",

    re.IGNORECASE,

)

HISTORICAL_PATTERN = re.compile(

    r"\b(history of|previously|previous|prior|formerly|"

    r"resolved|in the past|past medical history)\b",

    re.IGNORECASE,

)

PLANNED_PATTERN = re.compile(

    r"\b(plan to|planned|scheduled|recommend|"

    r"will undergo|consider)\b",

    re.IGNORECASE,

)

# Handles values such as 120, 37.5, and 120/80.

VITAL_PATTERN = re.compile(

    r"\b("

    r"blood pressure|bp|heart rate|pulse|temperature|"

    r"respiratory rate|spo2|oxygen saturation|"

    r"weight|height|bmi"

    r")\s*[:=]?\s*"

    r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)"

    r"\s*(mmhg|bpm|°c|°f|c|f|%|kg|cm|breaths/min)?",

    re.IGNORECASE,

)

# Comma-separated thousands are captured as a single value.

# Units include common lab units, including ng/L and cells/µL.

LAB_RESULT_PATTERN = re.compile(

    r"\b("

    r"white blood cell count|platelet count|haemoglobin|hemoglobin|"

    r"white blood cell|wbc|creatinine|glucose|troponin|d-dimer|"

    r"alanine aminotransferase|aspartate aminotransferase|"

    r"gamma-glutamyl transferase|alkaline phosphatase|"

    r"serum amylase|serum lipase|amylase|lipase|bilirubin|albumin|"

    r"alt|ast|alp|ggt"

    r")\s*[:=]?\s*"

    r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"

    r"\s*"

    r"(mg/dl|g/dl|mmol/l|ng/l|ng/ml|pg/ml|"

    r"10\^?3/?[µu]?l|10\^?6/?[µu]?l|"

    r"cells?/?[µu]l|[µu]g/l|%)?",

    re.IGNORECASE,

)

# Allergy statements that indicate no known allergy rather than a named allergen.

NO_KNOWN_ALLERGY_PATTERN = re.compile(

    r"\b(no known|no reported|denies|without)\s+"

    r"(?:(?:drug|medication|food|known)\s+)*allerg(?:y|ies)\b|"

    r"\bno known allergies\b",

    re.IGNORECASE,

)

ALLERGY_HEADING_PATTERN = re.compile(

    r"^\s*(?:allergy|allergies|allergy information|"

    r"allergy history|allergies and reactions)\s*[:\-]?\s*$",

    re.IGNORECASE,

)

ALLERGY_SUBSTANCE_PATTERN = re.compile(

    r"\ballerg(?:y|ies|ic)\s+(?:to\s+)?"

    r"([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,3})",

    re.IGNORECASE,

)

HISTORY_HEADING_PATTERN = re.compile(

    r"^\s*(?:history of present illness|"

    r"present illness|past medical history|"

    r"medical history|clinical history)\s*[:\-]?\s*$",

    re.IGNORECASE,

)

# --------------------------------------------------

# Utility functions

# --------------------------------------------------

def _sentences(text: str) -> list[str]:

    """Split text while retaining sentence-level evidence."""

    return [

        part.strip()

        for part in re.split(r"(?<=[.!?])\s+|\n+", text)

        if part.strip()

    ]

def _contains_term(sentence: str, term: str) -> bool:

    return bool(

        re.search(

            rf"(?<!\w){re.escape(term)}(?!\w)",

            sentence,

            re.IGNORECASE,

        )

    )

def _entity_id() -> str:

    return str(uuid.uuid4())

def _status(sentence: str, category: str) -> str:

    if NEGATION_PATTERN.search(sentence):

        return "NEGATED"

    if UNCERTAINTY_PATTERN.search(sentence):

        return "UNCERTAIN"

    if PLANNED_PATTERN.search(sentence):

        return "PLANNED"

    if HISTORICAL_PATTERN.search(sentence):

        return "HISTORICAL"

    return "MENTIONED"

def _negated(sentence: str) -> bool | None:

    if UNCERTAINTY_PATTERN.search(sentence):

        return None

    return bool(NEGATION_PATTERN.search(sentence))

def _matching_terms(sentence: str, terms: list[str]) -> list[str]:

    """

    Return matching terms while preferring longer terms over contained

    shorter terms. For example, return 'white blood cell count' without

    also returning 'white blood cell' for the same phrase.

    """

    matches = [

        term for term in sorted(terms, key=len, reverse=True)

        if _contains_term(sentence, term)

    ]

    selected: list[str] = []

    for term in matches:

        if any(

            re.search(

                rf"(?<!\w){re.escape(term)}(?!\w)",

                selected_term,

                re.IGNORECASE,

            )

            or re.search(

                rf"(?<!\w){re.escape(selected_term)}(?!\w)",

                term,

                re.IGNORECASE,

            )

            for selected_term in selected

        ):

            continue

        selected.append(term)

    return selected

def _add_term_entities(

    sentence: str,

    terms: list[str],

    category: str,

    extraction: ClinicalExtraction,

) -> None:

    for term in _matching_terms(sentence, terms):

        status = _status(sentence, category)

        evidence = sentence

        entity_id = _entity_id()

        if category == "symptom":

            extraction.symptoms.append(

                SymptomEntity(

                    entity_id=entity_id,

                    name=term,

                    status=status,

                    temporality=status.lower(),

                    negated=_negated(sentence),

                    evidence=evidence,

                    extraction="lexicon_rule",

                )

            )

        elif category == "diagnosis":

            extraction.diagnoses.append(

                DiagnosisEntity(

                    entity_id=entity_id,

                    name=term,

                    status=status,

                    negated=_negated(sentence),

                    temporality=status.lower(),

                    evidence=evidence,

                    extraction="lexicon_rule",

                )

            )

        elif category == "medication":

            extraction.medications.append(

                MedicationEntity(

                    entity_id=entity_id,

                    name=term,

                    status=status,

                    evidence=evidence,

                    extraction="lexicon_rule",

                )

            )

        elif category == "procedure":

            extraction.procedures.append(

                ProcedureEntity(

                    entity_id=entity_id,

                    name=term,

                    status=status,

                    evidence=evidence,

                    extraction="lexicon_rule",

                )

            )

        elif category == "investigation":

            extraction.investigations.append(

                InvestigationEntity(

                    entity_id=entity_id,

                    name=term,

                    evidence=evidence,

                    extraction="lexicon_rule",

                )

            )

def _normalise_lab_name(name: str) -> str:

    """Normalize common aliases to support investigation deduplication."""

    value = re.sub(r"\s+", " ", name.strip().lower())

    aliases = {

        "haemoglobin": "hemoglobin",

        "wbc": "white blood cell count",

        "white blood cell": "white blood cell count",

    }

    return aliases.get(value, value)

def _merge_duplicate_investigations(

    investigations: list[InvestigationEntity],

) -> list[InvestigationEntity]:

    """

    Merge detections for the same investigation in the same evidence sentence.

    Prefer a record with a numeric result and retain the existing entity ID.

    Separate mentions in different sentences are preserved.

    """

    merged: list[InvestigationEntity] = []

    positions: dict[tuple[str, str], int] = {}

    for item in investigations:

        key = (

            _normalise_lab_name(item.name),

            re.sub(r"\s+", " ", item.evidence.strip().lower()),

        )

        if key not in positions:

            positions[key] = len(merged)

            merged.append(item)

            continue

        index = positions[key]

        existing = merged[index]

        # Prefer the numeric extraction, but preserve the original entity ID.

        if existing.result is None and item.result is not None:

            data = item.model_dump()

            data["entity_id"] = existing.entity_id

            merged[index] = InvestigationEntity(**data)

        else:

            # Fill any useful fields that were missing from the first record.

            data = existing.model_dump()

            if data.get("result") is None and item.result is not None:

                data["result"] = item.result

            if data.get("unit") is None and item.unit is not None:

                data["unit"] = item.unit

            if data.get("reference_range") is None and item.reference_range:

                data["reference_range"] = item.reference_range

            merged[index] = InvestigationEntity(**data)

    return merged

def _deduplicate_symptoms(

    entities: list[SymptomEntity],

) -> list[SymptomEntity]:

    """Consolidate repeated symptom mentions while preserving their evidence."""

    merged: list[SymptomEntity] = []

    positions: dict[tuple[str, str, bool | None], int] = {}

    for item in entities:

        key = (

            re.sub(r"\s+", " ", item.name.strip().lower()),

            (item.status or "").upper(),

            item.negated,

        )

        if key not in positions:

            positions[key] = len(merged)

            merged.append(item)

            continue

        index = positions[key]

        existing = merged[index]

        evidence_parts = [

            part.strip() for part in existing.evidence.split(" | ")

        ]

        if item.evidence.strip() not in evidence_parts:

            data = existing.model_dump()

            data["evidence"] = f"{existing.evidence} | {item.evidence}"

            merged[index] = SymptomEntity(**data)

    return merged

# --------------------------------------------------

# Main extraction service — legacy schema

# --------------------------------------------------

def extract_clinical_entities(text: str) -> ClinicalExtraction:

    """

    Extract candidate clinical entities from original OCR text.

    This is a preliminary rule-based extraction layer, not a validated

    clinical NLP model. Extracted entities require appropriate human review.

    """

    result = ClinicalExtraction()

    if not isinstance(text, str) or not text.strip():

        return result

    for sentence in _sentences(text):

        _add_term_entities(sentence, SYMPTOMS, "symptom", result)

        _add_term_entities(sentence, DIAGNOSES, "diagnosis", result)

        _add_term_entities(sentence, MEDICATIONS, "medication", result)

        _add_term_entities(sentence, PROCEDURES, "procedure", result)

        _add_term_entities(sentence, INVESTIGATIONS, "investigation", result)

        # Allergy mentions: ignore headings and avoid extracting trailing

        # prose such as "are documented" as the allergy substance.

        if (

            re.search(r"\ballerg(?:y|ies|ic)\b", sentence, re.IGNORECASE)

            and not ALLERGY_HEADING_PATTERN.fullmatch(sentence)

        ):

            no_known_allergy = bool(

                NO_KNOWN_ALLERGY_PATTERN.search(sentence)

            )

            negated = _negated(sentence)

            if no_known_allergy:

                # Preserve negative history without inventing an allergen.

                result.allergies.append(

                    AllergyEntity(

                        entity_id=_entity_id(),

                        substance="unspecified",

                        status="NEGATED",

                        evidence=sentence,

                        extraction="allergy_rule",

                    )

                )

            else:

                match = ALLERGY_SUBSTANCE_PATTERN.search(sentence)

                substance = match.group(1).strip(" .,:;") if match else ""

                invalid_substances = {

                    "information", "history", "status", "details",

                    "are documented", "is documented",

                }

                if substance and substance.lower() not in invalid_substances:

                    allergy_status = (

                        "UNCERTAIN"

                        if negated is None

                        else "NEGATED"

                        if negated

                        else _status(sentence, "allergy")

                    )

                    result.allergies.append(

                        AllergyEntity(

                            entity_id=_entity_id(),

                            substance=substance,

                            status=allergy_status,

                            evidence=sentence,

                            extraction="allergy_rule",

                        )

                    )

        # Vital signs: extract measurement, value and unit.

        for match in VITAL_PATTERN.finditer(sentence):

            measurement = match.group(1).lower()

            value = re.sub(r"\s+", "", match.group(2))

            unit = match.group(3) or ""

            result.vital_signs.append(

                VitalSignEntity(

                    entity_id=_entity_id(),

                    measurement=measurement,

                    value=value,

                    unit=unit,

                    evidence=sentence,

                    extraction="vital_pattern",

                )

            )

        # ECG heart rate: handle phrases such as

        # "Sinus rhythm at 96 beats/min."

        if re.search(

            r"\b(?:ecg|ekg|electrocardiogram)\b",

            sentence,

            re.IGNORECASE,

        ):

            ecg_rate_pattern = re.compile(

                r"\b(?:at\s+)?(\d+(?:\.\d+)?)\s*"

                r"(?:beats\s*/\s*min|beats\s+per\s+minute)\b",

                re.IGNORECASE,

            )

            rate_match = ecg_rate_pattern.search(sentence)

            if rate_match:

                rate_value = rate_match.group(1)

                already_extracted = any(

                    vital.measurement.lower() == "heart rate"

                    and vital.value == rate_value

                    and vital.evidence == sentence

                    for vital in result.vital_signs

                )

                if not already_extracted:

                    result.vital_signs.append(

                        VitalSignEntity(

                            entity_id=_entity_id(),

                            measurement="heart rate",

                            value=rate_value,

                            unit="bpm",

                            evidence=sentence,

                            extraction="ecg_rate_pattern",

                        )

                    )

        # Numeric laboratory results. Thousands separators are preserved.

        for match in LAB_RESULT_PATTERN.finditer(sentence):

            raw_name = match.group(1)

            result_value = match.group(2)

            unit = match.group(3) or None

            result.investigations.append(

                InvestigationEntity(

                    entity_id=_entity_id(),

                    name=raw_name,

                    result=result_value,

                    unit=unit,

                    evidence=sentence,

                    extraction="lab_pattern",

                )

            )

        # Preserve historical context, but don't treat section headings as

        # patient history.

        if (

            HISTORICAL_PATTERN.search(sentence)

            and not HISTORY_HEADING_PATTERN.fullmatch(sentence)

        ):

            result.clinical_history.append(

                ClinicalHistoryEntity(

                    entity_id=_entity_id(),

                    history_type="historical_mention",

                    description=sentence,

                    status="HISTORICAL",

                    temporality="historical",

                    evidence=sentence,

                    extraction="history_rule",

                )

            )

    result.symptoms = _deduplicate_symptoms(result.symptoms)

    result.investigations = _merge_duplicate_investigations(

        result.investigations

    )

    return result

# --------------------------------------------------

def extract_department_specific_entities(text: str) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Extract specialty-specific mentions without changing the legacy schema.

    This is an interim internal representation. It intentionally does not
    replace ``extract_clinical_entities`` or decide the final API JSON format.
    Each item includes the matched term, category, status, negation, and source
    sentence so downstream code can review the extraction before persistence.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    output: dict[str, dict[str, list[dict[str, object]]]] = {}
    seen: set[tuple[str, str, str, str]] = set()

    for department, categories in DEPARTMENT_LEXICONS.items():
        department_result: dict[str, list[dict[str, object]]] = {}
        for category, terms in categories.items():
            department_result[category] = []

        for sentence in _sentences(text):
            for category, terms in categories.items():
                for term in _matching_terms(sentence, terms):
                    key = (department, category, term.lower(), sentence.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    status = _status(sentence, category)
                    department_result[category].append({
                        "name": term,
                        "category": category,
                        "status": status,
                        "negated": _negated(sentence),
                        "temporality": status.lower(),
                        "evidence": sentence,
                        "extraction": "department_lexicon_rule",
                    })

        output[department] = department_result

    return output

# Version 2.0 schema adapter

# --------------------------------------------------

def extract_clinical_entities_v2(text: str) -> ClinicalExtractionV2:

    """

    Return extraction results using the version 2.0 JSON schema.

    The legacy extractor remains responsible for identifying entities.

    This adapter places those results under `common` and marks the output

    for human review.

    Department-specific mapping is intentionally not inferred here because

    this function does not receive the classifier's predicted department.

    """

    try:

        legacy_result = extract_clinical_entities(text)

        return ClinicalExtractionV2(

            schema_version="2.0",

            common=legacy_result.model_dump(),

            metadata={

                "extraction_method": "rule_based",

                "requires_human_review": True,

            },

        )

    except Exception:

        logger.exception(

            "Failed to convert clinical extraction into schema version 2.0"

        )

        raise
