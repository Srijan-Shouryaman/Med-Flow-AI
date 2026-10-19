"""Ground-truth queries for the three synthetic Patient 5 cardiology reports."""
from dataclasses import dataclass

PATIENT_ID = 5
R1, R2, R3 = "2026-03-12", "2026-06-04", "2026-09-18"
ALL = (R1, R2, R3)

CHUNK_TYPES = ("symptoms_present", "symptoms_uncertain", "pertinent_negatives", "diagnoses", "family_history", "clinical_history", "vitals", "investigations", "medications", "procedures_tests", "exam_findings", "allergy_status", "data_gaps")

@dataclass(frozen=True)
class Case:
    id: str
    category: str
    query: str
    types: tuple[str, ...]
    terms: tuple[str, ...] = ()
    reports: tuple[str, ...] = ()
    lexical_gap: bool = False

C = Case
CASES = [
    C("sym01","symptoms","What chest pain symptoms does the patient have?",("symptoms_present",),("chest",),(R1,R2)),
    C("sym02","symptoms","palpitations",("symptoms_present","symptoms_uncertain"),("palpitation",),(R1,R3)),
    C("sym03","symptoms","Which symptoms is the patient unsure about?",("symptoms_uncertain",),("unsure","unclear","not sure","uncertain","dizz","light-headed","palpitation")),
    C("sym04","symptoms","ankle swelling",("symptoms_present",),("swelling","edema"),(R3,)),
    C("sym05","symptoms","sweating and nausea with pain radiating to the arm",("symptoms_present",),("sweat","nausea","radiat"),(R2,)),
    C("sym06","symptoms","shortness of breath",("symptoms_present",),("breathless","shortness of breath","dyspnea"),ALL,True),
    C("neg01","negatives","Does the patient have syncope?",("pertinent_negatives",),("syncope",),ALL),
    C("neg02","negatives","Was pulmonary embolism excluded?",("pertinent_negatives","diagnoses"),("pulmonary embolism",),(R1,R2)),
    C("neg03","negatives","Any history of stroke?",("pertinent_negatives","clinical_history"),("stroke",),(R2,R3)),
    C("neg04","negatives","Was aortic dissection ruled out?",("pertinent_negatives","diagnoses"),("dissection",),(R2,)),
    C("neg05","negatives","Is there jugular venous distension?",("pertinent_negatives",),("jugular","jvd"),ALL),
    C("neg06","negatives","Does the patient have orthopnea?",("pertinent_negatives",),("orthopnea",),(R1,R3)),
    C("neg07","negatives","Does the patient currently have chest pain?",("pertinent_negatives",),("chest pain",),(R3,)),
    C("dx01","diagnoses","What is the heart rhythm diagnosis?",("diagnoses",),("atrial fibrillation",),(R3,)),
    C("dx02","diagnoses","NSTEMI",("diagnoses",),("nstemi","myocardial infarction"),(R2,)),
    C("dx03","diagnoses","Which diagnoses are only possible or uncertain?",("diagnoses",),("possible","uncertain","cannot be excluded","unlikely","gerd","reflux","non-adherence","stent thrombosis"),ALL),
    C("dx04","diagnoses","heart failure",("diagnoses",),("heart failure",),(R2,R3)),
    C("dx05","diagnoses","stable angina",("diagnoses",),("angina",),(R1,)),
    C("fam01","family","family history of heart disease",("family_history",),("myocardial","coronary"),ALL),
    C("fam02","family","Does a sibling have diabetes?",("family_history",),("sister","diabetes"),(R3,)),
    C("fam03","family","Did the father have a heart attack?",("family_history",),("father",),ALL,True),
    C("his01","history","previous surgeries",("clinical_history",),("appendectomy",),(R1,)),
    C("his02","history","history of stent or PCI",("clinical_history",),("stent","nstemi","pci"),(R3,)),
    C("his03","history","How long has the patient had hypertension?",("clinical_history",),("hypertension",),ALL),
    C("vit01","vitals","blood pressure readings",("vitals",),("bp","blood pressure","148/92","156/96","126/78"),ALL),
    C("vit02","vitals","oxygen saturation",("vitals",),("spo2","oxygen"),ALL),
    C("vit03","vitals","heart rate 112",("vitals",),("112",),(R3,)),
    C("vit04","vitals","body weight",("vitals",),("weight","82","84"),(R1,R3)),
    C("inv01","investigations","troponin",("investigations",),("troponin",),(R2,)),
    C("inv02","investigations","ejection fraction",("investigations",),("lvef","ejection"),(R2,R3)),
    C("inv03","investigations","LDL cholesterol",("investigations",),("ldl",),ALL),
    C("inv04","investigations","NT-proBNP",("investigations",),("bnp",),(R3,)),
    C("inv05","investigations","ECG findings",("investigations",),("ecg",),ALL),
    C("med01","medications","What medications is the patient on?",("medications",),("furosemide","ramipril"),(R3,)),
    C("med02","medications","statin dose",("medications",),("atorvastatin",),ALL),
    C("med03","medications","diuretic",("medications",),("furosemide",),(R3,),True),
    C("med04","medications","anticoagulant given in hospital",("medications",),("enoxaparin",),(R2,),True),
    C("med05","medications","blood thinners",("medications",),("aspirin","clopidogrel","enoxaparin"),(R2,R3),True),
    C("pro01","procedures","Was coronary angiography performed?",("procedures_tests",),("angiograph",),(R2,)),
    C("pro02","procedures","Was a stent placed?",("procedures_tests",),("stent","pci","percutaneous"),(R2,)),
    C("pro03","procedures","Was the cardiac MRI done?",("procedures_tests",),("mri",),(R2,)),
    C("pro04","procedures","Holter monitor",("procedures_tests",),("holter",),(R3,)),
    C("pro05","procedures","treadmill stress test",("procedures_tests",),("treadmill","stress"),(R1,)),
    C("exa01","exam","gallop or extra heart sounds",("exam_findings",),("s3","s4","gallop"),(R1,R2)),
    C("exa02","exam","swelling in the legs on examination",("exam_findings",),("edema",),(R2,R3)),
    C("exa03","exam","lung crackles",("exam_findings",),("crackles",),(R2,)),
    C("exa04","exam","murmur",("exam_findings",),("murmur",),(R3,)),
    C("exa05","exam","irregular pulse",("exam_findings",),("irregular",),(R3,)),
    C("all01","allergy","Does the patient have any drug allergies?",("allergy_status",),(),ALL),
    C("all02","allergy","penicillin allergy",("allergy_status",),("penicillin",),(R2,)),
    C("all03","allergy","sulfa allergy",("allergy_status",),("sulfa",),(R3,)),
    C("gap01","gaps","Was smoking status recorded?",("data_gaps",),("smoking",),ALL),
    C("gap02","gaps","HbA1c result",("data_gaps",),("hba1c",),ALL),
    C("gap03","gaps","CHA2DS2-VASc score",("data_gaps",),("cha2ds2","vasc"),(R3,)),
    C("gap04","gaps","final angiography report",("data_gaps",),("angiograph",),(R2,)),
    C("gap05","gaps","previous ECG for comparison",("data_gaps",),("ecg",),(R1,)),
    C("par01","paraphrase","heart attack",("diagnoses",),("nstemi","myocardial infarction"),(R2,),True),
    C("par02","paraphrase","high blood pressure",("clinical_history",),("hypertension",),ALL,True),
    C("par03","paraphrase","sugar disease",("clinical_history",),("diabetes",),ALL,True),
    C("par04","paraphrase","irregular heartbeat",("diagnoses","investigations","exam_findings"),("atrial fibrillation","irregularly irregular"),(R3,),True),
    C("par05","paraphrase","weak heart muscle",("diagnoses","investigations"),("heart failure","lvef","ejection","cardiomyopathy"),(R2,R3),True),
]

FORBIDDEN = [
    ("f01","Does the patient have syncope?","symptoms_present","syncope",(),"syncope denied in all reports"),
    ("f02","Does the patient have fever?","symptoms_present","fever",(),"fever denied"),
    ("f03","Does the patient have orthopnea?","symptoms_present","orthopnea",(),"orthopnea denied"),
    ("f04","Is there bleeding?","symptoms_present","bleeding",(R3,),"bleeding denied in R3"),
    ("f05","Does the patient have chest pain now?","symptoms_present","chest pain",(R3,),"R3 denies chest pain since discharge"),
    ("f06","pedal edema on examination","exam_findings","edema",(R1,),"R1 says no pedal edema"),
    ("f07","jugular venous distension","exam_findings","jugular",(),"JVD absent"),
    ("f08","murmur on examination","exam_findings","murmur",(R1,R2),"R1/R2 say no murmurs"),
    ("f09","past medical history of myocardial infarction","clinical_history","myocardial infarction",(R1,),"R1 denies prior MI"),
    ("f10","previous heart attack","clinical_history","heart attack",(R2,),"R2 denies prior heart attack"),
    ("f11","history of stroke","clinical_history","stroke",(),"stroke denied"),
    ("f12","who in the family has coronary disease","clinical_history","father",(),"relatives belong in family_history"),
    ("f13","past surgeries","family_history","appendectomy",(),"patient's own surgery belongs in clinical_history"),
]

LABELS = [
    ("l01","treadmill stress test","procedures_tests","stress",R1,("mention","unconfirmed","not confirmed"),()),
    ("l02","cardiac MRI","procedures_tests","mri",R2,("mention","unconfirmed","not confirmed"),()),
    ("l03","Holter monitor","procedures_tests","holter",R3,("mention","unconfirmed","not confirmed"),()),
    ("l04","coronary angiography","procedures_tests","angiograph",R2,(),("mention only",)),
    ("l05","drug eluting stent","procedures_tests","stent",R2,(),("mention only",)),
]
ALLERGY_EXPECTED = {R1:"NONE_KNOWN", R2:"PRESENT", R3:"CONFLICTING"}
THRESHOLDS = {"hit@k":0.90, "mrr":0.70, "report_recall@k":0.80, "lexical_hit@k":0.60}
