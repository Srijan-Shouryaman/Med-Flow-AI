# MedFlow AI

MedFlow AI is a clinical operations application that brings patient records, medical report processing, case workflows, AI-assisted image analysis, predictions, recommendations, assignments, and a clinical records assistant into one workspace.

This README describes the repository as it currently exists. It does not include database credentials, API keys, user credentials, or patient data.

## Project at a glance

- **Frontend:** React 19, React Router, Vite, Tailwind CSS, and Lucide icons.
- **Backend:** FastAPI, Pydantic, SQLAlchemy, and PostgreSQL.
- **Clinical retrieval:** MedCPT embeddings (768 dimensions), PostgreSQL full-text search and pgvector, reciprocal-rank fusion, BGE cross-encoder reranking, and Gemini answer generation.
- **Medical image analysis:** Brain-stroke and lung-cancer model workflows with Grad-CAM support.
- **Authentication:** JWT bearer tokens; the frontend API client attaches the stored token to authenticated requests.

## Main workflows

- **Patients and reports:** Browse patient records, open patient details, and view uploaded reports.
- **Report upload:** The patient-specific upload flow at /patients/:patientId/upload calls POST /api/reports/upload and feeds the existing report/case workflow.
- **Clinical NLP processing:** The dedicated /reports/upload page and the Clinical NLP card on Patient Details call POST /api/clinical-nlp/process. This is a separate processing flow from the patient-specific report upload above.
- **Cases:** View case details, prediction and recommendation history, assignment decisions, and permitted case-status transitions.
- **Clinical Assistant:** Ask patient-scoped questions at /clinical-assistant; requests go to POST /api/rag/query and return evidence sources and citations.
- **AI Diagnosis:** Run the Brain Stroke or Lung Cancer image analysis workflows.
- **Operations:** Team performance and workflow administration views are available from the navigation. At this repository snapshot, /departments and /teams are routed through the generic module placeholder component even though the backend list/detail endpoints exist.

Clinical Assistant output is advisory and carries a human-review requirement. It is not an autonomous diagnosis or treatment decision.

## Repository structure

    backend/
      app/
        api/routes/          FastAPI route handlers
        auth/                JWT authentication and authorization helpers
        core/                Settings and database session setup
        ml/                  Image models and inference/XAI code
        schemas/             Request and response schemas
        services/            Report, NLP, retrieval, RAG, workflow services
      migrations/            SQL migrations currently checked in
      retrieval_quality/     Golden-set retrieval evaluation tools
      tests/                 Pytest-discovered unit and integration tests
      requirements.txt       Backend Python package pins
    frontend/
      src/
        auth/                Auth and Clinical Assistant session state
        components/          Layout, shared UI, and report components
        config/              Navigation configuration
        pages/               Route page components
        services/            API clients and feature services
        styles.css           Application styling
      tests/                 Node test-runner tests
      package.json
    requirements.txt         Additional/root Python dependency list

## Frontend routes

All application routes other than /login are wrapped in the authenticated application shell.

| Route | Page |
|---|---|
| /login | Sign in |
| / | Dashboard |
| /patients | Patient directory |
| /patients/new | Add patient |
| /patients/:patientId | Patient details, including report actions and Clinical NLP processing |
| /patients/:patientId/upload | Patient-specific report upload |
| /reports | Report history |
| /reports/upload | Clinical NLP report upload |
| /reports/:reportId | Report details |
| /cases | Case work queue |
| /cases/:caseId | Case details |
| /predictions | Prediction lookup/history |
| /predictions/:predictionId | Prediction details |
| /recommendations | Recommendations |
| /assignments | Assignment review/history |
| /assignments/:assignmentId | Assignment details |
| /case-status | Update a case status |
| /clinical-assistant | Clinical records assistant |
| /diagnosis | AI Diagnosis workspace |
| /team-performance | Team performance |
| /departments, /teams | Currently resolve through the generic module placeholder |

Navigation items can be role-filtered in frontend/src/config/navigation.js. The frontend route guard is in frontend/src/components/routing/ProtectedRoute.jsx.

## Backend API overview

FastAPI mounts the route groups from backend/app/main.py. The interactive API documentation is available at /docs while the backend is running; the health endpoint is /health.

| Area | Routes currently defined |
|---|---|
| Authentication | POST /api/auth/login |
| Patients | GET /api/patients, POST /api/patients, GET /api/patients/{patient_id}, GET /api/patients/ref/{patient_ref_id}, PUT /api/patients/{patient_id} |
| Reports | POST /api/reports/upload, GET /api/reports/, GET /api/reports/{report_id} |
| Clinical NLP | POST /api/clinical-nlp/process, GET /api/clinical-nlp/reports/{report_id} |
| Clinical RAG | POST /api/rag/query |
| Cases | GET /api/cases/, GET /api/cases/history, GET /api/cases/{case_id}, PATCH /api/cases/{case_id}/status; case-scoped prediction, recommendation, and assignment endpoints are also defined |
| Predictions | GET /api/predictions/, GET /api/predictions/{prediction_id}; case prediction endpoints are under /api/cases/{case_id}/prediction |
| Recommendations | GET /api/recommendations/, GET /api/recommendations/{case_id}; generation is available under /api/cases/{case_id}/recommendations |
| Assignments | GET /api/assignments/, GET /api/assignments/{assignment_id}; approve and override actions are under /api/cases/{case_id}/assignment/ |
| AI Diagnosis | POST /api/diagnosis/brain-stroke, POST /api/diagnosis/lung-cancer |
| Departments and teams | GET /api/departments/, GET /api/departments/{department_id}, GET /api/teams/, GET /api/teams/{team_id} |
| Team performance | GET /api/team-performance/, GET /api/team-performance/{team_id} |

Most API operations require a JWT bearer token. The frontend centralizes requests in frontend/src/services/apiClient.js; avoid bypassing it for authenticated features.

## Local setup

### Prerequisites

- Python 3.10 or newer.
- Node.js and npm compatible with the Vite version in frontend/package.json.
- PostgreSQL with the project schema already provisioned and the pgvector extension available for clinical retrieval.
- A Gemini API key for evidence-backed Clinical Assistant generation.
- Network access or pre-cached Hugging Face models for MedCPT and BGE model loading, unless your environment supplies them separately.

The repository does not include a complete database schema bootstrap or seed-data script. Provision the database using the project's existing database environment and approved migration process. The checked-in SQL migration currently updates permitted case status values: backend/migrations/20261009_allow_case_progress_statuses.sql.

### Backend configuration

Create backend/.env from backend/.env.example and fill it locally. Do not commit .env files or share their contents.

| Variable | Purpose | Default/requirement |
|---|---|---|
| DATABASE_URL | PostgreSQL connection string | Required |
| JWT_SECRET_KEY | Signing key for access tokens | Required; use a strong private value |
| JWT_ALGORITHM | JWT signing algorithm | Defaults to HS256 |
| JWT_EXPIRATION_MINUTES | Access token lifetime | Defaults to 60 |
| GEMINI_API_KEY | Gemini generation credential | Needed when evidence is present and answer generation runs |
| GEMINI_MODEL | Gemini model name | Defaults to gemini-2.5-flash |

Example shape only—replace the placeholders locally and never use them as real credentials:

    DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DATABASE
    JWT_SECRET_KEY=replace-with-a-long-random-secret
    JWT_ALGORITHM=HS256
    JWT_EXPIRATION_MINUTES=60
    GEMINI_API_KEY=replace-with-your-private-key
    GEMINI_MODEL=gemini-2.5-flash

### Backend dependency note

backend/requirements.txt is the checked-in backend manifest, but it currently omits several packages imported by active routes/services, including torch, torchvision, transformers, easyocr, google-genai, python-multipart, and email-validator. A clean environment installed only from that file may fail during application startup or when using the affected features. Add and validate compatible version pins in the dependency manifest before treating the listed install command as a reproducible deployment setup. pytest is also not listed there as a development dependency.

### Start the backend

From the repository root, in a PowerShell terminal:

    cd backend
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Resolve the dependency-manifest note above before expecting the server to start in a fresh environment. Once running, check:

- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

### Start the frontend

In a second terminal:

    cd frontend
    npm ci
    npm run dev

Vite serves the frontend locally (normally at http://localhost:5173). frontend/vite.config.js proxies /api and /health to http://127.0.0.1:8000 by default. You can override the proxy target with VITE_API_PROXY_TARGET or set VITE_API_BASE_URL to a backend origin. The provided frontend/.env.example leaves the API base URL empty so the dev proxy is used.

## Upload and clinical assistant flows

There are two report upload workflows; they should not be confused:

1. **Patient-specific report workflow:** open a patient and use the existing medical-report action. The frontend route is /patients/:patientId/upload; it submits patient_id and file to POST /api/reports/upload.
2. **Clinical NLP processing workflow:** open /reports/upload or the Clinical NLP Processing card on a Patient Details page. It submits multipart patient_id and file to POST /api/clinical-nlp/process. The backend handles extraction, chunking, embedding, and storage.

The Clinical Assistant page at /clinical-assistant sends JSON to POST /api/rag/query. Its request includes patient_id, question, and optional case_id; the current UI sends a patient-scoped question and displays response citations and source metadata.

Uploaded report files are stored under backend/storage/reports by the report service. Protect this directory and its backups as clinical data.

## Retrieval and AI components

- The report ingestion path uses MedCPT's article encoder to embed stored chunks; queries use the MedCPT query encoder. Vectors are 768-dimensional.
- Dense retrieval uses pgvector cosine distance. Sparse retrieval uses PostgreSQL full-text search. Hybrid retrieval combines ranked lists using reciprocal-rank fusion.
- BAAI/bge-reranker-base reranks the hybrid candidates. The pipeline currently uses code defaults of 20 hybrid candidates, 5 reranked results, and a 6,000-token context budget (with a character-based estimate in context assembly).
- Gemini generation uses evidence supplied by context assembly, validates cited source IDs against available sources, and returns an insufficient-evidence response when there are no usable sources.
- Image diagnosis services load project model weights under backend/app/ml/models/; Grad-CAM implementations are under backend/app/ml/xai/.

These descriptions document implementation, not clinical validation or diagnostic performance.

## Tests and retrieval evaluation

### Frontend

    cd frontend
    npm test
    npm run build

The frontend test script uses Node's built-in test runner and the tests are in frontend/tests/.

### Backend unit tests

The backend pytest.ini sets testpaths = tests, so running from backend/ discovers backend/tests/ by default:

    cd backend
    python -m pytest

Install pytest in the development environment if it is not already available. Several older or standalone backend/test_*.py files are outside the configured default test directory and are not automatically included by that command.

### Retrieval quality suite

backend/retrieval_quality/ contains a synthetic cardiology golden set, relevance metrics, a live retrieval adapter, and an ablation runner for sparse, dense, hybrid, and reranked retrieval. The live cases expect the corresponding synthetic patient/report data to exist in the configured database; the tests are not a database-independent mock suite.

From backend/:

    python -m pytest retrieval_quality/test_retrieval_quality.py
    python -m retrieval_quality.run_ablation --k 5

The ablation runner prints Hit@k, MRR, precision@k, report recall, and latency summaries and writes backend/ablation_results.json. It currently reports its configured synthetic patient identifier; keep generated evaluation outputs restricted and do not substitute real patient records into public reports. Running this suite or runner performs database reads and model inference; this README does not claim that either has been run.

## Current limitations and operational notes

- The database schema and initial data are not provisioned by a checked-in bootstrap script.
- The backend dependency manifest needs to be completed and pinned for the current ML, multipart, and generation imports.
- Department and Clinical Teams frontend routes currently render the generic module placeholder, despite the corresponding backend endpoints being present.
- The OCR service uses EasyOCR as a fallback when embedded PDF text is not usable; OCR and transformer model initialization can require substantial CPU, memory, and first-run model downloads.
- Keep clinical data, uploaded PDFs, access tokens, and evaluation outputs out of source control and ordinary logs.

## Contributing safely

- Reuse the authenticated frontend API client and existing backend routes.
- Preserve patient/case scope checks and workflow validation when changing clinical features.
- Keep evaluation data synthetic or appropriately deidentified and access-controlled.
- Do not treat a successful API response or fluent generated answer as proof of retrieval or clinical quality.

