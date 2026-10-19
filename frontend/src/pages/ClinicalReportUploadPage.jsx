import { useRef, useState } from "react";
import { ArrowRight, CheckCircle2, FileText, FileUp, LoaderCircle, UploadCloud, X } from "lucide-react";
import { Link } from "react-router";
import Card from "../components/ui/Card.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import { parsePatientId } from "../services/clinicalAssistantApi.js";
import {
  clinicalReportApi,
  getClinicalReportUploadErrorMessage,
  isSupportedClinicalReportFile,
} from "../services/clinicalReportApi.js";
import { formatFileSize } from "../utils/formatters.js";

export default function ClinicalReportUploadPage() {
  const fileInputRef = useRef(null);
  const submitLockRef = useRef(false);
  const [patientIdInput, setPatientIdInput] = useState("");
  const [patientIdError, setPatientIdError] = useState("");
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [processedPatientId, setProcessedPatientId] = useState(null);

  const normalizedPatientId = parsePatientId(patientIdInput);

  function selectFile(candidate) {
    if (!candidate) return;
    setRequestError("");
    if (!isSupportedClinicalReportFile(candidate)) {
      setFileError(candidate.size === 0
        ? "This file is empty. Choose a readable PDF report."
        : "Choose a PDF report file.");
      return;
    }
    setFile(candidate);
    setFileError("");
  }

  function clearFile() {
    if (isSubmitting) return;
    setFile(null);
    setFileError("");
    setRequestError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitLockRef.current || isSubmitting) return;

    const patientId = parsePatientId(patientIdInput);
    if (!patientId) {
      setPatientIdError("Enter a positive whole-number Patient ID.");
      return;
    }
    if (!file || !isSupportedClinicalReportFile(file)) {
      setFileError("Choose a non-empty PDF report before uploading.");
      return;
    }

    submitLockRef.current = true;
    setIsSubmitting(true);
    setRequestError("");
    try {
      const result = await clinicalReportApi.process(patientId, file);
      setProcessedPatientId(result.patientId);
    } catch (error) {
      setRequestError(getClinicalReportUploadErrorMessage(error));
    } finally {
      submitLockRef.current = false;
      setIsSubmitting(false);
    }
  }

  function uploadAnotherReport() {
    setProcessedPatientId(null);
    setFile(null);
    setFileError("");
    setRequestError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <>
      <PageHeader
        eyebrow="Report intake"
        title="Upload medical report"
        description="Process a PDF report and connect it to a patient record."
      />

      {processedPatientId ? (
        <Card className="clinical-report-success-card" role="status" aria-live="polite">
          <div className="clinical-report-success-heading">
            <span><CheckCircle2 size={20} aria-hidden="true" /></span>
            <div>
              <h2>Processing complete</h2>
              <p>Report uploaded and processed successfully.</p>
            </div>
          </div>
          <div className="clinical-report-upload-actions">
            <Link
              className="patient-primary-button"
              to="/clinical-assistant"
              state={{ patientId: processedPatientId }}
            >
              Go to Clinical Assistant <ArrowRight size={15} aria-hidden="true" />
            </Link>
            <button className="patient-secondary-button" type="button" onClick={uploadAnotherReport}>
              Upload another report
            </button>
          </div>
        </Card>
      ) : (
        <Card className="clinical-report-upload-card">
          <form onSubmit={handleSubmit} noValidate>
            <div className="clinical-report-upload-intro">
              <h2>Report details</h2>
              <p>Select a patient ID and a PDF report to begin processing.</p>
            </div>

            <label className="clinical-report-patient-field" htmlFor="clinical-report-patient-id">
              <span>Patient ID</span>
              <input
                id="clinical-report-patient-id"
                type="text"
                inputMode="numeric"
                autoComplete="off"
                value={patientIdInput}
                aria-invalid={Boolean(patientIdError)}
                aria-describedby={patientIdError ? "clinical-report-patient-error" : "clinical-report-patient-help"}
                placeholder="Enter positive patient ID"
                disabled={isSubmitting}
                onBlur={() => {
                  if (!parsePatientId(patientIdInput)) setPatientIdError("Enter a positive whole-number Patient ID.");
                }}
                onChange={(event) => {
                  const value = event.target.value;
                  setPatientIdInput(value);
                  setPatientIdError(value && !parsePatientId(value) ? "Enter a positive whole-number Patient ID." : "");
                  setRequestError("");
                }}
              />
              {patientIdError ? (
                <small className="clinical-report-field-error" id="clinical-report-patient-error" role="alert">{patientIdError}</small>
              ) : normalizedPatientId ? (
                <small id="clinical-report-patient-help" role="status">Patient ID selected: <strong>{normalizedPatientId}</strong></small>
              ) : (
                <small id="clinical-report-patient-help">Use the numeric Patient ID from the patient record.</small>
              )}
            </label>

            <input
              ref={fileInputRef}
              className="sr-only"
              type="file"
              accept=".pdf,application/pdf"
              aria-label="Choose a medical report PDF"
              disabled={isSubmitting}
              onChange={(event) => {
                selectFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />

            {!file ? (
              <div
                className={`report-dropzone clinical-report-dropzone${isDragging ? " is-dragging" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false);
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  selectFile(event.dataTransfer.files?.[0]);
                }}
                aria-label="Medical report PDF drop area"
              >
                <span className="report-drop-icon"><UploadCloud size={23} aria-hidden="true" /></span>
                <strong>Drop your report PDF here</strong>
                <span>or</span>
                <button
                  className="clinical-report-browse-button"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isSubmitting}
                >Browse files</button>
                <small>PDF files only</small>
              </div>
            ) : (
              <div className="report-selected-file">
                <span className="report-file-icon"><FileText size={19} aria-hidden="true" /></span>
                <span className="report-file-copy">
                  <strong title={file.name}>{file.name}</strong>
                  <small>{formatFileSize(file.size)} · PDF document</small>
                </span>
                <button className="report-remove-file" type="button" aria-label="Remove selected report" onClick={clearFile} disabled={isSubmitting}>
                  <X size={17} aria-hidden="true" />
                </button>
                <button className="report-change-file" type="button" onClick={() => fileInputRef.current?.click()} disabled={isSubmitting}>
                  Replace file
                </button>
              </div>
            )}

            {fileError ? <div className="patient-form-error" role="alert">{fileError}</div> : null}
            {requestError ? <div className="patient-form-error" role="alert">{requestError}</div> : null}

            {isSubmitting ? (
              <div className="clinical-report-upload-status" role="status" aria-live="polite">
                <LoaderCircle className="patient-spin" size={16} aria-hidden="true" />
                <span>Uploading and processing report. This may take a few moments.</span>
              </div>
            ) : null}

            <div className="clinical-report-upload-actions">
              <button className="patient-primary-button" type="submit" disabled={!normalizedPatientId || !file || Boolean(fileError) || isSubmitting}>
                {isSubmitting ? <LoaderCircle className="patient-spin" size={16} aria-hidden="true" /> : <FileUp size={16} aria-hidden="true" />}
                {isSubmitting ? "Processing report..." : "Upload & Process Report"}
              </button>
            </div>
          </form>
        </Card>
      )}
    </>
  );
}
