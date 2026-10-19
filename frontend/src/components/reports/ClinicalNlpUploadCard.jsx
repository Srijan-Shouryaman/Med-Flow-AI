import { useId, useRef, useState } from "react";
import { CheckCircle2, ChevronDown, FileText, FileUp, LoaderCircle, UploadCloud, X } from "lucide-react";
import Card from "../ui/Card.jsx";
import { parsePatientId } from "../../services/clinicalAssistantApi.js";
import {
  clinicalReportApi,
  getClinicalReportUploadErrorMessage,
  isSupportedClinicalReportFile,
} from "../../services/clinicalReportApi.js";
import { formatFileSize } from "../../utils/formatters.js";

export default function ClinicalNlpUploadCard({ patientId }) {
  const panelId = useId();
  const fileInputRef = useRef(null);
  const submitLockRef = useRef(false);
  const normalizedPatientId = parsePatientId(patientId);
  const [isExpanded, setIsExpanded] = useState(false);
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  function selectFile(candidate) {
    if (!candidate) return;
    setRequestError("");
    setIsComplete(false);
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
    setIsComplete(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitLockRef.current || isSubmitting || isComplete) return;
    if (!normalizedPatientId) {
      setRequestError("This patient record does not have a valid Patient ID. Reload the patient profile and try again.");
      return;
    }
    if (!file || !isSupportedClinicalReportFile(file)) {
      setFileError("Choose a non-empty PDF report before processing.");
      return;
    }

    submitLockRef.current = true;
    setIsSubmitting(true);
    setRequestError("");
    try {
      await clinicalReportApi.process(normalizedPatientId, file);
      setIsComplete(true);
    } catch (error) {
      setRequestError(getClinicalReportUploadErrorMessage(error));
    } finally {
      submitLockRef.current = false;
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="patient-detail-card clinical-nlp-card">
      <button
        className="clinical-nlp-toggle"
        type="button"
        aria-expanded={isExpanded}
        aria-controls={panelId}
        aria-label={`${isExpanded ? "Collapse" : "Expand"} Clinical NLP Processing`}
        onClick={() => setIsExpanded((expanded) => !expanded)}
      >
        <span className="patient-profile-icon"><FileText size={19} aria-hidden="true" /></span>
        <span className="clinical-nlp-toggle-title">Clinical NLP Processing</span>
        <ChevronDown className={isExpanded ? "is-expanded" : ""} size={16} aria-hidden="true" />
      </button>

      <div id={panelId} className="clinical-nlp-content" hidden={!isExpanded}>
        <p className="clinical-nlp-description">
          Upload a medical report for text extraction, preprocessing, chunking, embedding, and storage.
        </p>
      <div className="clinical-nlp-patient-context" aria-label={`Patient ID ${normalizedPatientId ?? "unavailable"}`}>
        <span>Patient ID</span>
        <strong>{normalizedPatientId ?? "Unavailable"}</strong>
      </div>
      {!normalizedPatientId ? (
        <div className="patient-form-error clinical-nlp-patient-error" role="alert">
          Patient ID is unavailable for this record. Processing is disabled until the patient profile is reloaded.
        </div>
      ) : null}

      <form onSubmit={handleSubmit} noValidate>
        <input
          ref={fileInputRef}
          className="sr-only"
          type="file"
          accept=".pdf,application/pdf"
          aria-label="Choose a clinical report PDF"
          disabled={isSubmitting}
          onChange={(event) => {
            selectFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />

        {!file ? (
          <div
            className={`report-dropzone clinical-nlp-dropzone${isDragging ? " is-dragging" : ""}`}
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
            aria-label="Clinical report PDF drop area"
          >
            <span className="report-drop-icon"><UploadCloud size={20} aria-hidden="true" /></span>
            <strong>Drop a PDF report here</strong>
            <span>or</span>
            <button
              className="clinical-nlp-browse-button"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isSubmitting}
            >Browse files</button>
            <small>PDF files only</small>
          </div>
        ) : (
          <div className="report-selected-file">
            <span className="report-file-icon"><FileText size={18} aria-hidden="true" /></span>
            <span className="report-file-copy">
              <strong title={file.name}>{file.name}</strong>
              <small>{formatFileSize(file.size)} · PDF document</small>
            </span>
            <button className="report-remove-file" type="button" aria-label="Remove selected report" onClick={clearFile} disabled={isSubmitting}>
              <X size={16} aria-hidden="true" />
            </button>
            <button className="report-change-file" type="button" onClick={() => fileInputRef.current?.click()} disabled={isSubmitting}>
              Replace file
            </button>
          </div>
        )}

        {fileError ? <div className="patient-form-error" role="alert">{fileError}</div> : null}
        {requestError ? <div className="patient-form-error" role="alert">{requestError}</div> : null}
        {isSubmitting ? (
          <div className="clinical-nlp-status" role="status" aria-live="polite">
            <LoaderCircle className="patient-spin" size={16} aria-hidden="true" />
            <span>Processing clinical report...</span>
          </div>
        ) : null}
        {isComplete ? (
          <div className="clinical-nlp-status is-success" role="status" aria-live="polite">
            <CheckCircle2 size={16} aria-hidden="true" />
            <span>Report uploaded and processed successfully.</span>
          </div>
        ) : null}

        <div className="clinical-nlp-actions">
          <button
            className="patient-primary-button patient-action-button"
            type="submit"
            disabled={!normalizedPatientId || !file || Boolean(fileError) || isSubmitting || isComplete}
          >
            {isSubmitting ? <LoaderCircle className="patient-spin" size={16} aria-hidden="true" /> : <FileUp size={16} aria-hidden="true" />}
            {isSubmitting ? "Processing report..." : "Process Clinical Report"}
          </button>
        </div>
      </form>
      </div>
    </Card>
  );
}
