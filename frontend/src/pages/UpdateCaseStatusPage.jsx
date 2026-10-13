import { useState } from "react";
import { ArrowRight } from "lucide-react";
import Card from "../components/ui/Card.jsx";
import FeedbackState from "../components/ui/FeedbackState.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import StatusBadge from "../components/ui/StatusBadge.jsx";
import WorkflowCaseIdForm from "../components/WorkflowCaseIdForm.jsx";
import { useAuth } from "../auth/AuthContext.jsx";
import { hasRoleAccess } from "../config/navigation.js";
import { ApiError } from "../services/apiClient.js";
import { getCaseRecord, getWorkflowErrorMessage, updateCaseStatus } from "../services/workflowApi.js";
import { loadFailureKind } from "../utils/formatters.js";

const STATUS_UPDATE_ROLES = ["SUPER_ADMIN", "HOSPITAL_ADMIN", "DEPARTMENT_HEAD", "DOCTOR"];
const NEXT_STATUS = {
  Assigned: "In Progress",
  "In Progress": "Resolved",
};

function failure(error, fallback) {
  return {
    kind: loadFailureKind(error),
    message: getWorkflowErrorMessage(error, fallback),
  };
}

function statusUpdateErrorMessage(error, fallback) {
  const existingMessage = getWorkflowErrorMessage(error, fallback);
  if (existingMessage !== fallback) return existingMessage;

  const detail = error?.payload?.detail;
  if (
    typeof detail === "string" &&
    detail.trim().length > 0 &&
    detail.trim().length <= 180 &&
    !/[\r\n]/.test(detail) &&
    !/(traceback|stack trace|password|secret|token|bearer|sqlalchemy|psycopg|database url|postgresql:\/\/|https?:\/\/)/i.test(detail)
  ) {
    return detail.trim();
  }

  return fallback;
}

export default function UpdateCaseStatusPage() {
  const { user } = useAuth();
  const canUpdateStatus = hasRoleAccess(user, STATUS_UPDATE_ROLES);
  const [caseId, setCaseId] = useState("");
  const [busy, setBusy] = useState("");
  const [state, setState] = useState({ kind: "idle", caseRecord: null, error: null });
  const [actionError, setActionError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  async function loadCase(value = caseId) {
    const cleanId = value.trim();
    if (!cleanId || busy) return;

    setBusy("loading");
    setActionError("");
    setSuccessMessage("");
    setState({ kind: "loading", caseRecord: null, error: null });
    try {
      const caseRecord = await getCaseRecord(cleanId);
      setState({ kind: "success", caseRecord, error: null });
    } catch (error) {
      setState({
        kind: error?.status === 404 ? "not-found" : "error",
        caseRecord: null,
        error: failure(error, "Unable to load this case. Please try again."),
      });
    } finally {
      setBusy("");
    }
  }

  const caseRecord = state.kind === "success" ? state.caseRecord : null;
  const nextStatus = caseRecord ? NEXT_STATUS[caseRecord.status] ?? null : null;

  async function confirmTransition() {
    if (!caseRecord || !nextStatus || busy) return;
    const currentStatus = caseRecord.status;
    const confirmed = window.confirm(
      `Update ${caseRecord.case_id} from ${currentStatus} to ${nextStatus}?`,
    );
    if (!confirmed) return;

    setBusy("updating");
    setActionError("");
    setSuccessMessage("");
    try {
      const updated = await updateCaseStatus(caseRecord.case_id, nextStatus);
      if (updated.case_id !== caseRecord.case_id || typeof updated.status !== "string") {
        throw new ApiError("The server returned an unexpected case status response.", 502, "invalid_response");
      }

      setState((current) => ({
        ...current,
        caseRecord: { ...current.caseRecord, status: updated.status },
      }));
      setSuccessMessage(`${caseRecord.case_id} status updated to ${updated.status}.`);

      // There is no shared query cache in this app; refresh the case record after
      // the mutation so the page stays aligned with persisted case data.
      try {
        const refreshedCase = await getCaseRecord(caseRecord.case_id);
        setState((current) => ({ ...current, caseRecord: refreshedCase }));
      } catch {
        // The PATCH response is authoritative for the updated status.
      }
    } catch (error) {
      setActionError(statusUpdateErrorMessage(error, "Unable to update this case status. Please try again."));
    } finally {
      setBusy("");
    }
  }

  if (!canUpdateStatus) {
    return (
      <>
        <PageHeader eyebrow="AI Workspace" title="Update Case Status" description="View a case's current status and update it when the next workflow transition is permitted." />
        <Card className="workflow-action-card">
          <FeedbackState type="unauthorized" title="Status updates are unavailable for this account." description="Your role can view the workspace but cannot update case status." />
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader eyebrow="AI Workspace" title="Update Case Status" description="View a case's current status and update it when the next workflow transition is permitted." />

      <Card className="workflow-action-card assignments-action-card">
        <div className="assignments-action-layout">
          <div className="workflow-section-heading assignments-action-heading">
            <div>
              <h2>Load a case</h2>
              <p className="assignments-action-description">Enter a case ID to view its current status and available transitions.</p>
            </div>
          </div>
          <div className="assignments-action-controls">
            <WorkflowCaseIdForm
              value={caseId}
              onChange={setCaseId}
              onSubmit={() => loadCase()}
              busy={Boolean(busy)}
              submitLabel="Load case"
              placeholder="Enter case ID, e.g. C0102"
            />
          </div>
        </div>

        {state.kind === "loading" ? <FeedbackState type="loading" title="Loading case status..." compact /> : null}
        {state.kind === "not-found" ? (
          <div className="workflow-action-feedback">
            <FeedbackState type="empty" title="Case not found." description="Check the case ID and try again." compact />
          </div>
        ) : null}
        {state.kind === "error" ? (
          <div className="workflow-action-feedback">
            <FeedbackState type={state.error?.kind ?? "error"} title="Unable to load case status." description={state.error?.message} compact />
            <button className="patient-secondary-button" type="button" disabled={Boolean(busy)} onClick={() => loadCase()}>Retry</button>
          </div>
        ) : null}
      </Card>

      {caseRecord ? (
        <Card className="workflow-detail-card case-status-update-card">
          <div className="workflow-section-heading">
            <div>
              <p className="card-eyebrow">Case status</p>
              <h2>{caseRecord.case_id}</h2>
            </div>
            <StatusBadge status={caseRecord.status}>{caseRecord.status || "Unknown"}</StatusBadge>
          </div>

          <dl className="workflow-detail-grid">
            <div className="workflow-detail-value">
              <dt>Current status</dt>
              <dd>{caseRecord.status || "Unknown"}</dd>
            </div>
            <div className="workflow-detail-value">
              <dt>Available next status</dt>
              <dd>{nextStatus || "No status update is currently available"}</dd>
            </div>
          </dl>

          {actionError ? <div className="patient-form-error" role="alert">{actionError}</div> : null}
          {successMessage ? <div className="patient-success-banner case-status-success" role="status"><span className="patient-success-dot" aria-hidden="true" />{successMessage}</div> : null}

          {nextStatus ? (
            <div className="workflow-review-actions">
              <button className="patient-primary-button" type="button" disabled={Boolean(busy)} onClick={confirmTransition}>
                {busy === "updating" ? "Updating status..." : `Move to ${nextStatus}`}
                {busy === "updating" ? null : <ArrowRight size={14} aria-hidden="true" />}
              </button>
            </div>
          ) : (
            <FeedbackState type="empty" title="No status update is currently available." description="Only Assigned and In Progress cases have a permitted next status." compact />
          )}
        </Card>
      ) : null}
    </>
  );
}
