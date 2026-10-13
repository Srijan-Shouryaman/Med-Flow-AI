import { ApiError, apiClient } from "./apiClient.js";

function casePath(caseId, suffix = "") {
  return `/api/cases/${encodeURIComponent(String(caseId).trim())}${suffix}`;
}

function requireList(value, label) {
  if (!Array.isArray(value)) {
    throw new ApiError(`The server returned an unexpected ${label} response.`, 502, "invalid_response");
  }
  return value;
}

function requireRecord(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ApiError(`The server returned an unexpected ${label} response.`, 502, "invalid_response");
  }
  return value;
}

async function list(path, label) {
  return requireList(await apiClient(path), label);
}

async function record(path, label, options) {
  return requireRecord(await apiClient(path, options), label);
}

export const getReports = () => list("/api/reports/", "reports");
export const getReport = (reportId) => record(`/api/reports/${encodeURIComponent(reportId)}`, "report");
export const getCaseRecord = (caseId) => record(casePath(caseId), "case");
export const updateCaseStatus = (caseId, status) => record(casePath(caseId, "/status"), "case status", {
  method: "PATCH",
  body: { status },
});
export const getCasePrediction = (caseId) => record(casePath(caseId, "/prediction"), "prediction");
export const createCasePrediction = (caseId) => record(casePath(caseId, "/prediction"), "prediction", { method: "POST" });
export const getPrediction = (predictionId) => record(`/api/predictions/${encodeURIComponent(predictionId)}`, "prediction");
export const getPredictionHistory = () => list("/api/predictions/", "predictions");
export const getCaseRecommendations = (caseId) => list(casePath(caseId, "/recommendations"), "recommendations");
export const getRecommendationsForCase = (caseId) => list(`/api/recommendations/${encodeURIComponent(String(caseId).trim())}`, "recommendations");
export const createCaseRecommendations = (caseId) => record(casePath(caseId, "/recommendations"), "recommendations", { method: "POST" });
export const getRecommendationHistory = () => list("/api/recommendations/", "recommendations");
export const getCaseAssignment = (caseId) => record(casePath(caseId, "/assignment"), "assignment");
export const getAssignment = (assignmentId) => record(`/api/assignments/${encodeURIComponent(assignmentId)}`, "assignment");
export const getAssignmentHistory = () => list("/api/assignments/", "assignments");
export const getTeams = () => list("/api/teams/", "teams");

export function approveCaseAssignment(caseId, approvingUser) {
  return record(casePath(caseId, "/assignment/approve"), "assignment", {
    method: "POST",
    body: { approving_user: approvingUser },
  });
}

export function overrideCaseAssignment(caseId, { teamId, reason, approvingUser }) {
  return record(casePath(caseId, "/assignment/override"), "assignment", {
    method: "POST",
    body: {
      selected_team_id: teamId,
      override_reason: reason,
      approving_user: approvingUser,
    },
  });
}

export function getWorkflowErrorMessage(error, fallback = "The request could not be completed. Please try again.") {
  if (error?.status === 404) {
    const detail = error?.payload?.detail;
    if (typeof detail === "string" && /case not found/i.test(detail)) return "Case not found.";
  }
  if ([400, 409, 422].includes(error?.status)) {
    const detail = error?.payload?.detail;
    if (typeof detail === "string" && detail.length < 240) return detail;
  }
  if (error?.status === 401 || error?.status === 403 || error?.code === "network_error") {
    return error.message;
  }
  return fallback;
}
