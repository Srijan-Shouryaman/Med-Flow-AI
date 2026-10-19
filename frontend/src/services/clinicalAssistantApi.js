import { ApiError, apiClient } from "./apiClient.js";

const RAG_QUERY_PATH = "/api/rag/query";
const SAFE_DETAIL_EXCLUSIONS = /(traceback|stack trace|password|secret|token|bearer|sqlalchemy|psycopg|database url|postgresql:\/\/|https?:\/\/)/i;

export function parsePatientId(value) {
  const normalized = String(value ?? "").trim();
  if (!/^\d+$/.test(normalized)) return null;

  const patientId = Number(normalized);
  return Number.isSafeInteger(patientId) && patientId > 0 ? patientId : null;
}

export function buildClinicalQueryPayload(patientId, question) {
  const normalizedPatientId = parsePatientId(patientId);
  if (!normalizedPatientId) {
    throw new ApiError("Enter a valid positive patient ID before sending a question.", 400, "invalid_patient_id");
  }

  if (typeof question !== "string" || !question.trim()) {
    throw new ApiError("Enter a question before sending it.", 400, "empty_question");
  }

  const normalizedQuestion = question.trim();
  if (normalizedQuestion.length > 5000) {
    throw new ApiError("Questions must be 5,000 characters or fewer.", 422, "question_too_long");
  }

  return { patient_id: normalizedPatientId, question: normalizedQuestion, case_id: null };
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validSources(value) {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

export function normalizeClinicalQueryResponse(response, expectedPatientId, expectedQuestion = "") {
  if (!isRecord(response)) {
    throw new ApiError("The clinical service returned an incomplete response. Please retry.", 502, "invalid_response");
  }

  const generation = isRecord(response.generation) ? response.generation : null;
  const answer = typeof generation?.answer === "string" ? generation.answer.trim() : "";
  if (!answer) {
    throw new ApiError("The clinical service returned an incomplete answer. Please retry.", 502, "invalid_response");
  }

  if (response.patient_id !== expectedPatientId) {
    throw new ApiError("The clinical service returned a response for a different patient. Please retry.", 502, "patient_context_mismatch");
  }
  if (expectedQuestion && response.query !== expectedQuestion) {
    throw new ApiError("The clinical service returned a response for a different question. Please retry.", 502, "question_context_mismatch");
  }

  const context = isRecord(response.context) ? response.context : null;
  const contextSources = validSources(context?.sources);
  const generationSources = validSources(generation?.sources);
  const citedChunkIds = Array.isArray(generation?.cited_chunk_ids)
    ? generation.cited_chunk_ids.filter((value) => typeof value === "string")
    : [];

  return {
    answer,
    patientId: expectedPatientId,
    sources: contextSources.length ? contextSources : generationSources,
    citedChunkIds,
    needsHumanReview: typeof response.needs_human_review === "boolean"
      ? response.needs_human_review
      : typeof generation?.needs_human_review === "boolean"
        ? generation.needs_human_review
        : null,
  };
}

function safeServerDetail(error) {
  const detail = error?.payload?.detail;
  const message = typeof detail === "string"
    ? detail.trim()
    : Array.isArray(detail)
      ? detail.map((item) => typeof item?.msg === "string" ? item.msg : "").filter(Boolean).join(" ")
      : "";

  if (!message || message.length > 180 || /[\r\n]/.test(message) || SAFE_DETAIL_EXCLUSIONS.test(message)) return "";
  return message;
}

export function getClinicalQueryErrorMessage(error) {
  if (error?.name === "AbortError") return "This request was canceled. You can retry the question.";
  if (error?.status === 401) return "Your session has expired. Sign in again before retrying.";
  if (error?.status === 403) return safeServerDetail(error) || "Your account is not authorized to query clinical records for this patient.";
  if (error?.status === 422) return safeServerDetail(error) || "The patient ID or question could not be validated. Check the entries and retry.";
  if (error?.status === 503) return safeServerDetail(error) || "The clinical answer service is temporarily unavailable. Please retry shortly.";
  if (error?.code === "network_error" || error?.status === 0) return "Could not connect to the clinical service. Check your connection and retry.";
  if (["invalid_response", "patient_context_mismatch", "question_context_mismatch"].includes(error?.code)) return error.message;
  return "An unexpected error interrupted this clinical query. Please retry.";
}

export function createClinicalAssistantService(request = apiClient) {
  return {
    async query(patientId, question, { signal } = {}) {
      const payload = buildClinicalQueryPayload(patientId, question);
      const response = await request(RAG_QUERY_PATH, {
        method: "POST",
        body: payload,
        signal,
      });
      return normalizeClinicalQueryResponse(response, payload.patient_id, payload.question);
    },
  };
}

export const clinicalAssistantApi = createClinicalAssistantService();
