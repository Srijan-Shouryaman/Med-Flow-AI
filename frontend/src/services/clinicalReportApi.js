import { ApiError, apiClient } from "./apiClient.js";
import { parsePatientId } from "./clinicalAssistantApi.js";

const PROCESS_REPORT_PATH = "/api/clinical-nlp/process";

export function isSupportedClinicalReportFile(file) {
  return Boolean(
    file
    && Number(file.size) > 0
    && typeof file.name === "string"
    && /\.pdf$/i.test(file.name),
  );
}

export function getClinicalReportUploadErrorMessage(error) {
  if (error?.code === "invalid_patient_id") return error.message;
  if (error?.code === "invalid_report_file") return error.message;
  if (error?.status === 401) return "Your session has expired. Sign in again before uploading a report.";
  if (error?.status === 403) return "Your account does not have permission to process reports.";
  if (error?.status === 404) return "We could not find that Patient ID. Check the ID and try again.";
  if (error?.status === 413) return "This PDF is too large for the upload service. Choose a smaller file.";
  if (error?.status === 415) return "The upload service only accepts PDF reports.";
  if (error?.status === 422) return "Check the Patient ID and PDF file, then try again.";
  if (error?.status === 429) return "The report service is busy. Wait a moment and try again.";
  if (error?.code === "network_error" || error?.status === 0 || error?.status >= 500 || error?.code === "invalid_response") {
    return "We could not confirm processing. Check Report History before retrying to avoid a duplicate upload.";
  }
  return "The report could not be uploaded and processed. Please try again.";
}

export function createClinicalReportService(request = apiClient) {
  return {
    async process(patientId, file, { signal } = {}) {
      const normalizedPatientId = parsePatientId(patientId);
      if (!normalizedPatientId) {
        throw new ApiError("Enter a positive whole-number Patient ID.", 400, "invalid_patient_id");
      }
      if (!isSupportedClinicalReportFile(file)) {
        throw new ApiError("Choose a non-empty PDF report.", 400, "invalid_report_file");
      }

      const body = new FormData();
      body.append("patient_id", String(normalizedPatientId));
      body.append("file", file);

      const response = await request(PROCESS_REPORT_PATH, {
        method: "POST",
        body,
        signal,
      });

      if (!response || typeof response !== "object" || Array.isArray(response)
        || Number(response.patient_id) !== normalizedPatientId) {
        throw new ApiError("The report service returned an incomplete result. Please check Report History before retrying.", 502, "invalid_response");
      }

      return { patientId: normalizedPatientId };
    },
  };
}

export const clinicalReportApi = createClinicalReportService();
