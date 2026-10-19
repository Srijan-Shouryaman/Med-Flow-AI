import assert from "node:assert/strict";
import test from "node:test";
import { File } from "node:buffer";
import { ApiError } from "../src/services/apiClient.js";
import {
  createClinicalReportService,
  getClinicalReportUploadErrorMessage,
  isSupportedClinicalReportFile,
} from "../src/services/clinicalReportApi.js";

function makePdf(name = "report.pdf", type = "application/pdf", contents = "%PDF-1.7") {
  return new File([contents], name, { type });
}

test("accepts non-empty PDF files by extension even when browser MIME differs", () => {
  assert.equal(isSupportedClinicalReportFile(makePdf()), true);
  assert.equal(isSupportedClinicalReportFile(makePdf("report.pdf", "application/octet-stream")), true);
  assert.equal(isSupportedClinicalReportFile(makePdf("report.pdf", "application/x-pdf")), true);
  assert.equal(isSupportedClinicalReportFile(makePdf("report.pdf", "")), true);
  assert.equal(isSupportedClinicalReportFile(makePdf("report.txt")), false);
  assert.equal(isSupportedClinicalReportFile(makePdf("report.pdf", "application/pdf", "")), false);
});

test("posts multipart patient_id and file to the existing processing endpoint", async () => {
  let capturedRequest;
  const service = createClinicalReportService(async (path, options) => {
    capturedRequest = { path, options };
    return { patient_id: 12, report_id: 27, case_id: "C0127" };
  });

  const result = await service.process("12", makePdf("patient-report.pdf", "application/octet-stream"));

  assert.equal(capturedRequest.path, "/api/clinical-nlp/process");
  assert.equal(capturedRequest.options.method, "POST");
  assert.equal(capturedRequest.options.body.get("patient_id"), "12");
  assert.equal(capturedRequest.options.body.get("file").name, "patient-report.pdf");
  assert.equal(capturedRequest.options.headers, undefined);
  assert.deepEqual(result, { patientId: 12 });
});

test("rejects invalid inputs before making a request", async () => {
  let requestCount = 0;
  const service = createClinicalReportService(async () => {
    requestCount += 1;
    return { patient_id: 12 };
  });

  await assert.rejects(service.process("0", makePdf()), { code: "invalid_patient_id" });
  await assert.rejects(service.process("12", makePdf("report.txt")), { code: "invalid_report_file" });
  assert.equal(requestCount, 0);
});

test("requires the backend to confirm the selected patient before success", async () => {
  const service = createClinicalReportService(async () => ({ patient_id: 13 }));
  await assert.rejects(
    service.process("12", makePdf()),
    (error) => error instanceof ApiError && error.code === "invalid_response",
  );
});

test("maps auth and network failures to safe user-facing upload messages", () => {
  assert.match(getClinicalReportUploadErrorMessage(new ApiError("unauthorized", 401)), /session has expired/i);
  assert.match(getClinicalReportUploadErrorMessage(new ApiError("offline", 0, "network_error")), /check Report History/i);
});
