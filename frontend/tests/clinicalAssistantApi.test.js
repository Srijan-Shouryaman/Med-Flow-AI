import assert from "node:assert/strict";
import test from "node:test";
import { ApiError } from "../src/services/apiClient.js";
import {
  buildClinicalQueryPayload,
  createClinicalAssistantService,
  getClinicalQueryErrorMessage,
  normalizeClinicalQueryResponse,
  parsePatientId,
} from "../src/services/clinicalAssistantApi.js";

const validResponse = {
  patient_id: 4,
  case_id: null,
  query: "Summarize the available clinical findings.",
  retrieval_count: 2,
  reranked_count: 2,
  context: {
    sources: [{ chunk_id: "chunk-1", patient_id: 4, report_id: 12, reranker_rank: 1 }],
  },
  generation: {
    answer: "The available records document a finding. [chunk-1]",
    cited_chunk_ids: ["chunk-1"],
    sources: [{ chunk_id: "chunk-1", patient_id: 4, report_id: 12, reranker_rank: 1 }],
  },
  needs_human_review: true,
};

test("validates patient IDs as positive safe integers", () => {
  assert.equal(parsePatientId("4"), 4);
  for (const value of ["", "0", "-2", "1.5", "1e2", "patient-4", "9007199254740992"]) {
    assert.equal(parsePatientId(value), null);
  }
});

test("builds the supported patient-scoped query payload", () => {
  assert.deepEqual(buildClinicalQueryPayload("4", "  Summarize findings.  "), {
    patient_id: 4,
    question: "Summarize findings.",
    case_id: null,
  });
  assert.throws(() => buildClinicalQueryPayload(4, "  "), /Enter a question/);
  assert.throws(() => buildClinicalQueryPayload(4, "x".repeat(5001)), /5,000/);
});

test("normalizes successful response and actual source fields", () => {
  const normalized = normalizeClinicalQueryResponse(validResponse, 4);
  assert.equal(normalized.answer, validResponse.generation.answer);
  assert.equal(normalized.needsHumanReview, true);
  assert.deepEqual(normalized.sources, validResponse.context.sources);
  assert.deepEqual(normalized.citedChunkIds, ["chunk-1"]);
});

test("rejects malformed or cross-patient responses without inventing data", () => {
  assert.throws(() => normalizeClinicalQueryResponse(null, 4), /incomplete response/);
  assert.throws(() => normalizeClinicalQueryResponse({ ...validResponse, generation: {} }, 4), /incomplete answer/);
  assert.throws(() => normalizeClinicalQueryResponse({ ...validResponse, patient_id: 9 }, 4), /different patient/);
  assert.throws(() => normalizeClinicalQueryResponse(validResponse, 4, "Another question"), /different question/);
  const incompleteOptional = normalizeClinicalQueryResponse({ ...validResponse, needs_human_review: undefined }, 4);
  assert.equal(incompleteOptional.needsHumanReview, null);
});

test("posts to the existing RAG endpoint and reports network and HTTP errors", async () => {
  let captured = null;
  const service = createClinicalAssistantService(async (path, options) => {
    captured = { path, options };
    return validResponse;
  });
  await service.query(4, "Summarize the available clinical findings.");
  assert.equal(captured.path, "/api/rag/query");
  assert.equal(captured.options.method, "POST");
  assert.deepEqual(captured.options.body, {
    patient_id: 4,
    question: "Summarize the available clinical findings.",
    case_id: null,
  });

  const networkService = createClinicalAssistantService(async () => {
    throw new ApiError("network", 0, "network_error");
  });
  await assert.rejects(networkService.query(4, "Retry me."), (error) => {
    assert.match(getClinicalQueryErrorMessage(error), /Could not connect/);
    return true;
  });

  for (const [status, detail, expected] of [
    [401, "expired", /session has expired/],
    [403, "You are not authorized to use clinical RAG queries.", /not authorized/],
    [422, "Question must not be blank.", /Question must not be blank/],
    [503, "The clinical answer generation service is temporarily unavailable.", /temporarily unavailable/],
  ]) {
    assert.match(getClinicalQueryErrorMessage(new ApiError("request failed", status, `http_${status}`, { detail })), expected);
  }
  assert.doesNotMatch(
    getClinicalQueryErrorMessage(new ApiError("request failed", 500, "http_500", { detail: "Traceback contains secret token" })),
    /Traceback|secret|token/i,
  );
});

test("a failed question can be retried with its original payload", async () => {
  let calls = 0;
  const service = createClinicalAssistantService(async (_path, options) => {
    calls += 1;
    if (calls === 1) throw new ApiError("unavailable", 503, "http_503");
    return validResponse;
  });

  await assert.rejects(service.query(4, "Summarize the available clinical findings."));
  const response = await service.query(4, "Summarize the available clinical findings.");
  assert.equal(calls, 2);
  assert.equal(response.patientId, 4);
  assert.equal(response.answer, validResponse.generation.answer);
});
