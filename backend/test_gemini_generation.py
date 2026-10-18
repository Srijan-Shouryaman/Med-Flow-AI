
import json
from unittest.mock import patch, MagicMock

from app.services.generation.gemini_generation_service import (
    generate_clinical_answer,
    GenerationServiceError,
)


def make_context():
    return {
        "patient_id": 4,
        "case_id": "C0148",
        "query": "What symptoms are documented?",
        "context": (
            "[Source: chunk-1 | Report: 56]\n"
            "The patient reports abdominal pain and nausea."
        ),
        "sources": [
            {
                "chunk_id": "chunk-1",
                "patient_id": 4,
                "case_id": "C0148",
                "report_id": 56,
                "reranker_rank": 1,
                "reranker_score": 2.0,
                "dense_rank": 1,
                "sparse_rank": 1,
                "rrf_score": 0.03,
            }
        ],
        "included_chunk_count": 1,
        "excluded_chunk_count": 0,
        "duplicate_chunk_count": 0,
        "token_count": 30,
        "token_budget": 1000,
        "has_evidence": True,
    }


def test_empty_evidence():
    context = make_context()
    context["context"] = ""
    context["sources"] = []
    context["has_evidence"] = False

    with patch(
        "app.services.generation.gemini_generation_service.genai.Client"
    ) as client:
        result = generate_clinical_answer(context)
        client.assert_not_called()

    assert result["insufficient_evidence"] is True
    assert result["cited_chunk_ids"] == []
    assert result["needs_human_review"] is True

    print("[PASS] Empty evidence skips Gemini")


def test_valid_response():
    context = make_context()

    response_payload = {
        "answer": (
            "Abdominal pain and nausea are documented [chunk-1]."
        ),
        "cited_chunk_ids": ["chunk-1"],
        "insufficient_evidence": False,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = json.dumps(
        response_payload
    )

    with patch(
        "app.services.generation.gemini_generation_service.genai.Client",
        return_value=mock_client,
    ):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-2.5-flash",
            },
        ):
            result = generate_clinical_answer(context)

    assert result["cited_chunk_ids"] == ["chunk-1"]
    assert result["sources"][0]["report_id"] == 56
    assert result["insufficient_evidence"] is False
    assert result["needs_human_review"] is True

    print("[PASS] Valid response and source mapping")


def test_unknown_citation_rejected():
    context = make_context()

    response_payload = {
        "answer": "A clinical finding is documented [unknown-chunk].",
        "cited_chunk_ids": ["unknown-chunk"],
        "insufficient_evidence": False,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = json.dumps(
        response_payload
    )

    with patch(
        "app.services.generation.gemini_generation_service.genai.Client",
        return_value=mock_client,
    ):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key"},
        ):
            try:
                generate_clinical_answer(context)
            except GenerationServiceError:
                pass
            else:
                raise AssertionError(
                    "Unknown citation was not rejected"
                )

    print("[PASS] Unknown citation rejected")


def test_invalid_json_rejected():
    context = make_context()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = (
        "This is not valid JSON"
    )

    with patch(
        "app.services.generation.gemini_generation_service.genai.Client",
        return_value=mock_client,
    ):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key"},
        ):
            try:
                generate_clinical_answer(context)
            except GenerationServiceError:
                pass
            else:
                raise AssertionError(
                    "Invalid JSON was not rejected"
                )

    print("[PASS] Invalid JSON rejected")


def test_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        try:
            generate_clinical_answer(make_context())
        except GenerationServiceError as exc:
            assert "GEMINI_API_KEY" in str(exc)
        else:
            raise AssertionError(
                "Missing API key was not detected"
            )

    print("[PASS] Missing API key handled")


if __name__ == "__main__":
    tests = [
        test_empty_evidence,
        test_valid_response,
        test_unknown_citation_rejected,
        test_invalid_json_rejected,
        test_missing_api_key,
    ]

    for test in tests:
        test()

    print("\nRESULT: GEMINI GENERATION TESTS PASSED")
