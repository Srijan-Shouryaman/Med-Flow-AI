
"""Unit tests for the MedCPT embedding service.

These tests use mocked model components. They do not download MedCPT
or require a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from app.services.embedding import medcpt_embedding_service as embedding_module


class FakeTokenizer:
    """Simulate tokenization and record truncation settings."""

    def __init__(self):
        self.last_truncation = None
        self.last_max_length = None

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
    ):
        # One simulated token per word.
        return text.split()

    def __call__(
        self,
        text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=None,
    ):
        self.last_truncation = truncation
        self.last_max_length = max_length

        token_count = len(text.split())

        if truncation and max_length is not None:
            token_count = min(token_count, max_length)

        # Simulate token IDs and attention mask.
        return {
            "input_ids": torch.ones(
                (1, token_count), dtype=torch.long
            ),
            "attention_mask": torch.ones(
                (1, token_count), dtype=torch.long
            ),
        }


class FakeModel:
    """Simulate a transformer returning 768-dimensional hidden states."""

    def __init__(self, fail_inference=False):
        self.config = SimpleNamespace(hidden_size=768)
        self.fail_inference = fail_inference

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        if self.fail_inference:
            raise RuntimeError("Simulated inference failure")

        token_count = inputs["input_ids"].shape[1]

        # Deterministic, finite output for predictable tests.
        hidden_states = torch.ones(
            (1, token_count, 768),
            dtype=torch.float32,
        )

        return SimpleNamespace(
            last_hidden_state=hidden_states
        )


@pytest.fixture
def fake_service(monkeypatch):
    """Create a fresh service with mocked tokenizer and model."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    monkeypatch.setattr(
        embedding_module.AutoTokenizer,
        "from_pretrained",
        lambda model_name: tokenizer,
    )
    monkeypatch.setattr(
        embedding_module.AutoModel,
        "from_pretrained",
        lambda model_name: model,
    )

    service = embedding_module.MedCPTEmbeddingService(
        device="cpu"
    )

    return service, tokenizer, model


def test_encode_returns_valid_768_dimension_vector(fake_service):
    service, _, _ = fake_service

    vector = service.encode(
        "Patient reports abdominal discomfort and heartburn."
    )

    assert len(vector) == 768
    assert all(isinstance(value, float) for value in vector)
    assert all(torch.isfinite(torch.tensor(value)).item()
               for value in vector)


def test_encode_rejects_empty_text(fake_service):
    service, _, _ = fake_service

    with pytest.raises(ValueError, match="cannot be empty"):
        service.encode("   ")


def test_encode_rejects_non_string_input(fake_service):
    service, _, _ = fake_service

    with pytest.raises(TypeError, match="must be a string"):
        service.encode(None)


def test_encode_truncates_oversized_text(fake_service, caplog):
    service, tokenizer, _ = fake_service

    long_text = "clinical " * 600

    with caplog.at_level("WARNING"):
        vector = service.encode(long_text)

    assert len(vector) == 768
    assert tokenizer.last_truncation is True
    assert tokenizer.last_max_length == 512
    assert "exceeds embedding token limit" in caplog.text


def test_model_is_loaded_only_once(fake_service, monkeypatch):
    service, _, _ = fake_service
    load_calls = {"count": 0}

    original_loader = embedding_module.AutoModel.from_pretrained

    def counted_loader(model_name):
        load_calls["count"] += 1
        return original_loader(model_name)

    monkeypatch.setattr(
        embedding_module.AutoModel,
        "from_pretrained",
        counted_loader,
    )

    service.encode("First clinical report.")
    service.encode("Second clinical report.")

    assert load_calls["count"] == 1


def test_model_loading_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        embedding_module.AutoTokenizer,
        "from_pretrained",
        lambda model_name: (_ for _ in ()).throw(
            OSError("Simulated model download failure")
        ),
    )

    service = embedding_module.MedCPTEmbeddingService(
        device="cpu"
    )

    with pytest.raises(
        RuntimeError,
        match="Could not load embedding model",
    ):
        service.encode("Clinical report text.")


def test_inference_failure_is_reported(fake_service):
    service, _, model = fake_service
    model.fail_inference = True

    with pytest.raises(
        RuntimeError,
        match="Failed to generate clinical text embedding",
    ):
        service.encode("Clinical report text.")


def test_batch_rejects_empty_input(fake_service):
    service, _, _ = fake_service

    with pytest.raises(ValueError, match="batch cannot be empty"):
        service.encode_batch([])


def test_batch_rejects_a_single_string(fake_service):
    service, _, _ = fake_service

    with pytest.raises(TypeError, match="sequence of text strings"):
        service.encode_batch("This is not a batch.")
