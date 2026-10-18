
"""MedCPT-based clinical text embedding service."""

import logging
import math
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

# Backward-compatible constant for existing imports.
MODEL_NAME = ARTICLE_MODEL_NAME

EMBEDDING_DIMENSION = 768
MAX_INPUT_TOKENS = 512


class MedCPTEmbeddingService:
    """Generate document and query embeddings using MedCPT."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Existing Article Encoder.
        self.tokenizer = None
        self.model = None

        # Separate Query Encoder for retrieval.
        self.query_tokenizer = None
        self.query_model = None

    def _load_encoder(self, model_name: str):
        """Load one encoder and tokenizer with validation."""
        try:
            logger.info(
                "Loading MedCPT encoder",
                extra={
                    "model_name": model_name,
                    "device": str(self.device),
                },
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Unexpected embedding dimension: {hidden_size}; "
                    f"expected {EMBEDDING_DIMENSION}"
                )

            logger.info(
                "MedCPT encoder loaded",
                extra={
                    "model_name": model_name,
                    "embedding_dimension": hidden_size,
                    "device": str(self.device),
                },
            )

            return tokenizer, model

        except Exception as exc:
            logger.exception(
                "Failed to load MedCPT encoder",
                extra={"model_name": model_name},
            )
            raise RuntimeError(
                f"Could not load MedCPT encoder {model_name}"
            ) from exc

    def load_model(self) -> None:
        """Load the Article Encoder once."""
        if self.model is not None and self.tokenizer is not None:
            return

        tokenizer, model = self._load_encoder(ARTICLE_MODEL_NAME)
        self.tokenizer = tokenizer
        self.model = model

    def load_query_model(self) -> None:
        """Load the Query Encoder once."""
        if (
            self.query_model is not None
            and self.query_tokenizer is not None
        ):
            return

        tokenizer, model = self._load_encoder(QUERY_MODEL_NAME)
        self.query_tokenizer = tokenizer
        self.query_model = model

    def _encode_text(self, text: str, tokenizer, model) -> list[float]:
        """Encode text with a loaded MedCPT encoder."""
        try:
            token_count = len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )

            if token_count > MAX_INPUT_TOKENS:
                logger.warning(
                    "Text exceeds MedCPT token limit; truncation applied",
                    extra={
                        "token_count": token_count,
                        "max_input_tokens": MAX_INPUT_TOKENS,
                    },
                )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )

            inputs = {
                name: tensor.to(self.device)
                for name, tensor in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]

            vector = embedding[0].detach().cpu().float().tolist()

            if len(vector) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"received {len(vector)}"
                )

            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("Embedding contains non-finite values")

            return vector

        except (TypeError, ValueError):
            raise
        except Exception as exc:
            logger.exception(
                "MedCPT encoding failed",
                extra={"device": str(self.device)},
            )
            raise RuntimeError("Failed to generate MedCPT embedding") from exc

    def encode(self, text: str) -> list[float]:
        """Encode one clinical document using the Article Encoder."""
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string")

        if not text.strip():
            raise ValueError("Embedding input cannot be empty")

        self.load_model()
        return self._encode_text(text, self.tokenizer, self.model)

    def encode_query(self, text: str) -> list[float]:
        """Encode a search query using the MedCPT Query Encoder."""
        if not isinstance(text, str):
            raise TypeError("Query input must be a string")

        if not text.strip():
            raise ValueError("Query input cannot be empty")

        self.load_query_model()
        return self._encode_text(
            text,
            self.query_tokenizer,
            self.query_model,
        )

    def encode_batch(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        """Encode multiple clinical documents using the Article Encoder."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("Expected a sequence of text strings")

        if not texts:
            raise ValueError("Embedding batch cannot be empty")

        self.load_model()
        return [self.encode(text) for text in texts]


# Reuse one service instance within the application process.
embedding_service = MedCPTEmbeddingService()
