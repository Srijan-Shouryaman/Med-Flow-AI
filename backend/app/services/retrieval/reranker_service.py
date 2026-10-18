
"""Cross-encoder reranking for MedFlow AI clinical retrieval."""

import logging
import math
import threading
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
MAX_TEXT_LENGTH = 12000
MAX_QUERY_LENGTH = 2000
MAX_BATCH_SIZE = 32

_model = None
_tokenizer = None
_model_name_loaded = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _load_model(model_name: str) -> None:
    """Load the tokenizer and model once, using CPU inference."""
    global _model, _tokenizer, _model_name_loaded

    with _model_lock:
        if _model is not None and _model_name_loaded == model_name:
            return

        try:
            logger.info(
                "Loading cross-encoder reranker",
                extra={"model_name": model_name, "device": "cpu"},
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            model.to("cpu")
            model.eval()

            _tokenizer = tokenizer
            _model = model
            _model_name_loaded = model_name

            logger.info(
                "Cross-encoder reranker loaded",
                extra={"model_name": model_name, "device": "cpu"},
            )

        except Exception:
            logger.exception(
                "Failed to load cross-encoder reranker",
                extra={"model_name": model_name},
            )
            raise


def rerank_clinical_chunks(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = MAX_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Rerank hybrid retrieval candidates by query–chunk relevance.

    Each returned candidate preserves its original fields and gains a
    reranker_score and reranker_rank.

    Scores are raw model logits, not probabilities.
    The caller must authorize patient/case access before retrieval.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query must not exceed {MAX_QUERY_LENGTH} characters"
        )

    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between 1 and {MAX_BATCH_SIZE}"
        )

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    model_name = model_name.strip()

    if len(candidates) > MAX_TOP_K:
        raise ValueError(
            f"candidates must not exceed {MAX_TOP_K} items"
        )

    if not candidates:
        return []

    # Validate the entire input before loading the model.
    prepared = []

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"Candidate {index} must be a dictionary")

        chunk_id = candidate.get("chunk_id")
        chunk_text = candidate.get("chunk_text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Candidate {index} has an invalid chunk_id"
            )

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(
                f"Candidate {chunk_id} has empty or invalid chunk_text"
            )

        if len(chunk_text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Candidate {chunk_id} exceeds the "
                f"{MAX_TEXT_LENGTH}-character text limit"
            )

        prepared.append((candidate, chunk_id, chunk_text.strip()))

    chunk_ids = [item[1] for item in prepared]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Candidates contain duplicate chunk_id values")

    _load_model(model_name)

    scores = []

    try:
        # Protect shared model inference when requests run concurrently.
        with _inference_lock:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                pairs = [
                    [query, chunk_text]
                    for _, _, chunk_text in batch
                ]

                inputs = _tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    outputs = _model(**inputs)
                    logits = outputs.logits

                # BGE reranker-base normally produces one logit per pair.
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                elif logits.ndim == 1:
                    batch_scores = logits
                else:
                    raise RuntimeError(
                        "Unexpected reranker output shape: "
                        f"{tuple(logits.shape)}"
                    )

                batch_scores = batch_scores.detach().cpu().tolist()

                if len(batch_scores) != len(batch):
                    raise RuntimeError(
                        "Reranker output count does not match input count"
                    )

                for score in batch_scores:
                    score = float(score)
                    if not math.isfinite(score):
                        raise RuntimeError(
                            "Reranker produced a non-finite score"
                        )
                    scores.append(score)

    except Exception:
        logger.exception(
            "Clinical chunk reranking failed",
            extra={
                "model_name": model_name,
                "candidate_count": len(prepared),
                "batch_size": batch_size,
            },
        )
        raise

    reranked = []

    for (candidate, _, _), score in zip(prepared, scores):
        item = dict(candidate)
        item["reranker_score"] = score
        reranked.append(item)

    reranked.sort(
        key=lambda item: (
            -item["reranker_score"],
            item["chunk_id"],
        )
    )

    final_results = reranked[:top_k]

    for rank, item in enumerate(final_results, start=1):
        item["reranker_rank"] = rank

    logger.info(
        "Clinical chunk reranking completed",
        extra={
            "model_name": model_name,
            "candidate_count": len(prepared),
            "returned_count": len(final_results),
            "top_k": top_k,
        },
    )

    return final_results
