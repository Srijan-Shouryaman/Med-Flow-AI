
from pydantic import BaseModel, ConfigDict, Field


class RAGQueryRequest(BaseModel):
    """Request schema for patient-level clinical RAG queries."""

    model_config = ConfigDict(extra="forbid")

    patient_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=5000)
    case_id: str | None = Field(default=None, min_length=1, max_length=100)


class RAGQueryResponse(BaseModel):
    """Response schema for the clinical RAG pipeline."""

    patient_id: int
    case_id: str | None
    query: str
    retrieval_count: int
    reranked_count: int
    context: dict
    generation: dict
    needs_human_review: bool
