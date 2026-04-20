from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float = Field(ge=0.0, le=1.0)


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    use_llm: bool = Field(default=False, description="If true, call configured LLM; else return retrieval-only answer.")


class RagChunk(BaseModel):
    source: str
    text: str
    score: float


class RagAskResponse(BaseModel):
    answer: str
    model: Optional[str] = None
    retrieved: List[RagChunk]

