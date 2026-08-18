from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvaluationRequest(BaseModel):
    """A single RAG interaction to evaluate."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    contexts: list[str] = Field(min_length=1)
    reference: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_context_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        has_context = "context" in normalized
        has_contexts = "contexts" in normalized
        if has_context and has_contexts:
            raise ValueError("use solo 'contexts' o 'context', no ambos")
        if has_context:
            normalized["contexts"] = normalized.pop("context")
        return normalized

    @field_validator("question", "answer")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("no puede estar vacio")
        return value

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, contexts: list[str]) -> list[str]:
        normalized: list[str] = []
        for context in contexts:
            stripped = context.strip()
            if not stripped:
                raise ValueError("cada contexto debe ser un string no vacio")
            normalized.append(stripped)
        return normalized

    @field_validator("reference")
    @classmethod
    def normalize_reference(cls, reference: str | None) -> str | None:
        if reference is None:
            return None
        stripped = reference.strip()
        return stripped or None


class MetricScores(BaseModel):
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


class EvaluationResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    context_items: int
    reference_provided: bool
    metrics: MetricScores
    evaluation_time_ms: int


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ragas-api"


class ConfigResponse(BaseModel):
    openai_configured: bool
    model: str
    embedding_model: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: str
