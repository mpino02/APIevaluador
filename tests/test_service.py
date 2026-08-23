import math
from types import SimpleNamespace

import pytest
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import SemanticSimilarity

from app.config import Settings
from app.schemas import EvaluationRequest
from app.services import ragas_service
from app.services.ragas_service import MissingConfigurationError, RagasEvaluationService, normalize_score


QUESTION = "¿Cuál es el objetivo del procedimiento?"
ANSWER = (
    "Establecer el flujo que se debe ejecutar para dar cumplimiento a las solicitudes "
    "de creaciones, modificaciones o bajas de cuentas de sistemas que son requeridas "
    "por el proceso de Alta, Baja o Modificación de usuarios."
)
REFERENCE = (
    "Establecer el flujo que se debe ejecutar para dar cumplimiento a las solitudes "
    "de creaciones, modificaciones o bajas de cuentas de sistemas que son requeridas "
    "por el proceso de Alta, Baja o Modificación de usuarios."
)
CONTEXTS = [f"1 Objetivo\n{ANSWER}"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, "not-a-number"])
def test_normalize_score_returns_none_for_non_json_values(value) -> None:
    assert normalize_score(value) is None


def test_normalize_score_accepts_metric_result_shape() -> None:
    class Result:
        value = 0.75

    assert normalize_score(Result()) == 0.75


@pytest.mark.asyncio
async def test_service_rejects_missing_api_key() -> None:
    service = RagasEvaluationService(Settings(openai_api_key=""))
    request = EvaluationRequest(
        question="Pregunta",
        answer="Respuesta",
        contexts=["Contexto"],
    )
    with pytest.raises(MissingConfigurationError):
        await service.evaluate(request)


class SimilarTextEmbeddings(BaseRagasEmbedding):
    """Deterministic test embeddings for two nearly identical answers."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        del kwargs
        return [1.0, 0.01] if text == REFERENCE else [1.0, 0.02]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text, **kwargs)


@pytest.mark.asyncio
async def test_semantic_similarity_is_high_and_finite_for_real_case() -> None:
    result = await SemanticSimilarity(embeddings=SimilarTextEmbeddings()).ascore(
        reference=REFERENCE,
        response=ANSWER,
    )
    assert math.isfinite(result.value)
    assert result.value > 0.99


@pytest.mark.asyncio
async def test_real_case_fields_are_mapped_to_ragas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict] = {}

    class FakeClient:
        async def close(self) -> None:
            return None

    def metric(metric_name: str, score: float):
        class FakeMetric:
            def __init__(self, **kwargs) -> None:
                del kwargs

            async def ascore(self, **kwargs):
                calls[metric_name] = kwargs
                return SimpleNamespace(value=score)

        return FakeMetric

    monkeypatch.setattr(ragas_service, "AsyncOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ragas_service, "llm_factory", lambda *args, **kwargs: object())
    monkeypatch.setattr(ragas_service, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(ragas_service, "Faithfulness", metric("faithfulness", 1.0))
    monkeypatch.setattr(
        ragas_service,
        "AnswerRelevancy",
        metric("answer_relevancy", 0.23665670011992002),
    )
    monkeypatch.setattr(
        ragas_service,
        "SemanticSimilarity",
        metric("answer_similarity", 0.9999),
    )
    monkeypatch.setattr(ragas_service, "ContextPrecision", metric("context_precision", 1.0))
    monkeypatch.setattr(ragas_service, "ContextRecall", metric("context_recall", 1.0))

    request = EvaluationRequest(
        question=QUESTION,
        answer=ANSWER,
        contexts=CONTEXTS,
        reference=REFERENCE,
    )
    result = await RagasEvaluationService(
        Settings(openai_api_key="test-key")
    )._evaluate_metrics(request)

    assert calls["faithfulness"] == {
        "user_input": QUESTION,
        "response": ANSWER,
        "retrieved_contexts": CONTEXTS,
    }
    assert calls["answer_relevancy"] == {
        "user_input": QUESTION,
        "response": ANSWER,
    }
    assert calls["answer_similarity"] == {
        "reference": REFERENCE,
        "response": ANSWER,
    }
    assert calls["context_precision"] == {
        "user_input": QUESTION,
        "reference": REFERENCE,
        "retrieved_contexts": CONTEXTS,
    }
    assert calls["context_recall"] == calls["context_precision"]
    assert result.metrics.answer_relevancy == 0.23665670011992002
    assert result.metrics.answer_similarity == 0.9999
    assert all(
        value is None or math.isfinite(value)
        for value in result.metrics.model_dump().values()
    )
