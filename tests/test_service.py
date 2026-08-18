import math

import pytest

from app.config import Settings
from app.schemas import EvaluationRequest
from app.services.ragas_service import MissingConfigurationError, RagasEvaluationService, normalize_score


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
