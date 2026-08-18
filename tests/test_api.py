from dataclasses import dataclass

import pytest
from httpx import AsyncClient

from app.api.routes import get_ragas_service
from app.schemas import MetricScores
from app.services.ragas_service import EvaluationResult, MissingConfigurationError
from main import app

pytestmark = pytest.mark.asyncio


@dataclass
class FakeRagasService:
    fail_configuration: bool = False

    async def evaluate(self, request):
        if self.fail_configuration:
            raise MissingConfigurationError("OPENAI_API_KEY no esta configurada")
        return EvaluationResult(
            metrics=MetricScores(
                faithfulness=0.98,
                answer_relevancy=0.95,
                context_precision=1.0 if request.reference is not None else None,
                context_recall=1.0 if request.reference is not None else None,
            )
        )


def use_fake_service(*, fail_configuration: bool = False) -> None:
    app.dependency_overrides[get_ragas_service] = lambda: FakeRagasService(fail_configuration)


def valid_payload(**overrides):
    payload = {
        "question": "¿Cual es la capital de Chile?",
        "answer": "La capital de Chile es Santiago.",
        "contexts": ["Santiago es la capital y principal ciudad de Chile."],
        "reference": "Santiago",
    }
    payload.update(overrides)
    return payload


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ragas-api"}


async def test_config_never_exposes_api_key(client: AsyncClient) -> None:
    response = await client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"openai_configured", "model", "embedding_model"}
    assert "api_key" not in response.text.lower()


async def test_valid_post(client: AsyncClient) -> None:
    use_fake_service()
    response = await client.post("/ragas/eval", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["context_items"] == 1
    assert body["reference_provided"] is True
    assert body["metrics"] == {
        "faithfulness": 0.98,
        "answer_relevancy": 0.95,
        "context_precision": 1.0,
        "context_recall": 1.0,
    }


@pytest.mark.parametrize("field", ["question", "answer"])
async def test_empty_required_text_is_422(client: AsyncClient, field: str) -> None:
    response = await client.post("/ragas/eval", json=valid_payload(**{field: "  "}))
    assert response.status_code == 422


async def test_empty_contexts_is_422(client: AsyncClient) -> None:
    response = await client.post("/ragas/eval", json=valid_payload(contexts=[]))
    assert response.status_code == 422


async def test_blank_context_item_is_422(client: AsyncClient) -> None:
    response = await client.post("/ragas/eval", json=valid_payload(contexts=[" "]))
    assert response.status_code == 422


async def test_context_alias_is_supported(client: AsyncClient) -> None:
    use_fake_service()
    payload = valid_payload()
    payload["context"] = payload.pop("contexts")
    response = await client.post("/ragas/eval", json=payload)
    assert response.status_code == 200
    assert response.json()["context_items"] == 1


async def test_context_and_contexts_together_are_422(client: AsyncClient) -> None:
    response = await client.post(
        "/ragas/eval",
        json=valid_payload(context=["duplicado"]),
    )
    assert response.status_code == 422


async def test_reference_is_optional(client: AsyncClient) -> None:
    use_fake_service()
    payload = valid_payload()
    payload.pop("reference")
    response = await client.post("/ragas/eval", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["reference_provided"] is False
    assert body["metrics"]["context_precision"] is None
    assert body["metrics"]["context_recall"] is None


async def test_configuration_error(client: AsyncClient) -> None:
    use_fake_service(fail_configuration=True)
    response = await client.post("/ragas/eval", json=valid_payload())
    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "error": "configuration_error",
        "detail": "OPENAI_API_KEY no esta configurada",
    }
