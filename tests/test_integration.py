import os

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


integration_enabled = (
    os.getenv("RUN_INTEGRATION_TESTS", "").lower() == "true"
    and bool(os.getenv("OPENAI_API_KEY", "").strip())
)


@pytest.mark.asyncio
@pytest.mark.skipif(not integration_enabled, reason="Requiere RUN_INTEGRATION_TESTS=true y OPENAI_API_KEY")
async def test_real_ragas_evaluation() -> None:
    payload = {
        "question": "¿Cual es la capital de Chile?",
        "answer": "La capital de Chile es Santiago.",
        "contexts": ["Santiago es la capital y principal ciudad de Chile."],
        "reference": "Santiago",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        timeout=180,
    ) as client:
        response = await client.post("/ragas/eval", json=payload)
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    assert all(metrics[name] is not None for name in metrics)
