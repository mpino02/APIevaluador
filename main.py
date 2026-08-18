from app.api.routes import router
from app.config import configure_logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

configure_logging()

app = FastAPI(
    title="RAGAS Evaluation API",
    description="Evaluacion de respuestas RAG con metricas reales de RAGAS.",
    version="1.0.0",
)
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep application errors in the documented flat JSON format."""
    del request
    if isinstance(exc.detail, dict) and exc.detail.get("status") == "error":
        content = exc.detail
    else:
        content = {"detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)
