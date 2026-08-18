import logging
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.schemas import (
    ConfigResponse,
    ErrorResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)
from app.services.ragas_service import (
    EvaluationTimeoutError,
    MissingConfigurationError,
    RagasEvaluationService,
    RagasServiceError,
    UpstreamServiceError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_ragas_service(settings: Settings = Depends(get_settings)) -> RagasEvaluationService:
    return RagasEvaluationService(settings)


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/config", response_model=ConfigResponse, tags=["system"])
async def config(settings: Settings = Depends(get_settings)) -> ConfigResponse:
    return ConfigResponse(
        openai_configured=settings.openai_configured,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )


@router.post(
    "/ragas/eval",
    response_model=EvaluationResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["evaluation"],
)
async def evaluate_rag(
    payload: EvaluationRequest,
    service: RagasEvaluationService = Depends(get_ragas_service),
) -> EvaluationResponse:
    started = perf_counter()
    metric_names = ["faithfulness", "answer_relevancy"]
    if payload.reference is not None:
        metric_names.extend(["context_precision", "context_recall"])
    logger.info(
        "Inicio de evaluacion contexts=%d reference=%s metrics=%s",
        len(payload.contexts),
        payload.reference is not None,
        metric_names,
    )

    try:
        result = await service.evaluate(payload)
    except MissingConfigurationError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "configuration_error", str(exc)) from exc
    except EvaluationTimeoutError as exc:
        raise _http_error(status.HTTP_504_GATEWAY_TIMEOUT, "evaluation_timeout", str(exc)) from exc
    except UpstreamServiceError as exc:
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "openai_error", str(exc)) from exc
    except RagasServiceError as exc:
        raise _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "evaluation_failed", str(exc)) from exc
    except Exception as exc:
        logger.exception("Error interno no controlado")
        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "Error interno durante la evaluacion",
        ) from exc

    elapsed_ms = round((perf_counter() - started) * 1000)
    logger.info("Evaluacion completada duration_ms=%d metrics=%s", elapsed_ms, result.metrics.model_dump())
    return EvaluationResponse(
        question=payload.question,
        answer=payload.answer,
        context_items=len(payload.contexts),
        reference_provided=payload.reference is not None,
        metrics=result.metrics,
        evaluation_time_ms=elapsed_ms,
    )


def _http_error(status_code: int, error: str, detail: str) -> HTTPException:
    logger.error("Evaluacion fallida error=%s detail=%s", error, detail)
    return HTTPException(
        status_code=status_code,
        detail={"status": "error", "error": error, "detail": detail},
    )
