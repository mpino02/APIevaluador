import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Awaitable

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.config import Settings
from app.schemas import EvaluationRequest, MetricScores

logger = logging.getLogger(__name__)


class RagasServiceError(Exception):
    """Base error for evaluation failures."""


class MissingConfigurationError(RagasServiceError):
    """Raised when a required secret is not configured."""


class EvaluationTimeoutError(RagasServiceError):
    """Raised when an evaluation exceeds its configured deadline."""


class UpstreamServiceError(RagasServiceError):
    """Raised when OpenAI cannot complete a request."""


@dataclass(frozen=True)
class EvaluationResult:
    metrics: MetricScores


def normalize_score(value: Any) -> float | None:
    """Convert a metric value to a finite JSON-safe float."""
    if value is None:
        return None
    raw_value = getattr(value, "value", value)
    try:
        score = float(raw_value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


class RagasEvaluationService:
    """Run RAGAS 0.4 collection metrics using native OpenAI clients."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if not self.settings.openai_configured:
            raise MissingConfigurationError("OPENAI_API_KEY no esta configurada")

        try:
            return await asyncio.wait_for(
                self._evaluate_metrics(request),
                timeout=self.settings.ragas_timeout,
            )
        except TimeoutError as exc:
            raise EvaluationTimeoutError(
                f"La evaluacion excedio el timeout de {self.settings.ragas_timeout:g} segundos"
            ) from exc
        except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as exc:
            logger.exception("OpenAI fallo durante la evaluacion")
            raise UpstreamServiceError(f"OpenAI no pudo completar la evaluacion: {exc}") from exc
        except RagasServiceError:
            raise
        except Exception as exc:
            if _contains_openai_error(exc):
                logger.exception("OpenAI fallo dentro de la evaluacion RAGAS")
                raise UpstreamServiceError(
                    f"OpenAI no pudo completar la evaluacion: {exc}"
                ) from exc
            logger.exception("RAGAS fallo durante la evaluacion")
            raise RagasServiceError(f"RAGAS no pudo completar la evaluacion: {exc}") from exc

    async def _evaluate_metrics(self, request: EvaluationRequest) -> EvaluationResult:
        client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.ragas_timeout,
            max_retries=2,
        )
        try:
            llm = llm_factory(self.settings.openai_model, client=client)
            embeddings = OpenAIEmbeddings(
                client=client,
                model=self.settings.openai_embedding_model,
            )

            tasks: dict[str, Awaitable[Any]] = {
                "faithfulness": Faithfulness(llm=llm).ascore(
                    user_input=request.question,
                    response=request.answer,
                    retrieved_contexts=request.contexts,
                ),
                "answer_relevancy": AnswerRelevancy(
                    llm=llm, embeddings=embeddings
                ).ascore(
                    user_input=request.question,
                    response=request.answer,
                ),
            }
            if request.reference is not None:
                tasks.update(
                    {
                        "context_precision": ContextPrecision(llm=llm).ascore(
                            user_input=request.question,
                            reference=request.reference,
                            retrieved_contexts=request.contexts,
                        ),
                        "context_recall": ContextRecall(llm=llm).ascore(
                            user_input=request.question,
                            reference=request.reference,
                            retrieved_contexts=request.contexts,
                        ),
                    }
                )

            names = list(tasks)
            values = await asyncio.gather(*(tasks[name] for name in names))
            scores = {name: normalize_score(value) for name, value in zip(names, values)}
            return EvaluationResult(
                metrics=MetricScores(
                    faithfulness=scores.get("faithfulness"),
                    answer_relevancy=scores.get("answer_relevancy"),
                    context_precision=scores.get("context_precision"),
                    context_recall=scores.get("context_recall"),
                )
            )
        finally:
            await client.close()


def _contains_openai_error(exc: BaseException) -> bool:
    """Detect SDK errors even when a metric or retry library wraps them."""
    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, OpenAIError):
            return True
        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
        nested = getattr(current, "exceptions", ())
        pending.extend(item for item in nested if isinstance(item, BaseException))
    return False
