"""Router FastAPI para los endpoints de feedback humano sobre traces MLflow."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.feedback.expectations import (
    EXPECTED_FACTS,
    EXPECTED_RESPONSE,
    log_expected_facts,
    log_expected_response,
)
from backend.feedback.schemas import (
    ExpectationRequest,
    ExpectationResponse,
    FeedbackRequest,
    FeedbackResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

def _is_enabled() -> bool:
    return settings.FEEDBACK_ENABLED

@router.post("", response_model=FeedbackResponse)
async def post_feedback(payload: FeedbackRequest):
    """Registra un assessment de tipo feedback humano en el trace indicado.

    - 503 si FEEDBACK_ENABLED esta desactivado.
    - 200 si el feedback se registra correctamente.
    - 202 con `status="queued"` si MLflow no esta disponible (sin retries).
    """
    if not _is_enabled():
        raise HTTPException(status_code=503, detail="Feedback deshabilitado por configuracion.")

    if not settings.MLFLOW_TRACKING_URI:
        logger.warning("Feedback recibido pero MLFLOW_TRACKING_URI no esta configurado.")
        return JSONResponse(
            status_code=202,
            content={"status": "queued", "detail": "MLFLOW_TRACKING_URI no configurado."},
        )

    try:
        import mlflow
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=payload.source_id)

        mlflow.log_feedback(
            trace_id=payload.trace_id,
            name=payload.name,
            value=payload.value,
            source=source,
            rationale=payload.rationale,
            metadata=payload.metadata or None,
        )
        return FeedbackResponse(status="ok")
    except Exception as exc:
        # MLflow caido / trace inexistente / error de red: no fallamos al cliente.
        logger.warning("No se pudo registrar feedback en MLflow: %s", exc)
        return JSONResponse(
            status_code=202,
            content={"status": "queued", "detail": str(exc)[:200]},
        )

@router.post("/expectation", response_model=ExpectationResponse)
async def post_expectation(payload: ExpectationRequest):
    """Registra `expected_response` y/o `expected_facts` como ground truth.

    Envuelve los helpers de `backend.feedback.expectations` para exponerlos al
    frontend (panel de anotacion en la tab Tracking). Permite anotar ambos
    campos en una unica llamada.
    """
    if not _is_enabled():
        raise HTTPException(status_code=503, detail="Feedback deshabilitado por configuracion.")

    if payload.expected_response is None and not payload.expected_facts:
        raise HTTPException(
            status_code=422,
            detail="Debes proporcionar expected_response y/o expected_facts.",
        )

    if not settings.MLFLOW_TRACKING_URI:
        logger.warning("Expectation recibida pero MLFLOW_TRACKING_URI no esta configurado.")
        return JSONResponse(
            status_code=202,
            content={"status": "queued", "registered": [], "detail": "MLFLOW_TRACKING_URI no configurado."},
        )

    registered: list[str] = []
    try:
        if payload.expected_response is not None:
            log_expected_response(
                trace_id=payload.trace_id,
                response=payload.expected_response,
                source_id=payload.source_id,
            )
            registered.append(EXPECTED_RESPONSE)

        if payload.expected_facts:
            log_expected_facts(
                trace_id=payload.trace_id,
                facts=payload.expected_facts,
                source_id=payload.source_id,
            )
            registered.append(EXPECTED_FACTS)

        return ExpectationResponse(status="ok", registered=registered)
    except Exception as exc:
        logger.warning("No se pudo registrar expectation en MLflow: %s", exc)
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "registered": registered,
                "detail": str(exc)[:200],
            },
        )
