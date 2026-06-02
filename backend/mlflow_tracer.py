"""
Modulo mlflow_tracer -- Tracing de inferencias LLM con MLflow Tracing API.

Usa MlflowClient imperativo (start_trace / start_span / end_span / end_trace)
para capturar el flujo de cada consulta como un trace jerarquico con spans
visibles en la pestana Traces de MLflow UI.

Complementa a mlflow_tracker.py (runs clasicos). Ambos coexisten.
"""

import json
import logging
from typing import Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_mlflow_available: Optional[bool] = None
_client = None

# Coste por millon de tokens (EUR)
COST_PER_MILLION_TOKENS = 1.0

def _check() -> bool:
    global _mlflow_available
    if _mlflow_available is not None:
        return _mlflow_available
    if not settings.MLFLOW_TRACKING_URI:
        _mlflow_available = False
        return False
    try:
        import mlflow  # noqa: F401
        _mlflow_available = True
        return True
    except ImportError:
        _mlflow_available = False
        return False

def _get_client():
    global _client
    if _client is None:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        _client = mlflow.MlflowClient()
    return _client

def compute_cost_eur(tokens_input: int = 0, tokens_output: int = 0) -> float:
    """Calcula el coste en EUR dado el numero de tokens (1 EUR / 1M tokens)."""
    total = (tokens_input or 0) + (tokens_output or 0)
    return round(total * COST_PER_MILLION_TOKENS / 1_000_000, 6)

class TraceContext:
    """Contexto de un trace activo con referencias a los LiveSpan abiertos."""

    def __init__(self, request_id: str, root_span):
        self.request_id = request_id
        self.root_span_id = root_span.span_id
        self._root_span = root_span
        # span_id -> LiveSpan abierto, para poder cerrarlos en end_span().
        self._open_spans: dict[str, Any] = {}

def start_trace(
    experiment_name: str,
    name: str,
    inputs: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, str]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[TraceContext]:
    """Inicia un trace raiz de MLflow.

    Args:
        session_id: ID de sesion para agrupar traces en la vista Sessions de MLflow.
        user_id: alias del evaluador / usuario final (firma los traces en la vista Users).

    Returns:
        TraceContext o None si tracing no esta disponible.
    """
    if not _check():
        return None
    try:
        import mlflow
        from mlflow.tracing.fluent import start_span_no_context

        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return None

        # Las claves reservadas mlflow.trace.session / mlflow.trace.user se
        # persisten como request_metadata (no como tags). La UI Sessions/Users
        # filtra por metadata.`mlflow.trace.session` y solo start_span_no_context
        # acepta el argumento `metadata=` para inyectarlas.
        metadata: dict[str, str] = {}
        if session_id:
            metadata["mlflow.trace.session"] = session_id
        if user_id:
            metadata["mlflow.trace.user"] = user_id

        root = start_span_no_context(
            name=name,
            inputs=inputs or {},
            attributes=attributes or {},
            metadata=metadata or None,
            experiment_id=exp.experiment_id,
        )
        return TraceContext(request_id=root.trace_id, root_span=root)
    except Exception:
        logger.debug("Error iniciando trace MLflow", exc_info=True)
        return None

def start_child_span(
    ctx: Optional[TraceContext],
    name: str,
    span_type: str = "UNKNOWN",
    inputs: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Inicia un span hijo dentro del trace activo.

    Args:
        span_type: Tipo de span MLflow (LLM, RETRIEVER, RERANKER, CHAIN, TOOL, etc.)

    Returns:
        El span_id o None.
    """
    if ctx is None:
        return None
    try:
        from mlflow.tracing.fluent import start_span_no_context

        span = start_span_no_context(
            name=name,
            span_type=span_type,
            parent_span=ctx._root_span,
            inputs=inputs or {},
            attributes=attributes or {},
        )
        ctx._open_spans[span.span_id] = span
        return span.span_id
    except Exception:
        logger.debug("Error iniciando span MLflow", exc_info=True)
        return None

def end_span(
    ctx: Optional[TraceContext],
    span_id: Optional[str],
    outputs: Optional[dict[str, Any]] = None,
) -> None:
    """Cierra un span registrando sus outputs."""
    if ctx is None or span_id is None:
        return
    span = ctx._open_spans.pop(span_id, None)
    if span is None:
        return
    try:
        span.end(outputs=outputs or {})
    except Exception:
        logger.debug("Error cerrando span MLflow", exc_info=True)

def emit_trace_id_event(request_id: Optional[str], **extra: str) -> dict:
    """Construye un evento SSE con el trace_id del trace activo.

    Devuelve None-safe: si no hay trace activo, emite cadena vacia (el frontend
    puede detectarlo y ocultar el widget de feedback).

    Args:
        request_id: request_id del trace MLflow (equivalente al trace_id).
        **extra: campos adicionales para el payload (p.ej. llm="A" en el comparador).
    """
    payload: dict[str, str] = {"type": "trace_id", "trace_id": request_id or ""}
    payload.update(extra)
    return {"event": "message", "data": json.dumps(payload)}

def end_trace(
    ctx: Optional[TraceContext],
    outputs: Optional[dict[str, Any]] = None,
) -> None:
    """Cierra el trace raiz."""
    if ctx is None:
        return
    try:
        ctx._root_span.end(outputs=outputs or {})
    except Exception:
        logger.debug("Error cerrando trace MLflow", exc_info=True)

def set_trace_tags(
    ctx: Optional[TraceContext],
    tags: dict[str, str],
) -> None:
    """Adjunta tags a la traza activa via `MlflowClient.set_trace_tag`.

    Resilient: si tracing no esta activo, si MLflow no responde, o si la traza
    ya no existe, NO propaga la excepcion. La filosofia es "tags son
    observabilidad, no funcionalidad" — un fallo de tagging no debe romper
    la respuesta SSE al usuario.

    usado para adjuntar `prompt.<name>.version` y
    `prompt.<name>.source` a la traza emitida por SSE, reusando el trace_id
    que ya correlaciona feedback humano.

    Args:
        ctx: contexto de la traza activa (None = no-op).
        tags: diccionario {key: value} con los tags a adjuntar.
    """
    if ctx is None or not tags:
        return
    if not _check():
        return
    try:
        client = _get_client()
        for key, value in tags.items():
            client.set_trace_tag(ctx.request_id, key, str(value))
    except Exception:
        logger.debug("Error adjuntando tags a la traza MLflow", exc_info=True)
