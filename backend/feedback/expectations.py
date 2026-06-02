"""
Helpers programaticos para registrar `expectation` (ground truth) en traces MLflow.

Las expectations son la pieza que MLflow GenAI usa como referencia para los
scorers integrados (`Correctness`, `RelevanceToQuery`, etc.). Se anotan con
nombres estandarizados:

- `expected_response`: respuesta de referencia esperada para la query del trace.
- `expected_facts`: lista de hechos que la respuesta deberia mencionar.

Uso desde Python REPL para anotar manualmente un trace ya existente:

    >>> from backend.feedback.expectations import log_expected_response
    >>> log_expected_response(
    ...     trace_id="tr-9268870fe3cc064d365f044ef9a5f07c",
    ...     response="El plazo es de 10 dias habiles.",
    ...     source_id="anotador_jane",
    ... )

Para anotar desde un LLM-as-a-judge (fase posterior):

    >>> from backend.feedback.expectations import log_expected_facts, ExpectationSource
    >>> log_expected_facts(
    ...     trace_id="tr-...",
    ...     facts=["Es necesario cita previa", "El tramite es online"],
    ...     source_id="gpt4-judge",
    ...     source_type=ExpectationSource.LLM_JUDGE,
    ... )
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

EXPECTED_RESPONSE = "expected_response"
EXPECTED_FACTS = "expected_facts"

class ExpectationSource(str, Enum):
    """Tipos de fuente soportados al registrar una expectation."""

    HUMAN = "HUMAN"
    LLM_JUDGE = "LLM_JUDGE"

def _build_source(source_type: ExpectationSource, source_id: str):
    """Construye un `AssessmentSource` con el `source_type` mapeado a MLflow."""
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    mapping = {
        ExpectationSource.HUMAN: AssessmentSourceType.HUMAN,
        ExpectationSource.LLM_JUDGE: AssessmentSourceType.LLM_JUDGE,
    }
    return AssessmentSource(source_type=mapping[source_type], source_id=source_id)

def _log_expectation(
    *,
    trace_id: str,
    name: str,
    value: Any,
    source_id: str,
    source_type: ExpectationSource,
    metadata: Optional[dict[str, Any]] = None,
):
    """Wrapper interno que invoca `mlflow.log_expectation` con la fuente construida."""
    if not settings.MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI no esta configurado.")

    import mlflow

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    source = _build_source(source_type, source_id)
    return mlflow.log_expectation(
        trace_id=trace_id,
        name=name,
        value=value,
        source=source,
        metadata=metadata or None,
    )

def log_expected_response(
    trace_id: str,
    response: str,
    source_id: str,
    *,
    source_type: ExpectationSource = ExpectationSource.HUMAN,
    metadata: Optional[dict[str, Any]] = None,
):
    """Registra la respuesta esperada para un trace.

    Usa el nombre estandar `expected_response` reconocido por los scorers
    integrados de MLflow GenAI (p.ej. `Correctness`).

    Args:
        trace_id: request_id del trace MLflow al que se asocia.
        response: texto de la respuesta esperada (ground truth).
        source_id: identificador de la fuente (anotador o juez LLM).
        source_type: HUMAN por defecto. LLM_JUDGE para fases posteriores.
        metadata: campos adicionales arbitrarios.

    Returns:
        El `Assessment` creado por MLflow.
    """
    return _log_expectation(
        trace_id=trace_id,
        name=EXPECTED_RESPONSE,
        value=response,
        source_id=source_id,
        source_type=source_type,
        metadata=metadata,
    )

def log_expected_facts(
    trace_id: str,
    facts: list[str],
    source_id: str,
    *,
    source_type: ExpectationSource = ExpectationSource.HUMAN,
    metadata: Optional[dict[str, Any]] = None,
):
    """Registra una lista de hechos esperados para un trace.

    Usa el nombre estandar `expected_facts` reconocido por los scorers
    integrados de MLflow GenAI.

    Args:
        trace_id: request_id del trace MLflow al que se asocia.
        facts: lista de strings con los hechos que deberia contener la respuesta.
        source_id: identificador de la fuente (anotador o juez LLM).
        source_type: HUMAN por defecto.
        metadata: campos adicionales arbitrarios.

    Returns:
        El `Assessment` creado por MLflow.
    """
    return _log_expectation(
        trace_id=trace_id,
        name=EXPECTED_FACTS,
        value=list(facts),
        source_id=source_id,
        source_type=source_type,
        metadata=metadata,
    )
