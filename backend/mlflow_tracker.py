"""
Módulo mlflow_tracker — Registro de inferencias LLM en MLflow.

Registra cada inferencia como un run de MLflow con parámetros, métricas
y artefactos. Si MLflow no está disponible, falla silenciosamente.

Uso:
    from backend.mlflow_tracker import log_to_mlflow
    log_to_mlflow(entry, context_full="...")
"""

import json
import logging
import os
import tempfile
from typing import Optional

from backend.config import settings
from backend.inference_logger.models import LogEntry

logger = logging.getLogger(__name__)

_mlflow_available: Optional[bool] = None

def _check_mlflow() -> bool:
    """Comprueba si MLflow está disponible y configurado. Cachea el resultado."""
    global _mlflow_available
    if _mlflow_available is not None:
        return _mlflow_available

    if not settings.MLFLOW_TRACKING_URI:
        logger.warning(
            "MLFLOW_TRACKING_URI no definido — tracking de MLflow desactivado"
        )
        _mlflow_available = False
        return False

    try:
        import mlflow  # noqa: F401
        _mlflow_available = True
        return True
    except ImportError:
        logger.warning("mlflow no instalado — tracking desactivado")
        _mlflow_available = False
        return False

def log_to_mlflow(entry: LogEntry, context_full: Optional[str] = None) -> None:
    """Registra un LogEntry como run de MLflow.

    Args:
        entry: LogEntry con los datos de la inferencia.
        context_full: Contexto RAG completo (sin truncar) para el artefacto.
    """
    if not _check_mlflow():
        return

    try:
        import mlflow

        mlflow.set_experiment(entry.tab)

        with mlflow.start_run():
            # Parámetros
            params = {
                "model": entry.model or "unknown",
                "tab": entry.tab,
                "session_id": entry.session_id,
            }
            if entry.municipio is not None:
                params["municipio"] = entry.municipio
            if entry.intent is not None:
                params["intent"] = entry.intent
            mlflow.log_params(params)

            # Métricas
            metrics = {"time_total_ms": entry.time_total_ms}
            if entry.time_llm_ms:
                metrics["time_llm_ms"] = entry.time_llm_ms
            if entry.time_retrieval_ms is not None:
                metrics["time_retrieval_ms"] = entry.time_retrieval_ms
            if entry.time_reranking_ms is not None:
                metrics["time_reranking_ms"] = entry.time_reranking_ms
            if entry.tokens_input is not None:
                metrics["tokens_input"] = entry.tokens_input
            if entry.tokens_output:
                metrics["tokens_output"] = entry.tokens_output
            mlflow.log_metrics(metrics)

            # Artefacto JSON
            artifact_data = {
                "query": entry.query,
                "response": entry.response,
                "context": context_full or entry.context_preview or "",
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="mlflow_artifact_"
            ) as f:
                json.dump(artifact_data, f, ensure_ascii=False, indent=2)
                tmp_path = f.name

            try:
                mlflow.log_artifact(tmp_path, artifact_path="inference")
            finally:
                os.unlink(tmp_path)

        logger.debug(
            "MLflow run registrado: experiment=%s, model=%s",
            entry.tab,
            entry.model,
        )

    except Exception as exc:
        logger.error(
            "Error registrando run en MLflow (experiment=%s, model=%s): %s",
            entry.tab,
            entry.model,
            exc,
            exc_info=True,
        )
