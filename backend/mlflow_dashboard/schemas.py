"""Modelos Pydantic de respuesta para el módulo mlflow_dashboard."""

from typing import List, Optional

from pydantic import BaseModel

class ExperimentSummary(BaseModel):
    """Resumen de un experimento MLflow con métricas agregadas."""

    experiment_name: str
    run_count: int
    avg_time_total_ms: Optional[float] = None
    avg_time_llm_ms: Optional[float] = None
    avg_time_retrieval_ms: Optional[float] = None
    avg_time_reranking_ms: Optional[float] = None
    avg_tokens_input: Optional[float] = None
    avg_tokens_output: Optional[float] = None

class RunDetail(BaseModel):
    """Detalle de un run individual de MLflow."""

    run_id: str
    timestamp: int
    model: Optional[str] = None
    tab: Optional[str] = None
    municipio: Optional[str] = None
    time_total_ms: Optional[float] = None
    time_llm_ms: Optional[float] = None
    time_retrieval_ms: Optional[float] = None
    time_reranking_ms: Optional[float] = None
    tokens_input: Optional[float] = None
    tokens_output: Optional[float] = None

class SpanDetail(BaseModel):
    """Detalle de un span dentro de un trace."""

    name: str
    status: str = ""
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    duration_ms: Optional[float] = None
    inputs: Optional[dict] = None
    outputs: Optional[dict] = None
    attributes: Optional[dict] = None

class TraceDetail(BaseModel):
    """Detalle de un trace MLflow con sus spans."""

    trace_id: str
    timestamp: int = 0
    duration_ms: Optional[float] = None
    status: str = ""
    experiment_name: Optional[str] = None
    tags: Optional[dict] = None
    spans: List[SpanDetail] = []
    request_preview: Optional[str] = None

class RunFullDetail(BaseModel):
    """Detalle completo de un run con params, metrics y artifacts."""

    run_id: str
    timestamp: int = 0
    status: str = ""
    params: dict = {}
    metrics: dict = {}
    tags: dict = {}
    artifact_query: Optional[str] = None
    artifact_response: Optional[str] = None
    artifact_context: Optional[str] = None

class DashboardResponse(BaseModel):
    """Respuesta del endpoint de resumen del dashboard."""

    experiments: List[ExperimentSummary]
    mlflow_url: str
