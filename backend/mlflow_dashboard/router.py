"""
Módulo mlflow_dashboard — Router FastAPI para consulta de métricas MLflow.

Endpoints:
  GET /mlflow/experiments — Lista de experimentos MLflow.
  GET /mlflow/summary    — Métricas agregadas por experimento.
  GET /mlflow/runs       — Runs recientes con detalle individual.
"""

import logging
from typing import List, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.mlflow_dashboard.schemas import (
    DashboardResponse,
    ExperimentSummary,
    RunDetail,
    RunFullDetail,
    SpanDetail,
    TraceDetail,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

_MLFLOW_URI = settings.MLFLOW_TRACKING_URI or "http://localhost:5001"

_METRIC_KEYS = [
    "time_total_ms",
    "time_llm_ms",
    "time_retrieval_ms",
    "time_reranking_ms",
    "tokens_input",
    "tokens_output",
]

async def _mlflow_get(path: str, **kwargs) -> dict:
    """Hace un GET a la API REST de MLflow y devuelve el JSON."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{_MLFLOW_URI}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()

async def _mlflow_post(path: str, payload: dict) -> dict:
    """Hace un POST a la API REST de MLflow y devuelve el JSON."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{_MLFLOW_URI}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

def _build_run_filter(
    experiment_ids: List[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Construye el filter_string para la API de búsqueda de runs.

    MLflow filtra por timestamp en milisegundos. Convertimos las fechas
    YYYY-MM-DD a epoch-ms para evitar problemas de formato y off-by-one.
    """
    from datetime import datetime, timedelta, timezone

    parts: list[str] = []
    if date_from:
        try:
            dt = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            parts.append(f"attributes.start_time >= {int(dt.timestamp() * 1000)}")
        except ValueError:
            pass
    if date_to:
        try:
            # Incluir todo el dia date_to: avanzar al inicio del dia siguiente
            dt = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            parts.append(f"attributes.start_time < {int(dt.timestamp() * 1000)}")
        except ValueError:
            pass
    return " AND ".join(parts)

def _extract_metric(run: dict, key: str) -> Optional[float]:
    """Extrae una métrica de un run de MLflow."""
    metrics = run.get("data", {}).get("metrics", [])
    for m in metrics:
        if m["key"] == key:
            return m["value"]
    return None

def _extract_param(run: dict, key: str) -> Optional[str]:
    """Extrae un parámetro de un run de MLflow."""
    params = run.get("data", {}).get("params", [])
    for p in params:
        if p["key"] == key:
            return p["value"]
    return None

def _aggregate_metrics(runs: list[dict]) -> dict:
    """Calcula medias de las métricas sobre una lista de runs."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for run in runs:
        for key in _METRIC_KEYS:
            val = _extract_metric(run, key)
            if val is not None:
                sums[key] = sums.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1
    return {
        f"avg_{key}": round(sums[key] / counts[key], 2) if key in counts else None
        for key in _METRIC_KEYS
    }

async def _get_experiment_ids(name: Optional[str] = None) -> list[dict]:
    """Devuelve experimentos de MLflow. Si name, filtra por nombre exacto."""
    payload: dict = {"max_results": 200}
    if name:
        payload["filter"] = f"name = '{name}'"
    data = await _mlflow_post("/api/2.0/mlflow/experiments/search", payload)
    return data.get("experiments", [])

async def _search_runs(
    experiment_ids: List[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_results: int = 1000,
) -> list[dict]:
    """Busca runs en MLflow con filtros opcionales."""
    payload: dict = {
        "experiment_ids": experiment_ids,
        "max_results": max_results,
        "order_by": ["attributes.start_time DESC"],
    }
    filter_str = _build_run_filter(experiment_ids, date_from, date_to)
    if filter_str:
        payload["filter"] = filter_str
    data = await _mlflow_post("/api/2.0/mlflow/runs/search", payload)
    return data.get("runs", [])

# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/experiments")
async def list_experiments():
    """Devuelve la lista de experimentos MLflow con su número de runs."""
    try:
        experiments = await _get_experiment_ids()
        result = []
        for exp in experiments:
            name = exp.get("name", "")
            if name == "Default":
                continue
            exp_id = exp["experiment_id"]
            all_runs = await _search_runs([exp_id])
            result.append({"experiment_name": name, "run_count": len(all_runs)})
        return result
    except Exception as exc:
        logger.warning("MLflow no disponible: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"MLflow no disponible: {exc}"},
        )

@router.get("/summary", response_model=DashboardResponse)
async def get_summary(
    experiment_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """Devuelve métricas agregadas por experimento."""
    try:
        experiments = await _get_experiment_ids(experiment_name)
        summaries: list[ExperimentSummary] = []
        for exp in experiments:
            name = exp.get("name", "")
            if name == "Default":
                continue
            exp_id = exp["experiment_id"]
            runs = await _search_runs([exp_id], date_from, date_to)
            agg = _aggregate_metrics(runs)
            summaries.append(
                ExperimentSummary(
                    experiment_name=name,
                    run_count=len(runs),
                    **agg,
                )
            )

        mlflow_url = settings.MLFLOW_EXTERNAL_URL or _MLFLOW_URI
        return DashboardResponse(experiments=summaries, mlflow_url=mlflow_url)
    except Exception as exc:
        logger.warning("MLflow no disponible: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"MLflow no disponible: {exc}"},
        )

@router.get("/runs", response_model=List[RunDetail])
async def get_runs(
    experiment_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
):
    """Devuelve runs individuales ordenados por timestamp descendente."""
    try:
        experiments = await _get_experiment_ids(experiment_name)
        exp_ids = [
            e["experiment_id"]
            for e in experiments
            if e.get("name") != "Default"
        ]
        if not exp_ids:
            return []

        runs = await _search_runs(exp_ids, date_from, date_to, max_results=limit)
        runs.sort(
            key=lambda r: r.get("info", {}).get("start_time", 0), reverse=True
        )
        result: list[RunDetail] = []
        for run in runs:
            info = run.get("info", {})
            result.append(
                RunDetail(
                    run_id=info.get("run_id", ""),
                    timestamp=info.get("start_time", 0),
                    model=_extract_param(run, "model"),
                    tab=_extract_param(run, "tab"),
                    municipio=_extract_param(run, "municipio"),
                    time_total_ms=_extract_metric(run, "time_total_ms"),
                    time_llm_ms=_extract_metric(run, "time_llm_ms"),
                    time_retrieval_ms=_extract_metric(run, "time_retrieval_ms"),
                    time_reranking_ms=_extract_metric(run, "time_reranking_ms"),
                    tokens_input=_extract_metric(run, "tokens_input"),
                    tokens_output=_extract_metric(run, "tokens_output"),
                )
            )
        return result
    except Exception as exc:
        logger.warning("MLflow no disponible (runs): %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"MLflow no disponible: {exc}"},
        )

# ── Endpoints de Traces ─────────────────────────────────────────────

def _get_mlflow_client():
    """Devuelve una instancia de MlflowClient configurada."""
    import mlflow
    mlflow.set_tracking_uri(_MLFLOW_URI)
    return mlflow.MlflowClient()

@router.get("/traces", response_model=List[TraceDetail])
async def get_traces(
    experiment_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
):
    """Devuelve traces recientes con sus spans."""
    try:
        client = _get_mlflow_client()

        experiments = await _get_experiment_ids(experiment_name)
        exp_ids = [
            e["experiment_id"]
            for e in experiments
            if e.get("name") != "Default"
        ]
        if not exp_ids:
            return []

        from datetime import datetime as dt_cls, timedelta, timezone
        ts_from = None
        ts_to = None
        if date_from:
            try:
                d = dt_cls.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ts_from = int(d.timestamp() * 1000)
            except ValueError:
                pass
        if date_to:
            try:
                d = dt_cls.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
                ts_to = int(d.timestamp() * 1000)
            except ValueError:
                pass

        exp_names = {e["experiment_id"]: e.get("name", "") for e in experiments if e.get("name") != "Default"}
        all_traces: list[TraceDetail] = []

        for exp_id in exp_ids:
            try:
                traces = client.search_traces(
                    experiment_ids=[exp_id],
                    max_results=limit,
                )
            except Exception:
                continue

            for t in traces:
                info = t.info
                if ts_from and info.timestamp_ms < ts_from:
                    continue
                if ts_to and info.timestamp_ms >= ts_to:
                    continue

                spans = []
                if t.data and t.data.spans:
                    for s in t.data.spans:
                        duration = None
                        s_start = getattr(s, "start_time_ns", None)
                        s_end = getattr(s, "end_time_ns", None)
                        if s_start and s_end:
                            duration = round((s_end - s_start) / 1_000_000, 1)

                        inputs_dict = None
                        if hasattr(s, "inputs") and s.inputs:
                            inputs_dict = s.inputs if isinstance(s.inputs, dict) else {}

                        outputs_dict = None
                        if hasattr(s, "outputs") and s.outputs:
                            outputs_dict = s.outputs if isinstance(s.outputs, dict) else {}

                        attrs_dict = None
                        if hasattr(s, "attributes") and s.attributes:
                            attrs_dict = s.attributes if isinstance(s.attributes, dict) else {}

                        spans.append(SpanDetail(
                            name=s.name,
                            status=str(s.status.status_code.name) if hasattr(s.status, "status_code") else str(s.status),
                            start_time=int(s_start / 1_000_000) if s_start else None,
                            end_time=int(s_end / 1_000_000) if s_end else None,
                            duration_ms=duration,
                            inputs=inputs_dict,
                            outputs=outputs_dict,
                            attributes=attrs_dict,
                        ))

                request_preview = None
                if t.data and t.data.spans:
                    root_span = t.data.spans[0]
                    if hasattr(root_span, "inputs") and root_span.inputs:
                        q = root_span.inputs.get("query", "")
                        if q:
                            request_preview = str(q)[:100]

                tags = info.tags or {}
                all_traces.append(TraceDetail(
                    trace_id=info.request_id,
                    timestamp=info.timestamp_ms,
                    duration_ms=info.execution_time_ms,
                    status=str(info.status.name) if hasattr(info.status, "name") else str(info.status),
                    experiment_name=exp_names.get(str(info.experiment_id), ""),
                    tags={k: v for k, v in tags.items() if not k.startswith("mlflow.")},
                    spans=spans,
                    request_preview=request_preview,
                ))

        all_traces.sort(key=lambda t: t.timestamp, reverse=True)
        return all_traces[:limit]

    except Exception as exc:
        logger.warning("Error obteniendo traces: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"detail": f"MLflow traces no disponible: {exc}"},
        )

@router.get("/runs/{run_id}", response_model=RunFullDetail)
async def get_run_detail(run_id: str):
    """Devuelve el detalle completo de un run."""
    try:
        data = await _mlflow_get("/api/2.0/mlflow/runs/get", params={"run_id": run_id})
        run = data.get("run", {})
        info = run.get("info", {})

        params = {p["key"]: p["value"] for p in run.get("data", {}).get("params", [])}
        metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
        tags = {
            t["key"]: t["value"]
            for t in run.get("data", {}).get("tags", [])
            if not t["key"].startswith("mlflow.")
        }

        artifact_query = None
        artifact_response = None
        artifact_context = None
        try:
            artifacts = await _mlflow_get(
                "/api/2.0/mlflow/artifacts/list",
                params={"run_id": run_id, "path": "inference"},
            )
            for f in artifacts.get("files", []):
                if f.get("path", "").endswith(".json"):
                    art_data = await _mlflow_get(
                        "/get-artifact",
                        params={"run_id": run_id, "path": f["path"]},
                    )
                    if isinstance(art_data, dict):
                        artifact_query = art_data.get("query")
                        artifact_response = art_data.get("response")
                        artifact_context = art_data.get("context")
                    break
        except Exception:
            pass

        return RunFullDetail(
            run_id=info.get("run_id", run_id),
            timestamp=info.get("start_time", 0),
            status=info.get("status", ""),
            params=params,
            metrics=metrics,
            tags=tags,
            artifact_query=artifact_query,
            artifact_response=artifact_response,
            artifact_context=artifact_context,
        )

    except Exception as exc:
        logger.warning("Error obteniendo run detail: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"Error: {exc}"},
        )
