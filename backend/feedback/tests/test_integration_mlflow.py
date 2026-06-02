"""
Tests de integracion contra MLflow real (requiere `docker-compose up`).

Verifica el round-trip completo:
1. Genera un trace real invocando el grafo (o usa una traza minimal sintetica).
2. Llama a `POST /feedback` y `POST /feedback/expectation`.
3. Recupera el trace via `MlflowClient.get_trace` y comprueba los assessments.

Se saltan automaticamente si MLflow no es alcanzable, asi no rompen CI cuando
no hay infraestructura levantada.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.feedback.router import router

pytestmark = pytest.mark.integration

def _mlflow_reachable() -> bool:
    """Devuelve True solo si MLFLOW_TRACKING_URI esta seteado y responde."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.MlflowClient().search_experiments(max_results=1)
        return True
    except Exception:
        return False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _mlflow_reachable(),
        reason="MLFLOW_TRACKING_URI no configurado o servidor no alcanzable.",
    ),
]

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/feedback")
    return TestClient(app)

def _create_minimal_trace(experiment_name: str) -> str:
    """Crea un trace MLflow minimo y devuelve su request_id."""
    import mlflow
    from backend.mlflow_tracer import end_trace, start_trace

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    ctx = start_trace(
        experiment_name=experiment_name,
        name=f"integration_{uuid.uuid4().hex[:8]}",
        inputs={"query": "integration test"},
        attributes={"tab": experiment_name},
    )
    assert ctx is not None, "No se pudo iniciar el trace"
    end_trace(ctx, outputs={"response": "ok"})
    # El backend es asincrono; esperamos un momento para que el trace persista.
    time.sleep(0.3)
    return ctx.request_id

def _get_trace(trace_id: str):
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.MlflowClient().get_trace(trace_id)

def test_feedback_round_trip(client):
    trace_id = _create_minimal_trace("chatbot")

    resp = client.post(
        "/feedback",
        json={
            "trace_id": trace_id,
            "value": True,
            "source_id": "integration_test",
            "rationale": "ok desde test",
        },
    )
    assert resp.status_code == 200, resp.text

    trace = _get_trace(trace_id)
    found = [a for a in (trace.info.assessments or []) if a.name == "user_thumbs"]
    assert len(found) == 1
    assert found[0].value is True
    assert found[0].rationale == "ok desde test"
    assert found[0].source.source_id == "integration_test"
    assert found[0].source.source_type == "HUMAN"

def test_expectation_round_trip(client):
    trace_id = _create_minimal_trace("chatbot")

    resp = client.post(
        "/feedback/expectation",
        json={
            "trace_id": trace_id,
            "expected_response": "respuesta esperada en test",
            "source_id": "integration_test",
        },
    )
    assert resp.status_code == 200, resp.text

    trace = _get_trace(trace_id)
    found = [a for a in (trace.info.assessments or []) if a.name == "expected_response"]
    assert len(found) == 1
    assert found[0].value == "respuesta esperada en test"
    assert found[0].source.source_id == "integration_test"
