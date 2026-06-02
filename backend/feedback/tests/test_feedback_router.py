"""
Tests unitarios del router /feedback.

Mockea `mlflow.log_feedback` y `mlflow.log_expectation` para verificar los
argumentos pasados, el manejo del feature flag `FEEDBACK_ENABLED`, los
codigos 422/503/202 y la cobertura de los 3 tipos de `value` admitidos.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings
from backend.feedback.router import router

@pytest.fixture(autouse=True)
def _mlflow_uri(monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "http://mlflow-mock:5000")
    monkeypatch.setattr(settings, "FEEDBACK_ENABLED", True)

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/feedback")
    return TestClient(app)

# ── /feedback ────────────────────────────────────────────────────────

def test_feedback_happy_path_bool_value(client):
    with patch("mlflow.log_feedback") as log_mock, patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback",
            json={
                "trace_id": "tr-abc",
                "value": True,
                "source_id": "jane.smith",
                "rationale": "respuesta clara",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok", "detail": None}
    kwargs = log_mock.call_args.kwargs
    assert kwargs["trace_id"] == "tr-abc"
    assert kwargs["name"] == "user_thumbs"
    assert kwargs["value"] is True
    assert kwargs["rationale"] == "respuesta clara"
    source = kwargs["source"]
    assert source.source_type == "HUMAN"
    assert source.source_id == "jane.smith"

@pytest.mark.parametrize("value", [0.85, "A"])
def test_feedback_accepts_float_and_str_values(client, value):
    with patch("mlflow.log_feedback") as log_mock, patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback",
            json={"trace_id": "tr-x", "value": value, "source_id": "u"},
        )
    assert resp.status_code == 200, resp.text
    assert log_mock.call_args.kwargs["value"] == value

def test_feedback_passes_metadata_through(client):
    with patch("mlflow.log_feedback") as log_mock, patch("mlflow.set_tracking_uri"):
        client.post(
            "/feedback",
            json={
                "trace_id": "tr-x",
                "value": "A",
                "source_id": "u",
                "name": "comparador_preference",
                "metadata": {"trace_id_a": "tr-a", "trace_id_b": "tr-b"},
            },
        )
    kwargs = log_mock.call_args.kwargs
    assert kwargs["name"] == "comparador_preference"
    assert kwargs["metadata"] == {"trace_id_a": "tr-a", "trace_id_b": "tr-b"}

def test_feedback_missing_source_id_returns_422(client):
    resp = client.post("/feedback", json={"trace_id": "tr-x", "value": True})
    assert resp.status_code == 422
    assert any(loc == "source_id" for err in resp.json()["detail"] for loc in err["loc"])

def test_feedback_empty_source_id_returns_422(client):
    resp = client.post(
        "/feedback", json={"trace_id": "tr-x", "value": True, "source_id": ""},
    )
    assert resp.status_code == 422

def test_feedback_disabled_returns_503(client, monkeypatch):
    monkeypatch.setattr(settings, "FEEDBACK_ENABLED", False)
    resp = client.post(
        "/feedback", json={"trace_id": "tr-x", "value": True, "source_id": "u"},
    )
    assert resp.status_code == 503

def test_feedback_returns_202_when_mlflow_raises(client):
    error = RuntimeError("FOREIGN KEY constraint failed")
    with patch("mlflow.log_feedback", side_effect=error), patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback", json={"trace_id": "tr-bad", "value": True, "source_id": "u"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert "FOREIGN KEY" in body["detail"]

def test_feedback_returns_202_when_tracking_uri_missing(client, monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", None)
    resp = client.post(
        "/feedback", json={"trace_id": "tr-x", "value": True, "source_id": "u"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"

# ── /feedback/expectation ───────────────────────────────────────────

def test_expectation_happy_path_response_only(client):
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback/expectation",
            json={
                "trace_id": "tr-abc",
                "expected_response": "42",
                "source_id": "jane.smith",
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["registered"] == ["expected_response"]
    assert log_mock.call_count == 1
    kwargs = log_mock.call_args.kwargs
    assert kwargs["name"] == "expected_response"
    assert kwargs["value"] == "42"

def test_expectation_registers_both_when_provided(client):
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback/expectation",
            json={
                "trace_id": "tr-abc",
                "expected_response": "42",
                "expected_facts": ["es par", "es la respuesta"],
                "source_id": "u",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["registered"] == ["expected_response", "expected_facts"]
    assert log_mock.call_count == 2
    names = [c.kwargs["name"] for c in log_mock.call_args_list]
    assert names == ["expected_response", "expected_facts"]

def test_expectation_without_any_field_returns_422(client):
    resp = client.post(
        "/feedback/expectation",
        json={"trace_id": "tr-abc", "source_id": "u"},
    )
    assert resp.status_code == 422
    assert "expected_response" in resp.json()["detail"]

def test_expectation_disabled_returns_503(client, monkeypatch):
    monkeypatch.setattr(settings, "FEEDBACK_ENABLED", False)
    resp = client.post(
        "/feedback/expectation",
        json={"trace_id": "tr-x", "expected_response": "x", "source_id": "u"},
    )
    assert resp.status_code == 503

def test_expectation_default_source_id_is_frontend_user(client):
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        client.post(
            "/feedback/expectation",
            json={"trace_id": "tr-x", "expected_response": "y"},
        )
    assert log_mock.call_args.kwargs["source"].source_id == "frontend_user"

def test_expectation_partial_failure_returns_202_with_what_succeeded(client):
    """Si expected_response se registra pero expected_facts falla, devolver 202 con lo logrado."""
    call_count = {"n": 0}

    def fake_log(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("MLflow caido")
        return MagicMock()

    with patch("mlflow.log_expectation", side_effect=fake_log), patch("mlflow.set_tracking_uri"):
        resp = client.post(
            "/feedback/expectation",
            json={
                "trace_id": "tr-x",
                "expected_response": "ok",
                "expected_facts": ["a"],
                "source_id": "u",
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["registered"] == ["expected_response"]
    assert "MLflow caido" in body["detail"]
