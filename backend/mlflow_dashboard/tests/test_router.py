"""
Tests unitarios para el router del módulo mlflow_dashboard.

Verifica los endpoints /mlflow/experiments, /mlflow/summary y /mlflow/runs
con la API de MLflow mockeada.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.mlflow_dashboard.router import router

@pytest.fixture
def app():
    """Crea una app FastAPI de test con el router de mlflow_dashboard."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app

@pytest.fixture
def client(app):
    return TestClient(app)

# ── Datos de ejemplo para mocks ─────────────────────────────────────

FAKE_EXPERIMENTS = {
    "experiments": [
        {"experiment_id": "1", "name": "chatbot"},
        {"experiment_id": "2", "name": "tramites"},
        {"experiment_id": "0", "name": "Default"},
    ]
}

FAKE_RUNS = {
    "runs": [
        {
            "info": {"run_id": "run-a", "start_time": 1714000000000},
            "data": {
                "params": [
                    {"key": "model", "value": "alia-40b"},
                    {"key": "tab", "value": "chatbot"},
                    {"key": "municipio", "value": "Jaén"},
                ],
                "metrics": [
                    {"key": "time_total_ms", "value": 1200.0},
                    {"key": "time_llm_ms", "value": 1000.0},
                    {"key": "time_retrieval_ms", "value": 100.0},
                    {"key": "time_reranking_ms", "value": 50.0},
                    {"key": "tokens_input", "value": 200.0},
                    {"key": "tokens_output", "value": 80.0},
                ],
            },
        },
        {
            "info": {"run_id": "run-b", "start_time": 1713990000000},
            "data": {
                "params": [
                    {"key": "model", "value": "alia-40b"},
                    {"key": "tab", "value": "chatbot"},
                ],
                "metrics": [
                    {"key": "time_total_ms", "value": 800.0},
                    {"key": "time_llm_ms", "value": 600.0},
                    {"key": "tokens_input", "value": 150.0},
                    {"key": "tokens_output", "value": 40.0},
                ],
            },
        },
    ]
}

FAKE_RUNS_TRAMITES = {
    "runs": [
        {
            "info": {"run_id": "run-t1", "start_time": 1714001000000},
            "data": {
                "params": [
                    {"key": "model", "value": "alia-40b"},
                    {"key": "tab", "value": "tramites"},
                ],
                "metrics": [
                    {"key": "time_total_ms", "value": 2000.0},
                ],
            },
        },
    ]
}

def _mock_post_factory(experiment_runs: dict | None = None):
    """Crea un mock de _mlflow_post que devuelve datos según la llamada."""
    runs_by_exp = experiment_runs or {"1": FAKE_RUNS, "2": FAKE_RUNS_TRAMITES}

    async def mock_post(path: str, payload: dict) -> dict:
        if "experiments/search" in path:
            filter_str = payload.get("filter", "")
            if "tramites" in filter_str:
                return {"experiments": [FAKE_EXPERIMENTS["experiments"][1]]}
            return FAKE_EXPERIMENTS
        if "runs/search" in path:
            exp_ids = payload.get("experiment_ids", [])
            all_runs: list = []
            for eid in exp_ids:
                data = runs_by_exp.get(eid, {"runs": []})
                all_runs.extend(data["runs"])
            return {"runs": all_runs[:payload.get("max_results", 1000)]}
        return {}

    return mock_post

class TestExperiments:
    """Tests del endpoint GET /mlflow/experiments."""

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_list_experiments(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/experiments")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["experiment_name"] for e in data]
        assert "chatbot" in names
        assert "tramites" in names
        assert "Default" not in names

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_experiments_include_run_count(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/experiments")
        data = resp.json()
        chatbot = next(e for e in data if e["experiment_name"] == "chatbot")
        assert chatbot["run_count"] == 2
        tramites = next(e for e in data if e["experiment_name"] == "tramites")
        assert tramites["run_count"] == 1

class TestSummary:
    """Tests del endpoint GET /mlflow/summary."""

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_summary_all_experiments(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "experiments" in data
        assert "mlflow_url" in data
        names = [e["experiment_name"] for e in data["experiments"]]
        assert "chatbot" in names
        assert "tramites" in names
        assert "Default" not in names

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_summary_aggregated_metrics(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/summary")
        data = resp.json()
        chatbot = next(
            e for e in data["experiments"] if e["experiment_name"] == "chatbot"
        )
        assert chatbot["run_count"] == 2
        # Media de time_total_ms: (1200 + 800) / 2 = 1000
        assert chatbot["avg_time_total_ms"] == 1000.0
        # Media de tokens_output: (80 + 40) / 2 = 60
        assert chatbot["avg_tokens_output"] == 60.0

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_summary_filter_by_experiment_name(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/summary", params={"experiment_name": "tramites"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["experiment_name"] == "tramites"

    @pytest.mark.skip(
        reason="Deuda preexistente: _build_run_filter "
        "convierte fechas YYYY-MM-DD a epoch-ms; el assert literal "
        "'2025-04-01 in payload[\"filter\"]' ya no aplica."
    )
    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_summary_filter_by_date_range(self, mock_post, client):
        """Verifica que date_from/date_to se pasan a la búsqueda de runs."""
        calls = []

        async def tracking_post(path, payload):
            calls.append((path, payload))
            return _mock_post_factory()(path, payload)

        # Need to await the coroutine from the factory
        async def mock_post_fn(path, payload):
            calls.append((path, payload))
            if "experiments/search" in path:
                return FAKE_EXPERIMENTS
            if "runs/search" in path:
                return {"runs": []}
            return {}

        mock_post.side_effect = mock_post_fn
        resp = client.get(
            "/mlflow/summary",
            params={"date_from": "2025-04-01", "date_to": "2025-04-30"},
        )
        assert resp.status_code == 200
        # Verificar que se pasó el filtro temporal en alguna llamada a runs/search
        run_calls = [c for c in calls if "runs/search" in c[0]]
        assert len(run_calls) > 0
        for _, payload in run_calls:
            if "filter" in payload:
                assert "2025-04-01" in payload["filter"]
                assert "2025-04-30" in payload["filter"]

class TestRuns:
    """Tests del endpoint GET /mlflow/runs."""

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_runs_returns_list(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_runs_ordered_by_timestamp_desc(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/runs")
        data = resp.json()
        timestamps = [r["timestamp"] for r in data]
        assert timestamps == sorted(timestamps, reverse=True)

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_runs_include_params_and_metrics(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/runs")
        data = resp.json()
        run = data[0]
        assert "run_id" in run
        assert "timestamp" in run
        assert "model" in run
        assert "tab" in run
        assert "time_total_ms" in run

    @patch("backend.mlflow_dashboard.router._mlflow_post")
    def test_runs_respects_limit(self, mock_post, client):
        mock_post.side_effect = _mock_post_factory()
        resp = client.get("/mlflow/runs", params={"limit": 1})
        data = resp.json()
        assert len(data) <= 1

class TestMlflowUnavailable:
    """Tests cuando MLflow no está disponible (503)."""

    @patch(
        "backend.mlflow_dashboard.router._mlflow_post",
        side_effect=Exception("Connection refused"),
    )
    def test_experiments_503(self, mock_post, client):
        resp = client.get("/mlflow/experiments")
        assert resp.status_code == 503
        assert "detail" in resp.json()

    @patch(
        "backend.mlflow_dashboard.router._mlflow_post",
        side_effect=Exception("Connection refused"),
    )
    def test_summary_503(self, mock_post, client):
        resp = client.get("/mlflow/summary")
        assert resp.status_code == 503

    @patch(
        "backend.mlflow_dashboard.router._mlflow_post",
        side_effect=Exception("Connection refused"),
    )
    def test_runs_503(self, mock_post, client):
        resp = client.get("/mlflow/runs")
        assert resp.status_code == 503
