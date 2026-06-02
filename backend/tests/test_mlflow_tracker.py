"""Tests para backend.mlflow_tracker — registro de inferencias en MLflow."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.inference_logger.models import LogEntry

@pytest.fixture(autouse=True)
def _reset_mlflow_state():
    """Resetea el estado cacheado de _mlflow_available entre tests."""
    import backend.mlflow_tracker as mod
    mod._mlflow_available = None
    yield
    mod._mlflow_available = None

class TestLogToMlflowDisabled:
    """Tests cuando MLflow no está disponible."""

    def test_no_tracking_uri_does_not_raise(self):
        """Sin MLFLOW_TRACKING_URI, log_to_mlflow no lanza excepción."""
        from backend.mlflow_tracker import log_to_mlflow

        env = os.environ.copy()
        env.pop("MLFLOW_TRACKING_URI", None)
        with patch.dict(os.environ, env, clear=True):
            entry = LogEntry(tab="chatbot", query="hola", response="mundo")
            log_to_mlflow(entry)  # No debe lanzar

    def test_no_tracking_uri_logs_warning(self, caplog):
        """Sin MLFLOW_TRACKING_URI, loguea un warning."""
        from backend.mlflow_tracker import log_to_mlflow

        env = os.environ.copy()
        env.pop("MLFLOW_TRACKING_URI", None)
        with patch.dict(os.environ, env, clear=True):
            import logging
            with caplog.at_level(logging.WARNING, logger="backend.mlflow_tracker"):
                entry = LogEntry(tab="chatbot", query="hola", response="mundo")
                log_to_mlflow(entry)
            assert "MLFLOW_TRACKING_URI" in caplog.text

class TestLogToMlflowEnabled:
    """Tests con MLflow mockeado."""

    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow:5000"})
    @patch("backend.mlflow_tracker.mlflow", create=True)
    def test_registers_experiment_and_run(self, mock_mlflow_module):
        """Verifica que se llama a set_experiment, start_run, log_params, log_metrics, log_artifact."""
        # Reimport to pick up the patched env
        import importlib
        import backend.mlflow_tracker as mod
        mod._mlflow_available = None

        # Patch the import inside the function
        mock_mlflow = MagicMock()
        mock_run_context = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_context)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        entry = LogEntry(
            tab="tramites",
            session_id="sess-1",
            query="tramites de Jaén",
            response="Aquí tienes los trámites...",
            model="alia-40b",
            time_total_ms=1500.0,
            time_retrieval_ms=200.0,
            time_reranking_ms=100.0,
            time_llm_ms=1200.0,
            tokens_input=150,
            tokens_output=42,
            context_preview="contexto preview",
            municipio="Jaén",
            intent="listado",
        )

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock_mlflow if name == "mlflow" else importlib.__import__(name, *a, **kw)):
            # Force re-check
            mod._mlflow_available = True
            mod.log_to_mlflow(entry, context_full="contexto completo extenso")

        mock_mlflow.set_experiment.assert_called_once_with("tramites")
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["tab"] == "tramites"
        assert params["municipio"] == "Jaén"
        assert params["intent"] == "listado"
        assert params["model"] == "alia-40b"

        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert metrics["time_total_ms"] == 1500.0
        assert metrics["tokens_output"] == 42

        mock_mlflow.log_artifact.assert_called_once()

    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow:5000"})
    def test_mlflow_exception_does_not_propagate(self):
        """Si MLflow lanza excepción, log_to_mlflow la captura."""
        import backend.mlflow_tracker as mod
        mod._mlflow_available = True

        with patch.dict("sys.modules", {"mlflow": MagicMock(set_experiment=MagicMock(side_effect=ConnectionError("MLflow down")))}):
            entry = LogEntry(tab="chatbot", query="hola", response="mundo")
            mod.log_to_mlflow(entry)  # No debe lanzar

class TestInferenceCaptureIntegration:
    """Verifica que InferenceCapture.finalize() llama a log_to_mlflow."""

    @patch("backend.inference_logger.middleware.log_to_mlflow")
    def test_finalize_calls_log_to_mlflow(self, mock_log):
        from backend.inference_logger.middleware import InferenceCapture

        capture = InferenceCapture(tab="chatbot", session_id="s1", query="hola")
        capture.add_token("mundo")
        capture.set_model("test-model")
        capture.set_context_preview("contexto de prueba completo")
        capture.finalize()

        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        entry = args[0]
        assert entry.tab == "chatbot"
        assert entry.response == "mundo"
        assert kwargs["context_full"] == "contexto de prueba completo"
