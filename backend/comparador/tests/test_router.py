"""
Tests unitarios para el router del comparador dual de LLMs.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.comparador.router import router

@pytest.fixture
def app():
    """Crea una app FastAPI de test con el router del comparador."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/comparador")
    return test_app

@pytest.fixture
def client(app):
    return TestClient(app)

def _mock_chunk(content: str, is_last: bool = False):
    """Crea un mock de chunk de streaming LLM."""
    chunk = MagicMock()
    chunk.content = content
    chunk.usage_metadata = (
        {"input_tokens": 100, "output_tokens": 20} if is_last else None
    )
    chunk.response_metadata = (
        {"model_name": "test-model"} if is_last else {}
    )
    return chunk

def _parse_sse_events(response_text: str) -> list[dict]:
    """Parsea eventos SSE del cuerpo de la respuesta."""
    events = []
    for line in response_text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
    return events

class TestComparadorStream:
    """Tests del endpoint POST /comparador/stream."""

    @patch("backend.comparador.router.node_detect_intent")
    @patch("backend.comparador.router.node_detect_municipio")
    @patch("backend.comparador.router.node_retrieve")
    @patch("backend.comparador.router.node_rerank")
    @patch("backend.comparador.router.llm_a")
    @patch("backend.comparador.router.llm_b")
    def test_sse_events_structure(
        self, mock_llm_b, mock_llm_a,
        mock_rerank, mock_retrieve, mock_municipio, mock_intent, client,
    ):
        """Verifica que los eventos SSE tienen los campos type, llm, content."""
        mock_municipio.return_value = {"municipio_detectado": "Jaén"}
        mock_intent.return_value = {"intent": "especifico"}
        mock_retrieve.return_value = {
            "candidatos": [
                {"documento": "Empadronamiento", "metadata": {"nombre": "Empadronamiento", "municipio": "Jaén", "url": "http://jaen.es", "auth": "Certificado"}},
            ],
        }
        mock_rerank.return_value = {
            "candidatos_reranked": [
                {"documento": "Empadronamiento", "metadata": {"nombre": "Empadronamiento", "municipio": "Jaén", "url": "http://jaen.es", "auth": "Certificado"}, "rerank_score": 0.9},
            ],
        }

        chunks = [_mock_chunk("Hola"), _mock_chunk(" mundo", is_last=True)]

        async def fake_astream(messages):
            for c in chunks:
                yield c

        mock_llm_a.astream = fake_astream
        mock_llm_b.astream = fake_astream

        response = client.post(
            "/comparador/stream",
            json={"query": "trámites en Jaén"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        types = {e.get("type") for e in events}

        assert "retrieve" in types
        assert "models" in types
        assert "token" in types
        assert "metrics" in types
        assert "done" in types

        token_events = [e for e in events if e["type"] == "token"]
        llm_labels = {e["llm"] for e in token_events}
        assert "A" in llm_labels
        assert "B" in llm_labels
        assert all("content" in e for e in token_events)

    @patch("backend.comparador.router.node_detect_intent")
    @patch("backend.comparador.router.node_detect_municipio")
    @patch("backend.comparador.router.node_retrieve")
    @patch("backend.comparador.router.node_rerank")
    @patch("backend.comparador.router.llm_a")
    @patch("backend.comparador.router.llm_b")
    @patch("backend.comparador.router.InferenceCapture")
    def test_two_log_entries_created(
        self, mock_capture_cls, mock_llm_b, mock_llm_a,
        mock_rerank, mock_retrieve, mock_municipio, mock_intent, client,
    ):
        """Verifica que se crean dos LogEntry con tab='comparador'."""
        mock_municipio.return_value = {"municipio_detectado": None}
        mock_intent.return_value = {"intent": "especifico"}
        mock_retrieve.return_value = {"candidatos": [{"documento": "Test", "metadata": {"nombre": "Test", "municipio": "X", "url": "", "auth": ""}}]}
        mock_rerank.return_value = {"candidatos_reranked": [{"documento": "Test", "metadata": {"nombre": "Test", "municipio": "X", "url": "", "auth": ""}}]}

        mock_entry = MagicMock()
        mock_entry.model = "test"
        mock_entry.time_total_ms = 100.0
        mock_entry.tokens_output = 10

        mock_instance = MagicMock()
        mock_instance.finalize.return_value = mock_entry
        mock_capture_cls.return_value = mock_instance

        async def fake_astream(messages):
            yield _mock_chunk("ok", is_last=True)

        mock_llm_a.astream = fake_astream
        mock_llm_b.astream = fake_astream

        response = client.post(
            "/comparador/stream",
            json={"query": "test"},
        )
        assert response.status_code == 200

        # Se crean exactamente 2 InferenceCapture con tab="comparador"
        assert mock_capture_cls.call_count == 2
        for call in mock_capture_cls.call_args_list:
            assert call.kwargs.get("tab") == "comparador" or call.args[0] == "comparador" if call.args else call.kwargs.get("tab") == "comparador"

    @patch("backend.comparador.router.llm_b", new=MagicMock())
    @patch("backend.comparador.router.llm_a", new=MagicMock())
    def test_empty_query_returns_error(self, client):
        """Verifica que una query vacía devuelve un evento de error."""
        response = client.post(
            "/comparador/stream",
            json={"query": "   "},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) > 0
        assert "vacía" in error_events[0]["content"].lower()

class TestComparadorAvailability:
    """Tests para el caso `llm_b is None`."""

    def test_returns_503_when_llm_b_is_none(self, client, monkeypatch):
        """Sin segundo LLM configurado, el endpoint devuelve 503 con detalle."""
        monkeypatch.setattr("backend.comparador.router.llm_b", None)

        response = client.post(
            "/comparador/stream",
            json={"query": "trámites en Jaén"},
        )
        assert response.status_code == 503
        body = response.json()
        assert "ALIA_LLM_URL_2" in body["detail"]
        assert "ALIA_LLM_MODEL_2" in body["detail"]

class TestComparadorTraceIds:
    """Tests para la emisión de trace_id por canal (A/B)."""

    @patch("backend.comparador.router.node_detect_intent")
    @patch("backend.comparador.router.node_detect_municipio")
    @patch("backend.comparador.router.node_retrieve")
    @patch("backend.comparador.router.node_rerank")
    @patch("backend.comparador.router.llm_a")
    @patch("backend.comparador.router.llm_b")
    def test_emits_trace_id_per_canal(
        self, mock_llm_b, mock_llm_a,
        mock_rerank, mock_retrieve, mock_municipio, mock_intent, client,
    ):
        mock_municipio.return_value = {"municipio_detectado": None}
        mock_intent.return_value = {"intent": "especifico"}
        mock_retrieve.return_value = {"candidatos": []}
        mock_rerank.return_value = {"candidatos_reranked": []}

        async def fake_astream(messages):
            yield _mock_chunk("ok", is_last=True)

        mock_llm_a.astream = fake_astream
        mock_llm_b.astream = fake_astream

        response = client.post(
            "/comparador/stream",
            json={"query": "consulta"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        trace_events = [e for e in events if e.get("type") == "trace_id"]
        canales = {e.get("llm") for e in trace_events}

        assert "A" in canales
        assert "B" in canales
