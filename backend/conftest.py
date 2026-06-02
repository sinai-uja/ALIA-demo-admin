"""Fixtures compartidas para tests del backend.

Reduce duplicación entre las suites de tab1, tab2, tab2bis, tab3 y
comparador. `backend/rag_tramites/tests/` mantiene su estilo histórico
sin depender de este conftest.

"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

def _mock_chunk(content: str, is_last: bool = False) -> MagicMock:
    """Chunk de streaming LLM compatible con la API de LangChain."""
    chunk = MagicMock()
    chunk.content = content
    chunk.usage_metadata = (
        {"input_tokens": 100, "output_tokens": 20} if is_last else None
    )
    chunk.response_metadata = (
        {"model_name": "test-model"} if is_last else {}
    )
    return chunk

def _parse_sse_events(response_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response_text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                continue
    return events

@pytest.fixture
def mock_llm():
    """LLM simulado con `ainvoke` (single response) y `astream` (chunks).

    Uso típico:
        def test_x(mock_llm, monkeypatch):
            monkeypatch.setattr("backend.tab1_chatbot.graph.llm", mock_llm)
            ...
    """
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="Respuesta simulada")

    async def fake_astream(messages, *args, **kwargs):
        yield _mock_chunk("Hola")
        yield _mock_chunk(" mundo", is_last=True)

    llm.astream = fake_astream
    return llm

@pytest.fixture
def mock_chroma_vectorstore():
    """Vectorstore Chroma simulado con `similarity_search` predefinido."""
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(
            page_content="Documento de prueba 1",
            metadata={"source": "test-1.txt", "question": "¿Cómo solicitar?"},
        ),
        Document(
            page_content="Documento de prueba 2",
            metadata={"source": "test-2.txt"},
        ),
    ]
    return vs

@pytest.fixture
def parse_sse_events():
    """Devuelve una función que parsea eventos SSE de un `response.text`.

    Uso:
        response = client.post("/chatbot/stream", json={...})
        events = parse_sse_events(response.text)
    """
    return _parse_sse_events

@pytest.fixture
def make_sse_client(parse_sse_events):
    """Helper que envía una request al TestClient y devuelve los eventos SSE.

    Uso:
        events = make_sse_client(client, "POST", "/rag/stream", json={...})
        assert any(e["type"] == "token" for e in events)
    """
    def _request(client, method: str, url: str, **kwargs) -> list[dict[str, Any]]:
        response = client.request(method, url, **kwargs)
        assert response.status_code == 200, (
            f"SSE request failed: {response.status_code} {response.text[:200]}"
        )
        return parse_sse_events(response.text)
    return _request

@pytest.fixture
def mock_settings_overrides(monkeypatch):
    """Aplica overrides al singleton `settings` para el alcance del test.

    Uso:
        def test_x(mock_settings_overrides):
            mock_settings_overrides(MLFLOW_TRACKING_URI="http://mock:5000")
            ...
    """
    from backend.config import settings

    def _apply(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setattr(settings, key, value)

    return _apply
