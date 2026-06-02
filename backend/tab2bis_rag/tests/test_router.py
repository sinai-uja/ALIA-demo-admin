"""Tests del router /rag del Tab 2bis (RAG siempre activo)."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

def _make_chunk(content: str, is_last: bool = False) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.usage_metadata = (
        {"input_tokens": 12, "output_tokens": 8} if is_last else None
    )
    chunk.response_metadata = (
        {"model_name": "test-model"} if is_last else {}
    )
    return chunk

@pytest.fixture
def client(monkeypatch):
    from backend.tab2bis_rag import router as router_mod

    async def fake_astream_events(*args, **kwargs):
        yield {"event": "on_chain_start", "name": "retrieve", "data": {}}
        yield {
            "event": "on_chain_end",
            "name": "retrieve",
            "data": {"output": {"context": "Documento de prueba"}},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": _make_chunk("respuesta")},
        }
        last = _make_chunk(" final", is_last=True)
        yield {"event": "on_chat_model_stream", "data": {"chunk": last}}
        yield {"event": "on_chat_model_end", "data": {"output": last}}

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_astream_events
    monkeypatch.setattr(router_mod, "graph", mock_graph)

    app = FastAPI()
    app.include_router(router_mod.router, prefix="/rag")
    return TestClient(app)

def test_rag_stream_emits_retrieve_token_and_done(client, parse_sse_events):
    response = client.post(
        "/rag/stream",
        json={"messages": [{"role": "user", "content": "consulta"}]},
    )
    assert response.status_code == 200

    events = parse_sse_events(response.text)
    types = {e.get("type") for e in events}

    assert "retrieve" in types
    assert "token" in types
    assert "done" in types

    tokens = [e for e in events if e.get("type") == "token"]
    assert any(e["content"] == "respuesta" for e in tokens)
    assert any(e["content"] == " final" for e in tokens)
