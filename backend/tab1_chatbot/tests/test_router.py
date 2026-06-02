"""Tests del router /chatbot del Tab 1 (chatbot simple)."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

def _make_chunk(content: str, is_last: bool = False) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.usage_metadata = (
        {"input_tokens": 10, "output_tokens": 5} if is_last else None
    )
    chunk.response_metadata = (
        {"model_name": "test-model"} if is_last else {}
    )
    return chunk

@pytest.fixture
def client(monkeypatch):
    from backend.tab1_chatbot import router as router_mod

    async def fake_astream_events(*args, **kwargs):
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": _make_chunk("Hola")},
        }
        last = _make_chunk(" mundo", is_last=True)
        yield {"event": "on_chat_model_stream", "data": {"chunk": last}}
        yield {"event": "on_chat_model_end", "data": {"output": last}}

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_astream_events
    monkeypatch.setattr(router_mod, "graph", mock_graph)

    app = FastAPI()
    app.include_router(router_mod.router, prefix="/chatbot")
    return TestClient(app)

def test_chatbot_stream_emits_token_and_done(client, parse_sse_events):
    response = client.post(
        "/chatbot/stream",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )
    assert response.status_code == 200

    events = parse_sse_events(response.text)
    tokens = [e for e in events if "token" in e and not e.get("done")]
    done = [e for e in events if e.get("done") is True]

    assert len(tokens) >= 1, f"esperaba algún evento token, recibí {events}"
    assert any(t["token"] == "Hola" for t in tokens)
    assert any(t["token"] == " mundo" for t in tokens)
    assert len(done) == 1
