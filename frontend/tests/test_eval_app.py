"""Tests del entrypoint paralelo `frontend/eval_app.py`.

Cubre tres bloques:

1. Smoke del entrypoint: el `gr.Blocks` se construye, expone una sola tab
   visible (Trámites) y el título correcto.
2. Flujo SSE Trámites: el handler `bot_response` parsea correctamente los
   eventos del backend (`trace_id`, `retrieve`, `token`, `tramites`, `done`)
   y los traduce al formato que consume el `gr.Chatbot`.
3. Validación de identidad: `save_evaluator_identity` valida API key contra
   `/auth/check` y propaga el alias al `user_id_state`.

La atribución del feedback en MLflow al `user_id` se verifica end-to-end
manualmente; el widget vive en `frontend/components/feedback_widget.py`
y su test unitario pertenece a esa unidad cuando exista.

Convención de async: el repo usa `asyncio.run(...)` desde tests síncronos
(ver `backend/tab2bis_rag/tests/test_graph.py`); aquí se replica.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import gradio as gr
import httpx

from frontend import eval_app

def _run(coro):
    return asyncio.run(coro)

async def _drain(agen) -> list[tuple[list, str]]:
    return [item async for item in agen]

# ---------------------------------------------------------------------------
# 1. Smoke del entrypoint
# ---------------------------------------------------------------------------

def test_demo_is_gr_blocks_with_eval_title():
    assert isinstance(eval_app.demo, gr.Blocks)
    assert "Evaluación pública" in (eval_app.demo.title or "")

def test_module_exposes_handlers_for_testing():
    """Los handlers están a nivel de módulo (no closures) para facilitar tests."""
    for attr in ("user_message", "bot_response", "clear_conversation", "save_evaluator_identity"):
        assert hasattr(eval_app, attr), f"eval_app debe exponer {attr}"

# ---------------------------------------------------------------------------
# 2. Flujo SSE Trámites — bot_response
# ---------------------------------------------------------------------------

def _build_mock_async_client(sse_lines: list[str], status_code: int = 200):
    """Mock async-context-manager-friendly que simula
    `async with httpx.AsyncClient() as client: async with client.stream(...) as r`.
    """
    response = MagicMock()
    response.status_code = status_code

    async def aiter_lines():
        for line in sse_lines:
            yield line

    response.aiter_lines = aiter_lines

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    return client_cm

def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"

def test_bot_response_yields_empty_when_no_history():
    yields = _run(_drain(eval_app.bot_response([], "key", "sid", "")))
    assert yields == [([], "")]

def test_bot_response_warns_when_no_api_key():
    history = [{"role": "user", "content": "hola"}]
    yields = _run(_drain(eval_app.bot_response(history, "", "sid", "")))
    assert len(yields) == 1
    new_history, trace = yields[0]
    assert trace == ""
    assert new_history[-1]["role"] == "assistant"
    assert "Identifícate" in new_history[-1]["content"]

def test_bot_response_emits_trace_token_and_tramites(monkeypatch):
    sse_stream = [
        _sse({"type": "trace_id", "trace_id": "trace-abc"}),
        _sse({"type": "retrieve", "content": "buscando trámites"}),
        _sse({"type": "token", "content": "Hola"}),
        _sse({"type": "token", "content": " mundo"}),
        _sse({
            "type": "tramites",
            "content": [
                {"nombre": "Empadronamiento", "municipio": "Jaén", "url": "https://x", "auth": "cert"},
            ],
        }),
        _sse({"type": "done"}),
    ]
    monkeypatch.setattr(
        eval_app.httpx, "AsyncClient",
        lambda *a, **kw: _build_mock_async_client(sse_stream),
    )

    history = [{"role": "user", "content": "¿Cómo me empadrono?"}]
    yields = _run(_drain(eval_app.bot_response(history, "valid-key", "sid-1", "evaluador_X")))

    traces = {trace for _, trace in yields if trace}
    assert traces == {"trace-abc"}, f"esperaba captura única del trace_id, vi {traces}"

    final_history, _ = yields[-1]
    final_content = final_history[-1]["content"]
    assert "Hola mundo" in final_content
    assert "🔍 buscando trámites" in final_content
    assert "Empadronamiento" in final_content
    assert "https://x" in final_content
    assert "Auth: cert" in final_content

def test_bot_response_handles_auth_error(monkeypatch):
    monkeypatch.setattr(
        eval_app.httpx, "AsyncClient",
        lambda *a, **kw: _build_mock_async_client([], status_code=401),
    )

    history = [{"role": "user", "content": "hola"}]
    yields = _run(_drain(eval_app.bot_response(history, "bad-key", "sid", "")))

    assert len(yields) == 1
    final_history, trace = yields[0]
    assert trace == ""
    assert "API key inválida" in final_history[-1]["content"]

def test_bot_response_handles_network_error(monkeypatch):
    class _BoomClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("connection refused")

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr(eval_app.httpx, "AsyncClient", _BoomClient)

    history = [{"role": "user", "content": "hola"}]
    yields = _run(_drain(eval_app.bot_response(history, "key", "sid", "")))

    assert len(yields) >= 1
    last_history, _ = yields[-1]
    assert "Error de conexión con el backend" in last_history[-1]["content"]

# ---------------------------------------------------------------------------
# 3. save_evaluator_identity
# ---------------------------------------------------------------------------

_REAL_ASYNC_CLIENT = httpx.AsyncClient

def _patched_async_client(monkeypatch, transport: httpx.MockTransport):
    """Sustituye `eval_app.httpx.AsyncClient` por uno con MockTransport.

    Captura la clase real antes del monkeypatch para evitar recursión:
    `eval_app.httpx is httpx` (mismo módulo importado), por lo que el patch
    afecta también a `httpx.AsyncClient` global.
    """
    class _Client:
        def __init__(self, *a, **kw):
            kw.pop("transport", None)
            self._client = _REAL_ASYNC_CLIENT(transport=transport, **kw)

        async def __aenter__(self):
            return await self._client.__aenter__()

        async def __aexit__(self, *a):
            return await self._client.__aexit__(*a)

    monkeypatch.setattr(eval_app.httpx, "AsyncClient", _Client)

def test_save_evaluator_identity_rejects_empty():
    api_key, user_id, status = _run(eval_app.save_evaluator_identity("", ""))
    assert (api_key, user_id) == ("", "")
    assert "Introduce" in status

def test_save_evaluator_identity_requires_user_id():
    api_key, user_id, status = _run(eval_app.save_evaluator_identity("key", ""))
    assert api_key == "key" and user_id == ""
    assert "alias del evaluador" in status.lower()

def test_save_evaluator_identity_validates_key_against_backend(monkeypatch):
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    _patched_async_client(monkeypatch, httpx.MockTransport(handler))

    api_key, user_id, status = _run(
        eval_app.save_evaluator_identity("good-key", "evaluador_X")
    )
    assert api_key == "good-key"
    assert user_id == "evaluador_X"
    assert "evaluador_X" in status
    assert captured["auth"] == "Bearer good-key"
    assert captured["url"].endswith("/auth/check")

def test_save_evaluator_identity_rejects_invalid_key(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid"})

    _patched_async_client(monkeypatch, httpx.MockTransport(handler))

    api_key, user_id, status = _run(
        eval_app.save_evaluator_identity("bad-key", "evaluador_X")
    )
    assert api_key == ""
    assert user_id == "evaluador_X"
    assert "API Key inválida" in status
