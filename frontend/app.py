"""
Frontend Gradio con pestañas para la aplicación LangGraph Experimental.

Cada pestaña se comunica con su endpoint correspondiente en el backend FastAPI.
Todas las peticiones incluyen autenticación Bearer con la API key configurada
en la pestaña Parámetros.
"""

import base64
import datetime
import json
import mimetypes
import os
import uuid

import gradio as gr
import httpx
from dotenv import load_dotenv

import sys as _sys

_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _FRONTEND_DIR not in _sys.path:
    _sys.path.insert(0, _FRONTEND_DIR)

from components import render_feedback_widget  # noqa: E402

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
FEEDBACK_ENABLED = os.getenv("FEEDBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

# Directorio base de logos
_LOGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "diseno", "Logos")

# Registro de session_states que se inicializan por sesión vía demo.load().
# Cada `create_tab_*` que necesita aislamiento de sesión añade aquí su State;
# el bloque principal registra un único demo.load() que les asigna un UUID
# distinto por cada navegador que abre la app. Esto sustituye al patrón
# gr.State(value=lambda: uuid.uuid4()) que en Gradio 6.x causaba que la app
# dejara de responder a interacciones tras unos segundos.
_SESSION_STATES_TO_INIT: list = []

def _logo_b64(filename: str) -> str:
    """Codifica un logo en base64 para uso inline en HTML."""
    path = os.path.join(_LOGOS_DIR, filename)
    if not os.path.exists(path):
        return ""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def _auth_headers(api_key: str) -> dict:
    """Genera las cabeceras de autenticación Bearer."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}

# =============================================================================
# PESTAÑA 1 — ChatBot
# =============================================================================

def create_tab1(api_key_state, user_id_state):
    """Crea la pestaña del chatbot simple."""
    with gr.Tab("ChatBot"):
        gr.Markdown("### Chat con LLM (LangGraph Simple)")
        session_state = gr.State("")
        _SESSION_STATES_TO_INIT.append(session_state)
        trace_id_state = gr.State("")
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(
            placeholder="Escribe tu mensaje...",
            show_label=False,
        )
        with gr.Row():
            send_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar conversación")

        if FEEDBACK_ENABLED:
            render_feedback_widget(api_key_state, trace_id_state, user_id_state)

        async def user_message(message, history):
            """Añade el mensaje del usuario al historial."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history

        async def bot_response(history, api_key, sid, user_id):
            """Envía el historial al backend y muestra la respuesta en streaming."""
            if not history:
                yield history, ""
                return

            if not api_key:
                history = list(history) + [{"role": "assistant", "content": "Configura la API Key en la pestaña Parámetros."}]
                yield history, ""
                return

            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            history = list(history) + [{"role": "assistant", "content": ""}]
            captured_trace = ""

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{BACKEND_URL}/chatbot/stream",
                        json={"messages": messages, "session_id": sid, "user_id": (user_id or "").strip() or None},
                        headers=_auth_headers(api_key),
                    ) as response:
                        if response.status_code in (401, 403):
                            history[-1] = {"role": "assistant", "content": "Error de autenticación: API key inválida"}
                            yield history, captured_trace
                            return
                        buffer = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("type") == "trace_id":
                                        captured_trace = data.get("trace_id", "")
                                        yield history, captured_trace
                                        continue
                                    if data.get("done"):
                                        break
                                    token = data.get("token", "")
                                    buffer += token
                                    history[-1] = {"role": "assistant", "content": buffer}
                                    yield history, captured_trace
                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                history[-1] = {"role": "assistant", "content": f"Error de conexión con el backend: {str(e)}"}
                yield history, captured_trace

        def clear_conversation():
            """Limpia el chat y genera un nuevo session_id."""
            return [], str(uuid.uuid4()), ""

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        clear_btn.click(clear_conversation, outputs=[chatbot, session_state, trace_id_state])

# =============================================================================
# PESTAÑA 2 — Agente ReAct
# =============================================================================

def create_tab2(api_key_state, user_id_state):
    """Crea la pestaña del agente ReAct con RAG."""
    with gr.Tab("Agente ReAct"):
        gr.Markdown("### Agente ReAct con RAG (ChromaDB)")
        gr.Markdown(
            "Este agente puede buscar en la base de conocimiento local "
            "(ChromaDB) para responder tus preguntas."
        )
        gr.Markdown(
            "📚 **Contenido vectorizado:** la base de conocimiento contiene el contenido de "
            "[ALIA Kit — FAQ de Adopción/Integración](https://langtech-bsc.gitbook.io/alia-kit/integracion/adopcion-faq). "
            "Puedes preguntar sobre temas relacionados con la integración y adopción de ALIA Kit."
        )
        session_state = gr.State("")
        _SESSION_STATES_TO_INIT.append(session_state)
        trace_id_state = gr.State("")

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(
            placeholder="Pregunta algo sobre la base de conocimiento...",
            show_label=False,
        )
        with gr.Row():
            send_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar")

        if FEEDBACK_ENABLED:
            render_feedback_widget(api_key_state, trace_id_state, user_id_state)

        async def user_message(message, history):
            """Añade el mensaje del usuario al historial."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history

        async def bot_response(history, api_key, sid, user_id):
            """Envía al backend y muestra respuesta con pasos intermedios."""
            if not history:
                yield history, ""
                return

            if not api_key:
                history = list(history) + [{"role": "assistant", "content": "Configura la API Key en la pestaña Parámetros."}]
                yield history, ""
                return

            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            history = history + [{"role": "assistant", "content": ""}]
            captured_trace = ""

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{BACKEND_URL}/react-agent/stream",
                        json={"messages": messages, "session_id": sid, "user_id": (user_id or "").strip() or None},
                        headers=_auth_headers(api_key),
                    ) as response:
                        if response.status_code in (401, 403):
                            history[-1]["content"] = "Error de autenticación: API key inválida"
                            yield history, captured_trace
                            return
                        buffer = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    event_type = data.get("type", "")

                                    if event_type == "trace_id":
                                        captured_trace = data.get("trace_id", "")
                                        yield history, captured_trace

                                    elif event_type == "token":
                                        token = data.get("content", "")
                                        buffer += token
                                        history[-1]["content"] = buffer
                                        yield history, captured_trace

                                    elif event_type == "tool_call":
                                        tool_input = data.get("input", "")
                                        tool_msg = f'\n\n🔍 Buscando en knowledge base: "{tool_input}"\n\n'
                                        buffer += tool_msg
                                        history[-1]["content"] = buffer
                                        yield history, captured_trace

                                    elif event_type == "tool_result":
                                        result_content = data.get("content", "")
                                        result_preview = result_content[:200]
                                        result_msg = f'📄 Resultado: "{result_preview}..."\n\n'
                                        buffer += result_msg
                                        history[-1]["content"] = buffer
                                        yield history, captured_trace

                                    elif event_type == "done":
                                        break

                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                history[-1]["content"] = f"Error de conexión con el backend: {str(e)}"
                yield history, captured_trace

        def clear_conversation():
            """Limpia el chat y genera un nuevo session_id."""
            return [], str(uuid.uuid4()), ""

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        clear_btn.click(clear_conversation, outputs=[chatbot, session_state, trace_id_state])

# =============================================================================
# PESTAÑA 2bis — RAG (siempre activo)
# =============================================================================

def create_tab2bis(api_key_state, user_id_state):
    """Crea la pestaña de RAG siempre activo (sin depender de tool calling)."""
    with gr.Tab("RAG"):
        gr.Markdown("### RAG — Consulta la base de conocimiento")
        gr.Markdown(
            "Este chat **siempre** busca en ChromaDB antes de responder. "
            "No depende de tool calling del LLM."
        )
        gr.Markdown(
            "📚 **Contenido vectorizado:** la base de conocimiento contiene el contenido de "
            "[ALIA Kit — FAQ de Adopción/Integración](https://langtech-bsc.gitbook.io/alia-kit/integracion/adopcion-faq). "
            "Puedes preguntar sobre temas relacionados con la integración y adopción de ALIA Kit."
        )
        session_state = gr.State("")
        _SESSION_STATES_TO_INIT.append(session_state)
        trace_id_state = gr.State("")

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(
            placeholder="Pregunta algo sobre la base de conocimiento...",
            show_label=False,
        )
        with gr.Row():
            send_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar")

        if FEEDBACK_ENABLED:
            render_feedback_widget(api_key_state, trace_id_state, user_id_state)

        async def user_message(message, history):
            """Añade el mensaje del usuario al historial."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history

        async def bot_response(history, api_key, sid, user_id):
            """Envía al backend RAG y muestra respuesta con contexto recuperado."""
            if not history:
                yield history, ""
                return

            if not api_key:
                history = list(history) + [{"role": "assistant", "content": "Configura la API Key en la pestaña Parámetros."}]
                yield history, ""
                return

            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            history = list(history) + [{"role": "assistant", "content": ""}]
            captured_trace = ""

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{BACKEND_URL}/rag/stream",
                        json={"messages": messages, "session_id": sid, "user_id": (user_id or "").strip() or None},
                        headers=_auth_headers(api_key),
                    ) as response:
                        if response.status_code in (401, 403):
                            history[-1] = {"role": "assistant", "content": "Error de autenticación: API key inválida"}
                            yield history, captured_trace
                            return
                        buffer = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    event_type = data.get("type", "")

                                    if event_type == "trace_id":
                                        captured_trace = data.get("trace_id", "")
                                        yield history, captured_trace

                                    elif event_type == "retrieve":
                                        buffer += "🔍 Buscando en la base de conocimiento...\n\n"
                                        history[-1] = {"role": "assistant", "content": buffer}
                                        yield history, captured_trace

                                    elif event_type == "context":
                                        context_preview = data.get("content", "")[:200]
                                        buffer += f"📄 Contexto encontrado: \"{context_preview}...\"\n\n"
                                        history[-1] = {"role": "assistant", "content": buffer}
                                        yield history, captured_trace

                                    elif event_type == "token":
                                        token = data.get("content", "")
                                        buffer += token
                                        history[-1] = {"role": "assistant", "content": buffer}
                                        yield history, captured_trace

                                    elif event_type == "done":
                                        break

                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                history[-1] = {"role": "assistant", "content": f"Error de conexión con el backend: {str(e)}"}
                yield history, captured_trace

        def clear_conversation():
            """Limpia el chat y genera un nuevo session_id."""
            return [], str(uuid.uuid4()), ""

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        clear_btn.click(clear_conversation, outputs=[chatbot, session_state, trace_id_state])

# =============================================================================
# PESTAÑA Trámites — RAG Trámites Municipales
# =============================================================================

def create_tab_tramites(api_key_state, user_id_state):
    """Crea la pestaña del asistente de trámites municipales."""
    with gr.Tab("Trámites", id="tab-tramites"):
        gr.Markdown("### Trámites Municipales de Andalucía")
        gr.Markdown(
            "Consulta trámites administrativos municipales. "
            "Puedes especificar el municipio para resultados más precisos."
        )
        session_state = gr.State("")
        _SESSION_STATES_TO_INIT.append(session_state)
        trace_id_state = gr.State("")

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(
            placeholder="Ej: ¿Cómo me empadrono en Adamuz?",
            show_label=False,
        )
        with gr.Row():
            send_btn = gr.Button("Enviar", variant="primary")
            clear_btn = gr.Button("Limpiar")

        if FEEDBACK_ENABLED:
            render_feedback_widget(api_key_state, trace_id_state, user_id_state)

        async def user_message(message, history):
            """Añade el mensaje del usuario al historial."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            return "", history

        async def bot_response(history, api_key, sid, user_id):
            """Envía al backend de trámites y muestra respuesta con streaming."""
            if not history:
                yield history, ""
                return

            if not api_key:
                history = list(history) + [{"role": "assistant", "content": "Configura la API Key en la pestaña Parámetros."}]
                yield history, ""
                return

            last_user_msg = ""
            for m in reversed(history):
                if m.get("role") == "user":
                    content = m["content"]
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                            elif isinstance(part, str):
                                parts.append(part)
                        last_user_msg = " ".join(parts)
                    else:
                        last_user_msg = str(content)
                    break

            history = list(history) + [{"role": "assistant", "content": ""}]
            captured_trace = ""

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{BACKEND_URL}/tramites/stream",
                        json={"query": last_user_msg, "session_id": sid, "user_id": (user_id or "").strip() or None},
                        headers=_auth_headers(api_key),
                    ) as response:
                        if response.status_code in (401, 403):
                            history[-1] = {"role": "assistant", "content": "Error de autenticación: API key inválida"}
                            yield history, captured_trace
                            return
                        buffer = ""
                        tramites_info = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    event_type = data.get("type", "")

                                    if event_type == "trace_id":
                                        captured_trace = data.get("trace_id", "")
                                        yield history, captured_trace

                                    elif event_type == "retrieve":
                                        buffer += f"🔍 {data.get('content', '')}\n\n"
                                        history[-1] = {"role": "assistant", "content": buffer}
                                        yield history, captured_trace

                                    elif event_type == "token":
                                        token = data.get("content", "")
                                        buffer += token
                                        history[-1] = {"role": "assistant", "content": buffer + tramites_info}
                                        yield history, captured_trace

                                    elif event_type == "tramites":
                                        tramites = data.get("content", [])
                                        if tramites:
                                            tramites_info = "\n\n---\n**Trámites encontrados:**\n"
                                            for t in tramites:
                                                nombre = t.get("nombre", "")
                                                municipio = t.get("municipio", "")
                                                url = t.get("url", "")
                                                auth = t.get("auth", "")
                                                tramites_info += f"\n- **{nombre}** ({municipio})"
                                                if url:
                                                    tramites_info += f" — [Enlace]({url})"
                                                if auth:
                                                    tramites_info += f" | Auth: {auth}"
                                            history[-1] = {"role": "assistant", "content": buffer + tramites_info}
                                            yield history, captured_trace

                                    elif event_type == "done":
                                        break

                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                history[-1] = {"role": "assistant", "content": f"Error de conexión con el backend: {str(e)}"}
                yield history, captured_trace

        def clear_conversation():
            """Limpia el chat y genera un nuevo session_id."""
            return [], str(uuid.uuid4()), ""

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        clear_btn.click(clear_conversation, outputs=[chatbot, session_state, trace_id_state])

# =============================================================================
# PESTAÑA Comparador — Comparación dual de LLMs
# =============================================================================

def create_tab_comparador(api_key_state, user_id_state):
    """Crea la pestaña del comparador dual de LLMs sobre RAG trámites."""
    with gr.Tab("Comparador", id="tab-comparador"):
        gr.Markdown("### Comparador dual de LLMs")
        gr.Markdown(
            "Envía la misma consulta de trámites a dos LLMs locales en paralelo "
            "y compara las respuestas lado a lado."
        )

        trace_id_a_state = gr.State("")
        trace_id_b_state = gr.State("")
        # Comparador es stateless, pero session_id sirve para trazar en MLflow
        # (sin él, todos los usuarios concurrentes se mezclarían en un único
        # trace "default"). Se inicializa por sesión vía demo.load() global.
        session_state = gr.State("")
        _SESSION_STATES_TO_INIT.append(session_state)

        query_input = gr.Textbox(
            placeholder="Ej: ¿Qué trámites hay en Jaén?",
            show_label=False,
        )
        compare_btn = gr.Button("Comparar", variant="primary")

        with gr.Row():
            model_a_label = gr.Markdown("**Modelo A**")
            model_b_label = gr.Markdown("**Modelo B**")

        with gr.Row():
            response_a = gr.Textbox(
                label="Respuesta Modelo A",
                lines=15,
                interactive=False,
            )
            response_b = gr.Textbox(
                label="Respuesta Modelo B",
                lines=15,
                interactive=False,
            )

        with gr.Row():
            metrics_a = gr.Markdown("")
            metrics_b = gr.Markdown("")

        status_text = gr.Markdown("")

        if FEEDBACK_ENABLED:
            with gr.Group(elem_classes=["uja-feedback-card"]):
                gr.HTML('<div class="uja-feedback-heading">⚖️ ¿Qué respuesta prefieres?</div>')
                preference_radio = gr.Radio(
                    choices=[("A es mejor", "A"), ("B es mejor", "B"), ("Empate", "tie")],
                    show_label=False,
                )
                preference_rationale = gr.Textbox(
                    placeholder="Comentario opcional sobre la elección…",
                    show_label=False,
                    lines=2,
                )
                preference_send = gr.Button("Enviar valoración", variant="primary", size="sm")
                preference_status = gr.Markdown("", elem_classes=["uja-feedback-status"])

        async def run_comparison(query, api_key, user_id, sid):
            """Consume el SSE del comparador y actualiza ambas columnas."""
            if not query or not query.strip():
                yield (
                    "**Modelo A**", "**Modelo B**",
                    "", "",
                    "", "",
                    "Escribe una consulta.",
                    "", "",
                )
                return

            if not api_key:
                yield (
                    "**Modelo A**", "**Modelo B**",
                    "", "",
                    "", "",
                    "Configura la API Key en la pestaña Parámetros.",
                    "", "",
                )
                return

            buf_a = ""
            buf_b = ""
            label_a = "**Modelo A**"
            label_b = "**Modelo B**"
            met_a = ""
            met_b = ""
            status = ""
            trace_a = ""
            trace_b = ""

            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream(
                        "POST",
                        f"{BACKEND_URL}/comparador/stream",
                        json={"query": query.strip(), "user_id": (user_id or "").strip() or None, "session_id": sid},
                        headers=_auth_headers(api_key),
                    ) as response:
                        if response.status_code in (401, 403):
                            yield (
                                label_a, label_b,
                                "", "",
                                "", "",
                                "Error de autenticación: API key inválida.",
                                "", "",
                            )
                            return

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue

                            evt_type = data.get("type", "")

                            if evt_type == "trace_id":
                                tid = data.get("trace_id", "")
                                if data.get("llm") == "A":
                                    trace_a = tid
                                elif data.get("llm") == "B":
                                    trace_b = tid
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "models":
                                label_a = f"**{data.get('model_a', 'A')}**"
                                label_b = f"**{data.get('model_b', 'B')}**"
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "retrieve":
                                status = data.get("content", "")
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "token":
                                llm = data.get("llm", "")
                                token = data.get("content", "")
                                if llm == "A":
                                    buf_a += token
                                else:
                                    buf_b += token
                                status = "Generando respuestas..."
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "metrics":
                                llm = data.get("llm", "")
                                model = data.get("model", "")
                                time_ms = data.get("time_ms", 0)
                                tokens = data.get("tokens_output", 0)
                                metric_text = (
                                    f"Modelo: `{model}` | "
                                    f"Tiempo: **{time_ms:.0f} ms** | "
                                    f"Tokens: **{tokens}**"
                                )
                                if llm == "A":
                                    met_a = metric_text
                                else:
                                    met_b = metric_text
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "error":
                                status = data.get("content", "Error desconocido")
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

                            elif evt_type == "done":
                                status = "Comparación completada."
                                yield (
                                    label_a, label_b,
                                    buf_a, buf_b,
                                    met_a, met_b,
                                    status,
                                    trace_a, trace_b,
                                )

            except Exception as e:
                yield (
                    label_a, label_b,
                    buf_a, buf_b,
                    met_a, met_b,
                    f"Error de conexión: {e}",
                    trace_a, trace_b,
                )

        async def submit_preference(choice, rationale, trace_a, trace_b, api_key, user_id):
            """Envía la preferencia A/B/tie como un único feedback al backend."""
            if not choice:
                return "Selecciona A es mejor, B es mejor o Empate antes de enviar."
            if not (trace_a or trace_b):
                return "Aún no hay trace_id (lanza una comparación primero)."
            if not api_key:
                return "Configura la API Key en la pestaña Parámetros."
            if not (user_id or "").strip():
                return "Configura tu identificador de usuario en la pestaña Parámetros."

            payload = {
                # Asociamos el feedback al trace de A; si la arquitectura evoluciona
                # a dos traces independientes, basta con cambiar este valor.
                "trace_id": trace_a or trace_b,
                "value": choice,
                "source_id": user_id.strip(),
                "name": "comparador_preference",
                "rationale": (rationale or "").strip() or None,
                "metadata": {
                    "trace_id_a": trace_a or "",
                    "trace_id_b": trace_b or "",
                },
            }
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{BACKEND_URL}/feedback",
                        json=payload,
                        headers=_auth_headers(api_key),
                    )
                if resp.status_code == 200:
                    return "✓ Gracias por tu valoración."
                if resp.status_code == 202:
                    detail = resp.json().get("detail", "MLflow no disponible")
                    return f"⚠ Encolada: {detail[:120]}"
                return f"✗ Error {resp.status_code}: {resp.text[:120]}"
            except Exception as exc:
                return f"✗ Error de red: {exc}"

        compare_btn.click(
            run_comparison,
            inputs=[query_input, api_key_state, user_id_state, session_state],
            outputs=[
                model_a_label, model_b_label,
                response_a, response_b,
                metrics_a, metrics_b,
                status_text,
                trace_id_a_state, trace_id_b_state,
            ],
        )
        query_input.submit(
            run_comparison,
            inputs=[query_input, api_key_state, user_id_state, session_state],
            outputs=[
                model_a_label, model_b_label,
                response_a, response_b,
                metrics_a, metrics_b,
                status_text,
                trace_id_a_state, trace_id_b_state,
            ],
        )
        if FEEDBACK_ENABLED:
            preference_send.click(
                submit_preference,
                inputs=[
                    preference_radio,
                    preference_rationale,
                    trace_id_a_state,
                    trace_id_b_state,
                    api_key_state,
                    user_id_state,
                ],
                outputs=preference_status,
            )

# =============================================================================
# PESTAÑA Tracking — Tracking de rendimiento
# =============================================================================

_MLFLOW_EXTERNAL_URL = os.getenv(
    "MLFLOW_EXTERNAL_URL",
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
)

def create_tab_mlflow(api_key_state, user_id_state):
    """Crea la pestaña de tracking MLflow."""
    with gr.Tab("Tracking", id="tab-mlflow"):
        gr.Markdown("### Tracking MLflow")
        gr.Markdown(
            "Consulta el rendimiento historico de los servicios LLM. "
            "Las metricas se obtienen del servidor MLflow Tracking."
        )
        gr.Markdown(
            f"**[Abrir dashboard MLflow completo]({_MLFLOW_EXTERNAL_URL})**"
        )

        with gr.Row():
            exp_dropdown = gr.Dropdown(
                choices=["Todos"],
                value="Todos",
                label="Experimento / servicio",
                interactive=True,
            )
            _default_date = datetime.date.today().isoformat()
            date_from = gr.Textbox(label="Fecha desde (YYYY-MM-DD)", value=_default_date)
            date_to = gr.Textbox(label="Fecha hasta (YYYY-MM-DD)", value=_default_date)
            query_btn = gr.Button("Consultar", variant="primary")

        # ── Seccion: Resumen agregado ──
        gr.Markdown("#### Metricas agregadas por experimento")
        summary_table = gr.Dataframe(
            headers=[
                "Experimento", "Runs", "Tiempo total (ms)",
                "Tiempo LLM (ms)", "Tiempo retrieval (ms)",
                "Tiempo reranking (ms)", "Tokens entrada", "Tokens salida",
            ],
            datatype=["str", "number", "number", "number", "number", "number", "number", "number"],
            interactive=False,
        )

        # ── Seccion: Traces recientes ──
        gr.Markdown("#### Traces recientes")
        gr.Markdown(
            "_Cada trace captura el flujo completo de una consulta con sus spans "
            "(retrieval, reranking, LLM). Selecciona un trace para ver sus spans._"
        )
        traces_table = gr.Dataframe(
            headers=[
                "Trace ID", "Timestamp", "Servicio", "Duracion (ms)",
                "Estado", "Spans", "Query",
            ],
            datatype=["str", "str", "str", "number", "str", "number", "str"],
            interactive=False,
        )
        trace_detail = gr.JSON(label="Detalle del trace (spans)")

        # ── Seccion: Anotacion humana de ground truth ──
        selected_trace_id_state = gr.State("")
        if FEEDBACK_ENABLED:
            with gr.Group(elem_classes=["uja-feedback-card"]):
                gr.HTML(
                    '<div class="uja-feedback-heading">'
                    '📝 Anotar ground truth para el trace seleccionado'
                    '</div>'
                )
                expected_response_input = gr.Textbox(
                    label="Expected response",
                    placeholder="Escribe la respuesta de referencia (ground truth) para esta consulta…",
                    lines=5,
                )
                with gr.Row():
                    save_expectation_btn = gr.Button(
                        "Guardar como ground truth", variant="primary", size="sm"
                    )
                    expectation_status = gr.Markdown("", elem_classes=["uja-feedback-status"])

        # ── Seccion: Runs recientes ──
        gr.Markdown("#### Runs recientes")
        gr.Markdown(
            "_Cada run registra parametros, metricas y artefactos de una inferencia. "
            "Selecciona un run para ver su detalle completo._"
        )
        runs_table = gr.Dataframe(
            headers=[
                "Run ID", "Timestamp", "Modelo", "Tab", "Municipio",
                "Tiempo total (ms)", "Tiempo LLM (ms)", "Tokens entrada", "Tokens salida",
            ],
            datatype=["str", "str", "str", "str", "str", "number", "number", "number", "number"],
            interactive=False,
        )
        run_detail = gr.JSON(label="Detalle del run")

        # ── Helpers ──
        def _ts_to_str(ts_ms):
            if not ts_ms:
                return ""
            return datetime.datetime.fromtimestamp(
                ts_ms / 1000, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")

        def _build_params(experiment, d_from, d_to):
            params = {}
            if experiment and experiment != "Todos":
                params["experiment_name"] = experiment
            if d_from:
                params["date_from"] = d_from
            if d_to:
                params["date_to"] = d_to
            return params

        # ── Callbacks ──
        async def fetch_mlflow_summary(experiment, d_from, d_to, api_key):
            if not api_key:
                return [], gr.update()
            params = _build_params(experiment, d_from, d_to)
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/mlflow/summary",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code != 200:
                        return [], gr.update()
                    data = resp.json()

                    rows = []
                    for e in data.get("experiments", []):
                        rows.append([
                            e.get("experiment_name", ""),
                            e.get("run_count", 0),
                            e.get("avg_time_total_ms") or "",
                            e.get("avg_time_llm_ms") or "",
                            e.get("avg_time_retrieval_ms") or "",
                            e.get("avg_time_reranking_ms") or "",
                            e.get("avg_tokens_input") or "",
                            e.get("avg_tokens_output") or "",
                        ])

                    resp_exp = await client.get(
                        f"{BACKEND_URL}/mlflow/experiments",
                        headers=_auth_headers(api_key),
                    )
                    choices = ["Todos"]
                    if resp_exp.status_code == 200:
                        for exp in resp_exp.json():
                            choices.append(exp["experiment_name"])

                    return rows, gr.update(choices=choices)
            except Exception:
                return [], gr.update()

        async def fetch_mlflow_traces(experiment, d_from, d_to, api_key):
            if not api_key:
                return [], None
            params = _build_params(experiment, d_from, d_to)
            params["limit"] = 50
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/mlflow/traces",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code != 200:
                        return [], None
                    data = resp.json()

                    rows = []
                    for t in data:
                        rows.append([
                            (t.get("trace_id") or "")[:16],
                            _ts_to_str(t.get("timestamp")),
                            t.get("experiment_name") or "",
                            t.get("duration_ms") or "",
                            t.get("status") or "",
                            len(t.get("spans") or []),
                            t.get("request_preview") or "",
                        ])
                    return rows, None
            except Exception:
                return [], None

        async def select_trace(evt: gr.SelectData, experiment, d_from, d_to, api_key):
            if not api_key:
                return None, ""
            params = _build_params(experiment, d_from, d_to)
            params["limit"] = 50
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/mlflow/traces",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        if 0 <= idx < len(data):
                            trace = data[idx]
                            # Formatear detalle de spans
                            detail = {
                                "trace_id": trace.get("trace_id"),
                                "status": trace.get("status"),
                                "duration_ms": trace.get("duration_ms"),
                                "experiment": trace.get("experiment_name"),
                                "mlflow_url": f"{_MLFLOW_EXTERNAL_URL}/#/experiments",
                                "spans": [],
                            }
                            for s in trace.get("spans", []):
                                detail["spans"].append({
                                    "name": s.get("name"),
                                    "status": s.get("status"),
                                    "duration_ms": s.get("duration_ms"),
                                    "inputs": s.get("inputs"),
                                    "outputs": s.get("outputs"),
                                })
                            return detail, trace.get("trace_id") or ""
            except Exception:
                pass
            return None, ""

        async def submit_expectation(trace_id, expected_response, api_key, user_id):
            """Registra la respuesta esperada como ground truth en el trace seleccionado."""
            if not (trace_id or "").strip():
                return "Selecciona un trace de la tabla antes de anotar."
            if not (expected_response or "").strip():
                return "Escribe la respuesta esperada antes de guardar."
            if not api_key:
                return "Configura la API Key en la pestaña Parámetros."
            if not (user_id or "").strip():
                return "Configura tu identificador de usuario en la pestaña Parámetros."

            payload = {
                "trace_id": trace_id.strip(),
                "expected_response": expected_response.strip(),
                "source_id": user_id.strip(),
            }
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{BACKEND_URL}/feedback/expectation",
                        json=payload,
                        headers=_auth_headers(api_key),
                    )
                if resp.status_code == 200:
                    body = resp.json()
                    registered = ", ".join(body.get("registered", [])) or "expected_response"
                    return f"✓ Ground truth guardado ({registered})."
                if resp.status_code == 202:
                    detail = resp.json().get("detail", "MLflow no disponible")
                    return f"⚠ Encolado: {detail[:120]}"
                if resp.status_code in (401, 403):
                    return "✗ API key inválida."
                return f"✗ Error {resp.status_code}: {resp.text[:120]}"
            except Exception as exc:
                return f"✗ Error de red: {exc}"

        async def fetch_mlflow_runs(experiment, d_from, d_to, api_key):
            if not api_key:
                return [], None
            params = _build_params(experiment, d_from, d_to)
            params["limit"] = 50
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/mlflow/runs",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code != 200:
                        return [], None
                    data = resp.json()

                    rows = []
                    for r in data:
                        rows.append([
                            (r.get("run_id") or "")[:8],
                            _ts_to_str(r.get("timestamp")),
                            r.get("model") or "",
                            r.get("tab") or "",
                            r.get("municipio") or "",
                            r.get("time_total_ms") or "",
                            r.get("time_llm_ms") or "",
                            r.get("tokens_input") or "",
                            r.get("tokens_output") or "",
                        ])
                    return rows, None
            except Exception:
                return [], None

        async def select_run(evt: gr.SelectData, experiment, d_from, d_to, api_key):
            if not api_key:
                return None
            params = _build_params(experiment, d_from, d_to)
            params["limit"] = 50
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Obtener lista de runs para saber el run_id completo
                    resp = await client.get(
                        f"{BACKEND_URL}/mlflow/runs",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code != 200:
                        return None
                    runs = resp.json()
                    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    if idx < 0 or idx >= len(runs):
                        return None
                    run_id = runs[idx].get("run_id")
                    if not run_id:
                        return None

                    # Obtener detalle completo
                    resp2 = await client.get(
                        f"{BACKEND_URL}/mlflow/runs/{run_id}",
                        headers=_auth_headers(api_key),
                    )
                    if resp2.status_code == 200:
                        detail = resp2.json()
                        detail["mlflow_url"] = f"{_MLFLOW_EXTERNAL_URL}/#/experiments"
                        return detail
            except Exception:
                pass
            return None

        # ── Wiring ──
        query_btn.click(
            fetch_mlflow_summary,
            [exp_dropdown, date_from, date_to, api_key_state],
            [summary_table, exp_dropdown],
        )
        query_btn.click(
            fetch_mlflow_traces,
            [exp_dropdown, date_from, date_to, api_key_state],
            [traces_table, trace_detail],
        )
        query_btn.click(
            fetch_mlflow_runs,
            [exp_dropdown, date_from, date_to, api_key_state],
            [runs_table, run_detail],
        )
        exp_dropdown.change(
            fetch_mlflow_traces,
            [exp_dropdown, date_from, date_to, api_key_state],
            [traces_table, trace_detail],
        )
        exp_dropdown.change(
            fetch_mlflow_runs,
            [exp_dropdown, date_from, date_to, api_key_state],
            [runs_table, run_detail],
        )
        traces_table.select(
            select_trace,
            [exp_dropdown, date_from, date_to, api_key_state],
            [trace_detail, selected_trace_id_state],
        )
        if FEEDBACK_ENABLED:
            save_expectation_btn.click(
                submit_expectation,
                inputs=[
                    selected_trace_id_state,
                    expected_response_input,
                    api_key_state,
                    user_id_state,
                ],
                outputs=expectation_status,
            )
        runs_table.select(
            select_run,
            [exp_dropdown, date_from, date_to, api_key_state],
            [run_detail],
        )

# =============================================================================
# PESTAÑA Logs — Logs de inferencia
# =============================================================================

def create_tab_logs(api_key_state):
    """Crea la pestaña de logs de inferencia."""
    with gr.Tab("Logs", id="tab-logs"):
        gr.Markdown("### Logs de inferencia")
        gr.Markdown(
            "Consulta los registros de inferencia de todas las tabs. "
            "Los logs se almacenan en memoria y se pierden al reiniciar el backend."
        )

        with gr.Row():
            tab_filter = gr.Dropdown(
                choices=["Todas", "chatbot", "react-agent", "rag", "tramites", "comparador"],
                value="Todas",
                label="Filtrar por tab",
                interactive=True,
            )
            refresh_btn = gr.Button("Refrescar", variant="primary")
            clear_btn = gr.Button("Limpiar logs", variant="stop")

        logs_table = gr.Dataframe(
            headers=["Timestamp", "Tab", "Query", "Tiempo (ms)", "Tokens out", "Modelo", "Docs"],
            datatype=["str", "str", "str", "number", "number", "str", "str"],
            interactive=False,
        )

        gr.Markdown("#### Detalle de la entrada seleccionada")
        log_detail = gr.JSON(label="Detalle completo")

        async def fetch_logs(tab_value, api_key):
            """Obtiene los logs del backend."""
            if not api_key:
                return [], None

            params = {"limit": 100}
            if tab_value and tab_value != "Todas":
                params["tab"] = tab_value

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/logs",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code != 200:
                        return [], None
                    data = resp.json()
                    entries = data.get("entries", [])

                    # Tabla resumida
                    rows = []
                    for e in entries:
                        ts = e.get("timestamp", "")[:19].replace("T", " ")
                        query = (e.get("query", "") or "")[:60]
                        n_ret = len(e.get("retrieved_docs") or [])
                        n_rerank = len(e.get("reranked_docs") or [])
                        docs_str = f"{n_rerank}" if n_rerank else (f"{n_ret}" if n_ret else "-")
                        rows.append([
                            ts,
                            e.get("tab", ""),
                            query,
                            round(e.get("time_total_ms", 0), 0),
                            e.get("tokens_output", 0),
                            e.get("model", ""),
                            docs_str,
                        ])

                    return rows, entries[0] if entries else None

            except Exception:
                return [], None

        async def clear_logs_action(api_key):
            """Vacía los logs del backend."""
            if not api_key:
                return [], None
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.delete(
                        f"{BACKEND_URL}/logs",
                        headers=_auth_headers(api_key),
                    )
            except Exception:
                pass
            return [], None

        async def select_row(evt: gr.SelectData, tab_value, api_key):
            """Muestra el detalle de la entrada seleccionada."""
            if not api_key:
                return None

            params = {"limit": 100}
            if tab_value and tab_value != "Todas":
                params["tab"] = tab_value

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/logs",
                        params=params,
                        headers=_auth_headers(api_key),
                    )
                    if resp.status_code == 200:
                        entries = resp.json().get("entries", [])
                        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        if 0 <= idx < len(entries):
                            return entries[idx]
            except Exception:
                pass
            return None

        refresh_btn.click(
            fetch_logs, [tab_filter, api_key_state], [logs_table, log_detail]
        )
        tab_filter.change(
            fetch_logs, [tab_filter, api_key_state], [logs_table, log_detail]
        )
        clear_btn.click(
            clear_logs_action, [api_key_state], [logs_table, log_detail]
        )
        logs_table.select(
            select_row, [tab_filter, api_key_state], [log_detail]
        )

# =============================================================================
# PESTAÑA Parámetros
# =============================================================================

def create_tab_params(api_key_state, user_id_state):
    """Crea la pestaña de parámetros con configuración de API Key e identificador de usuario."""
    with gr.Tab("Parámetros"):
        gr.Markdown("### Configuración")
        gr.Markdown(
            "Introduce la **API Key** para autenticarte y un **identificador de usuario** "
            "que se asociará a tus valoraciones de feedback. "
            "Ambos valores se usan durante esta sesión y **no se almacenan** en disco."
        )

        api_key_input = gr.Textbox(
            label="API Key",
            placeholder="Introduce tu API Key...",
            interactive=True,
        )
        user_id_input = gr.Textbox(
            label="Identificador de usuario",
            placeholder="Ej: Francisco Pérez — sirve para firmar tus valoraciones",
            interactive=True,
        )
        save_btn = gr.Button("Guardar", variant="primary")
        status = gr.Markdown("")

        async def save_params(key, user_id):
            """Valida y guarda API key + identificador de usuario en el estado de sesión.

            Devuelve (api_key_state, user_id_state, status_md). Ambos campos son
            obligatorios para poder enviar feedback.
            """
            key_clean = (key or "").strip()
            uid_clean = (user_id or "").strip()

            if not key_clean and not uid_clean:
                return "", "", "Introduce la API Key y el identificador de usuario."
            if not key_clean:
                return "", uid_clean, "API Key no puede estar vacía."
            if not uid_clean:
                return key_clean, "", "El identificador de usuario es obligatorio para registrar feedback."

            # Validar la key contra un endpoint ligero (no invoca al LLM)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{BACKEND_URL}/auth/check",
                        headers=_auth_headers(key_clean),
                    )
                    if resp.status_code in (401, 403):
                        return "", uid_clean, "API Key inválida."
            except Exception:
                return key_clean, uid_clean, "No se pudo conectar al backend. Credenciales guardadas localmente."

            return (
                key_clean,
                uid_clean,
                f"Token registrado y firmando como **{uid_clean}**. Ya puedes trabajar con las pestañas.",
            )

        save_btn.click(
            save_params,
            inputs=[api_key_input, user_id_input],
            outputs=[api_key_state, user_id_state, status],
        )

# =============================================================================
# TEMA UJA (Universidad de Jaén)
# =============================================================================

# Logos codificados en base64 para uso inline (portátil local + Docker)
_LOGO_UJA = _logo_b64("logo-b-1.png")
_LOGO_ALIA_BLANCO = _logo_b64("alia-blanco.svg")
_LOGO_ALIA_COLOR = _logo_b64("alia.svg")
_LOGO_SINAI = _logo_b64("SINAI.png")
_LOGO_AGRADECIMIENTOS = _logo_b64("alia_logos_agradecimientos.png")

ujaen_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#e6f5ed", c100="#b3e0c9", c200="#80cca5",
        c300="#4db881", c400="#26a85e", c500="#006d38",
        c600="#005f31", c700="#004921", c800="#003a1a",
        c900="#002b13", c950="#001d0d",
    ),
    secondary_hue=gr.themes.Color(
        c50="#e6eef5", c100="#b3cde0", c200="#80abcc",
        c300="#4d8ab8", c400="#2673a8", c500="#014c8c",
        c600="#01437a", c700="#013460", c800="#012a4d",
        c900="#011f3a", c950="#011528",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f7f7f7", c100="#ececed", c200="#d2d2d4",
        c300="#b0aeb0", c400="#8e8c8e", c500="#636466",
        c600="#4c4345", c700="#3d3537", c800="#2e2829",
        c900="#1f1b1c", c950="#100e0e",
    ),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "Helvetica Neue", "Arial", "sans-serif"],
)

ujaen_css = """
.gradio-container {
    max-width: 1200px !important;
    padding-bottom: 80px !important;
}
/* ── Header ────────────────────────────────────────── */
#uja-header {
    background-color: #006d38;
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}
#uja-header .uja-header__left {
    display: flex;
    align-items: center;
    gap: 16px;
}
#uja-header .uja-header__left img {
    height: 50px;
}
#uja-header .uja-header__right {
    display: flex;
    align-items: center;
    gap: 16px;
}
#uja-header .uja-header__right img {
    height: 36px;
}
#uja-header h1 {
    color: white !important;
    margin: 0 !important;
    font-size: 1.5em !important;
}
#uja-header p {
    color: rgba(255,255,255,0.85) !important;
    margin: 4px 0 0 0 !important;
    font-size: 0.9em !important;
}
/* ── Footer sticky ─────────────────────────────────── */
#uja-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background-color: #f7f7f7;
    border-top: 1px solid #d2d2d4;
    padding: 8px 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
}
#uja-footer img {
    height: 32px;
    object-fit: contain;
}
#uja-footer .uja-footer__sinai img {
    height: 40px;
}
#uja-footer .uja-footer__text {
    color: #636466;
    font-size: 0.75em;
    text-align: center;
}
/* ── Tabs ──────────────────────────────────────────── */
.tab-nav button {
    font-weight: 600 !important;
}
.tab-nav button.selected {
    border-color: #006d38 !important;
    color: #006d38 !important;
}
/* Tabs destacadas (Trámites, Comparador, Logs) */
button#tab-tramites,
button#tab-comparador,
button#tab-mlflow,
button#tab-logs {
    font-weight: 900 !important;
}
"""

# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

with gr.Blocks(title="UJA — Asistente IA en Trámites Administrativos Públicos") as demo:
    gr.HTML(f"""
    <header id="uja-header">
        <div class="uja-header__left">
            <img src="{_LOGO_UJA}" alt="Universidad de Jaén" />
            <div>
                <h1>Asistente IA en Trámites Administrativos Públicos</h1>
                <p>Universidad de Jaén — Aplicación experimental con LangGraph, FastAPI y Gradio</p>
            </div>
        </div>
        <div class="uja-header__right">
            <img src="{_LOGO_ALIA_BLANCO}" alt="ALIA" />
            <img src="{_LOGO_SINAI}" alt="SINAI — Sistemas Inteligentes de Acceso a la Información" style="height: 44px;" />
        </div>
    </header>
    """)

    # Estado compartido — no se persiste, solo vive en la sesión.
    # `user_id_state` se usa como `source_id` en los assessments de feedback.
    api_key_state = gr.State("")
    user_id_state = gr.State("")

    create_tab1(api_key_state, user_id_state)
    create_tab2(api_key_state, user_id_state)
    create_tab2bis(api_key_state, user_id_state)
    create_tab_tramites(api_key_state, user_id_state)
    create_tab_comparador(api_key_state, user_id_state)
    create_tab_mlflow(api_key_state, user_id_state)
    create_tab_logs(api_key_state)
    create_tab_params(api_key_state, user_id_state)

    # Asignar un session_id único por navegador: en cada page-load se invoca
    # uuid4() una vez por cada State registrado. Sustituye al patrón
    # gr.State(value=lambda: ...) que en Gradio 6.x dejaba la app inoperante.
    if _SESSION_STATES_TO_INIT:
        demo.load(
            fn=lambda: [str(uuid.uuid4()) for _ in _SESSION_STATES_TO_INIT],
            outputs=list(_SESSION_STATES_TO_INIT),
        )

    # Footer sticky con logos de patrocinadores (normativa subvención)
    gr.HTML(f"""
    <footer id="uja-footer">
        <span class="uja-footer__sinai">
            <img src="{_LOGO_SINAI}" alt="SINAI" />
        </span>
        <img src="{_LOGO_ALIA_COLOR}" alt="ALIA" />
        <img src="{_LOGO_AGRADECIMIENTOS}" alt="Financiado por la Unión Europea — NextGenerationEU | Gobierno de España | Plan de Recuperación" />
        <span class="uja-footer__text">
            Proyecto financiado por la Unión Europea — NextGenerationEU
        </span>
    </footer>
    """)

if __name__ == "__main__":
    from starlette.requests import Request
    from starlette.responses import Response

    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    _MLFLOW_INTERNAL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    # Lanzar Gradio normalmente (preserva theme, CSS, header, footer, tabs)
    demo.launch(
        server_name="0.0.0.0", server_port=port,
        theme=ujaen_theme, css=ujaen_css,
        prevent_thread_lock=True,
    )

    # ── Reverse proxy para MLflow UI ──
    # Registrar ruta en el app FastAPI subyacente de Gradio DESPUES de launch.
    gradio_app = demo.server.config.app

    @gradio_app.api_route(
        "/mlflow-ui/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def mlflow_proxy(request: Request, path: str = ""):
        target = f"{_MLFLOW_INTERNAL}/mlflow-ui/{path}"
        qs = str(request.query_params)
        if qs:
            target += f"?{qs}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.request(
                    method=request.method,
                    url=target,
                    headers={
                        k: v for k, v in request.headers.items()
                        if k.lower() not in ("host", "transfer-encoding", "connection")
                    },
                    content=await request.body(),
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
                    },
                )
        except Exception:
            return Response(content=b"MLflow no disponible", status_code=502)

    # Bloquear el hilo principal
    demo.block_thread()
