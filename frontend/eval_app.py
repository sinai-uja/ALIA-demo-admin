"""
Frontend Gradio dedicado a la evaluación pública del RAG de Trámites.

Es un entrypoint paralelo a `frontend/app.py` con una sola tab interna
(Trámites) y un bloque de identidad de evaluador en lugar de la pestaña
Parámetros completa. Pensado para servirse en una URL distinta (puerto
host 7863 en Docker, `GRADIO_EVAL_SERVER_PORT` por defecto 7861 en local)
y para que los thumbs/rationale generados aquí queden firmados en MLflow
con el alias del evaluador.

Decisión de diseño: la lógica de la tab Trámites, el tema UJA y los logos
están **duplicados literalmente** desde `frontend/app.py` para garantizar
cero acoplamiento entre el entrypoint de evaluación pública y el principal.
"""

import base64
import json
import mimetypes
import os
import sys as _sys
import uuid

import gradio as gr
import httpx
from dotenv import load_dotenv

_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _FRONTEND_DIR not in _sys.path:
    _sys.path.insert(0, _FRONTEND_DIR)

from components import render_feedback_widget  # noqa: E402

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
FEEDBACK_ENABLED = os.getenv("FEEDBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

_LOGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "diseno", "Logos")

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
# Handlers (duplicados de frontend/app.py:385 — decisión 5)
# =============================================================================
# Los handlers se exponen a nivel de módulo en lugar de cerrar dentro de
# `create_eval_tramites_tab` para que los tests puedan ejercitarlos sin
# montar Gradio.

async def user_message(message, history):
    if not message.strip():
        return "", history
    history = history + [{"role": "user", "content": message}]
    return "", history

async def bot_response(history, api_key, sid, user_id):
    if not history:
        yield history, ""
        return

    if not api_key:
        history = list(history) + [{"role": "assistant", "content": "Identifícate como evaluador (API Key + alias) en el bloque superior."}]
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
    return [], str(uuid.uuid4()), ""

async def save_evaluator_identity(key, user_id):
    """Valida y guarda API key + alias del evaluador."""
    key_clean = (key or "").strip()
    uid_clean = (user_id or "").strip()

    if not key_clean and not uid_clean:
        return "", "", "Introduce la API Key y el alias del evaluador."
    if not key_clean:
        return "", uid_clean, "API Key no puede estar vacía."
    if not uid_clean:
        return key_clean, "", "El alias del evaluador es obligatorio para registrar feedback."

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
        f"Token registrado y firmando como **{uid_clean}**. Ya puedes consultar la tab Trámites.",
    )

def create_eval_tramites_tab(api_key_state, user_id_state):
    """Crea la pestaña del asistente de trámites municipales para la URL de evaluación."""
    with gr.Tab("Trámites", id="tab-tramites"):
        gr.Markdown("### Trámites Municipales de Andalucía")
        gr.Markdown(
            "Consulta trámites administrativos municipales. "
            "Puedes especificar el municipio para resultados más precisos."
        )
        session_state = gr.State(str(uuid.uuid4()))
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

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot, api_key_state, session_state, user_id_state], [chatbot, trace_id_state]
        )
        clear_btn.click(clear_conversation, outputs=[chatbot, session_state, trace_id_state])

# =============================================================================
# Bloque de identificación del evaluador (replica `create_tab_params`)
# =============================================================================

def render_evaluator_identity(api_key_state, user_id_state):
    """Accordion superior con API Key + alias del evaluador.

    Replica la lógica de `frontend/app.py:create_tab_params` pero como
    `gr.Accordion` siempre visible en lugar de pestaña separada.
    """
    with gr.Accordion("Identifícate como evaluador", open=True):
        gr.Markdown(
            "Introduce la **API Key** facilitada por el equipo y un **alias** "
            "(p. ej. tu nombre o el código que te asignaron). "
            "Ambos valores se usan solo durante esta sesión y **no se almacenan**. "
            "El alias firmará tus thumbs/rationale en MLflow."
        )

        api_key_input = gr.Textbox(
            label="API Key",
            placeholder="Introduce tu API Key...",
            interactive=True,
        )
        user_id_input = gr.Textbox(
            label="Alias del evaluador",
            placeholder="Ej: Francisco Pérez — sirve para firmar tus valoraciones",
            interactive=True,
        )
        save_btn = gr.Button("Guardar", variant="primary")
        status = gr.Markdown("")

        save_btn.click(
            save_evaluator_identity,
            inputs=[api_key_input, user_id_input],
            outputs=[api_key_state, user_id_state, status],
        )

# =============================================================================
# TEMA UJA (duplicado literal de frontend/app.py:1397+ — decisión 5)
# =============================================================================

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
/* ── Disclaimer evaluación ─────────────────────────── */
#uja-eval-disclaimer {
    background-color: #fff8e1;
    border-left: 4px solid #f5a623;
    padding: 10px 16px;
    margin-bottom: 16px;
    border-radius: 4px;
    color: #4c4345;
    font-size: 0.95em;
}
#uja-eval-disclaimer a {
    color: #014c8c;
    font-weight: 600;
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
button#tab-tramites {
    font-weight: 900 !important;
}
"""

# =============================================================================
# APLICACIÓN — Evaluación pública de Trámites
# =============================================================================

with gr.Blocks(title="Evaluación pública — UJAenAgent / Trámites") as demo:
    gr.HTML(f"""
    <header id="uja-header">
        <div class="uja-header__left">
            <img src="{_LOGO_UJA}" alt="Universidad de Jaén" />
            <div>
                <h1>Evaluación pública — Asistente IA en Trámites</h1>
                <p>Universidad de Jaén — URL dedicada a evaluadores externos</p>
            </div>
        </div>
        <div class="uja-header__right">
            <img src="{_LOGO_ALIA_BLANCO}" alt="ALIA" />
            <img src="{_LOGO_SINAI}" alt="SINAI — Sistemas Inteligentes de Acceso a la Información" style="height: 44px;" />
        </div>
    </header>
    """)

    gr.HTML("""
    <div id="uja-eval-disclaimer">
        Estás participando en la <strong>evaluación pública</strong> del asistente de Trámites.
        Tus thumbs y comentarios se registran en MLflow firmados con el alias que indiques abajo.
    </div>
    """)

    api_key_state = gr.State("")
    user_id_state = gr.State("")

    render_evaluator_identity(api_key_state, user_id_state)
    create_eval_tramites_tab(api_key_state, user_id_state)

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
    port = int(os.getenv("GRADIO_EVAL_SERVER_PORT", "7861"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=ujaen_theme,
        css=ujaen_css,
    )
