"""
Widget reutilizable de feedback humano (thumbs up/down + rationale opcional).

Patron de uso desde una pestana de chat:

    from components import render_feedback_widget

    trace_id_state = gr.State("")
    feedback = render_feedback_widget(api_key_state, trace_id_state)
    # `trace_id_state` debe actualizarse en el handler de streaming en cuanto
    # llega el primer evento SSE `{"type": "trace_id", "trace_id": ...}`.

El widget gestiona internamente la seleccion 👍/👎 con un `gr.State` propio.
La llamada al backend usa `BACKEND_URL` y la API key ya configurada en la app.
"""

import os
from typing import Optional, Tuple

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

_WIDGET_CSS = """
<style>
.uja-feedback-card {
    border: 2px solid #006d38;
    border-radius: 10px;
    padding: 14px 18px 12px 18px;
    margin: 12px 0 8px 0;
    background: linear-gradient(180deg, rgba(0,109,56,0.06) 0%, rgba(0,109,56,0.02) 100%);
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.uja-feedback-card .uja-feedback-heading {
    font-weight: 600;
    color: #006d38;
    font-size: 1.02rem;
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.uja-feedback-card .uja-feedback-thumbs button {
    font-size: 1.45rem !important;
    padding: 8px 14px !important;
    min-width: 72px !important;
}
.uja-feedback-card .uja-feedback-status {
    font-size: 0.92rem;
    color: #014c8c;
    margin-left: 4px;
}
.uja-feedback-card textarea { min-height: 52px !important; }
</style>
"""

_HEADING_HTML = (
    '<div class="uja-feedback-heading">'
    '💬 ¿Te ha resultado útil esta respuesta?'
    '</div>'
)

def render_feedback_widget(
    api_key_state: gr.State,
    trace_id_state: gr.State,
    user_id_state: gr.State,
    name: str = "user_thumbs",
) -> Tuple[gr.Button, gr.Button, gr.Textbox, gr.Button, gr.Markdown]:
    """Renderiza la UI de feedback bajo una respuesta del LLM.

    Args:
        api_key_state: gr.State con la API key del usuario.
        trace_id_state: gr.State con el `trace_id` de la respuesta a valorar.
        user_id_state: gr.State con el identificador de usuario (source_id en MLflow).
        name: nombre del assessment en MLflow (default `"user_thumbs"`).

    Returns:
        Tupla (thumbs_up_btn, thumbs_down_btn, rationale_textbox, send_btn, status_md).
    """
    selected_value = gr.State(None)

    gr.HTML(_WIDGET_CSS)
    with gr.Group(elem_classes=["uja-feedback-card"]):
        gr.HTML(_HEADING_HTML)
        with gr.Row(elem_classes=["uja-feedback-thumbs"]):
            thumbs_up_btn = gr.Button("👍 Sí", scale=0, min_width=88, variant="secondary")
            thumbs_down_btn = gr.Button("👎 No", scale=0, min_width=88, variant="secondary")
            status_md = gr.Markdown("", elem_classes=["uja-feedback-status"])
        rationale_textbox = gr.Textbox(
            placeholder="Comentario opcional sobre la respuesta…",
            show_label=False,
            lines=2,
        )
        send_btn = gr.Button("Enviar valoración", variant="primary", size="sm")

    def _select(value: bool) -> Tuple[bool, str]:
        label = "👍 positivo" if value else "👎 negativo"
        return value, f"Seleccionado: **{label}**. Pulsa _Enviar valoración_."

    thumbs_up_btn.click(
        lambda: _select(True), outputs=[selected_value, status_md]
    )
    thumbs_down_btn.click(
        lambda: _select(False), outputs=[selected_value, status_md]
    )

    async def submit_feedback(
        value: Optional[bool],
        rationale: Optional[str],
        trace_id: Optional[str],
        api_key: Optional[str],
        user_id: Optional[str],
    ) -> Tuple[str, str]:
        """Handler del boton Enviar. Devuelve (status_md, rationale_reset)."""
        if value is None:
            return "Selecciona 👍 o 👎 antes de enviar.", rationale or ""
        if not trace_id:
            return "Aun no hay trace_id (espera al inicio de la respuesta).", rationale or ""
        if not api_key:
            return "Configura la API Key en la pestaña Parámetros.", rationale or ""
        if not (user_id or "").strip():
            return "Configura tu identificador de usuario en la pestaña Parámetros.", rationale or ""

        payload = {
            "trace_id": trace_id,
            "value": value,
            "source_id": user_id.strip(),
            "name": name,
            "rationale": (rationale or "").strip() or None,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{BACKEND_URL}/feedback",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 200:
                return "✓ Gracias por tu valoración.", ""
            if resp.status_code == 202:
                detail = resp.json().get("detail", "MLflow no disponible")
                return f"⚠ Valoración encolada (MLflow no disponible): {detail[:120]}", rationale or ""
            if resp.status_code in (401, 403):
                return "✗ API key inválida.", rationale or ""
            return f"✗ Error {resp.status_code}: {resp.text[:120]}", rationale or ""
        except Exception as exc:
            return f"✗ Error de red: {exc}", rationale or ""

    send_btn.click(
        submit_feedback,
        inputs=[selected_value, rationale_textbox, trace_id_state, api_key_state, user_id_state],
        outputs=[status_md, rationale_textbox],
    )

    return thumbs_up_btn, thumbs_down_btn, rationale_textbox, send_btn, status_md
