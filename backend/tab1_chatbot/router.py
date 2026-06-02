"""
Router FastAPI para el chatbot simple (Tab 1).

Endpoint: POST /chatbot/stream
Devuelve streaming SSE token a token.
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.tab1_chatbot.graph import graph
from backend.inference_logger.middleware import InferenceCapture
from backend.mlflow_tracer import emit_trace_id_event

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    """Modelo de petición para el chatbot."""
    messages: list[dict]
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@router.post("/stream")
async def chatbot_stream(request: ChatRequest):
    """
    Endpoint de streaming SSE para el chatbot.

    Recibe mensajes y devuelve la respuesta del LLM token a token.
    El historial completo lo gestiona el cliente y lo envía en cada petición;
    el backend es stateless para evitar mezcla de sesiones entre usuarios.
    """

    async def event_generator():
        try:
            # Si el cliente no manda session_id (o llega vacío), generamos uno
            # en el servidor para garantizar aislamiento en trazas/logs.
            session_id = (request.session_id or "").strip() or str(uuid.uuid4())

            # Extraer la query del usuario para el log
            user_query = ""
            for m in reversed(request.messages):
                if m.get("role") == "user":
                    user_query = m.get("content", "")
                    if isinstance(user_query, list):
                        user_query = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in user_query
                        )
                    break

            capture = InferenceCapture(
                tab="chatbot", session_id=session_id, query=user_query, user_id=request.user_id
            )
            capture.start_span("llm_generation", span_type="LLM", inputs={"query": user_query})

            yield emit_trace_id_event(capture.request_id)

            # Streaming del grafo. El historial llega íntegro en request.messages.
            async for event in graph.astream_events(
                {"messages": request.messages},
                version="v2",
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        token = chunk.content
                        capture.add_token(token)
                        yield {
                            "event": "message",
                            "data": json.dumps({"token": token, "done": False}),
                        }
                # Capturar modelo y tokens del evento final del LLM
                if kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    if output and hasattr(output, "usage_metadata") and output.usage_metadata:
                        capture.set_tokens(
                            input=output.usage_metadata.get("input_tokens"),
                            output=output.usage_metadata.get("output_tokens"),
                        )
                    if output and hasattr(output, "response_metadata"):
                        capture.set_model(output.response_metadata.get("model_name", ""))

            capture.finalize()

            yield {
                "event": "message",
                "data": json.dumps({"token": "", "done": True}),
            }

        except Exception as e:
            logger.error(f"Error en chatbot stream: {e}")
            yield {
                "event": "message",
                "data": json.dumps({"token": f"Error: {str(e)}", "done": True}),
            }

    return EventSourceResponse(event_generator())
