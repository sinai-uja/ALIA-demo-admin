"""
Router FastAPI para el agente ReAct con RAG (Tab 2).

Endpoint: POST /react-agent/stream
Devuelve streaming SSE con eventos detallados del agente.
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.tab2_react_agent.graph import graph
from backend.inference_logger.middleware import InferenceCapture
from backend.mlflow_tracer import emit_trace_id_event

logger = logging.getLogger(__name__)

router = APIRouter()

class ReactAgentRequest(BaseModel):
    """Modelo de petición para el agente ReAct."""
    messages: list[dict]
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@router.post("/stream")
async def react_agent_stream(request: ReactAgentRequest):
    """
    Endpoint de streaming SSE para el agente ReAct.

    Devuelve eventos de tipo token, tool_call, tool_result y done.
    El historial completo lo gestiona el cliente y lo envía en cada petición;
    el backend es stateless para evitar mezcla de sesiones entre usuarios.
    """

    async def event_generator():
        try:
            session_id = (request.session_id or "").strip() or str(uuid.uuid4())

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
                tab="react-agent", session_id=session_id, query=user_query, user_id=request.user_id
            )
            capture.start_span("llm_generation", span_type="LLM", inputs={"query": user_query})

            yield emit_trace_id_event(capture.request_id)

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
                            "data": json.dumps({
                                "type": "token",
                                "content": token,
                                "done": False,
                            }),
                        }
                    if chunk and chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            if tc.get("name"):
                                yield {
                                    "event": "message",
                                    "data": json.dumps({
                                        "type": "tool_call",
                                        "tool": tc["name"],
                                        "input": json.dumps(tc.get("args", {})),
                                    }),
                                }

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    if hasattr(output, "content"):
                        output = output.content
                    capture.set_context_preview(str(output))
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_result",
                            "content": str(output)[:500],
                        }),
                    }

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
                "data": json.dumps({"type": "done"}),
            }

        except Exception as e:
            logger.error(f"Error en react-agent stream: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "token",
                    "content": f"Error: {str(e)}",
                    "done": True,
                }),
            }

    return EventSourceResponse(event_generator())
