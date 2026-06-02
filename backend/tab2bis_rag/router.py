"""
Router FastAPI para RAG siempre activo (Tab 2bis).

Endpoint: POST /rag/stream
Devuelve streaming SSE con eventos de recuperación y generación.
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.tab2bis_rag.graph import graph
from backend.inference_logger.middleware import InferenceCapture
from backend.mlflow_tracer import emit_trace_id_event
from backend.prompt_registry.provider import begin_prompt_tracking, get_prompts_used

logger = logging.getLogger(__name__)

router = APIRouter()

class RAGRequest(BaseModel):
    """Modelo de petición para el RAG."""
    messages: list[dict]
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@router.post("/stream")
async def rag_stream(request: RAGRequest):
    """
    Endpoint de streaming SSE para el RAG.

    Devuelve eventos de tipo retrieve, token y done.
    """

    async def event_generator():
        # tracking de prompts usados para tag MLflow al cerrar.
        begin_prompt_tracking()
        try:
            user_messages = [m for m in request.messages if m.get("role") == "user"]
            if not user_messages:
                return
            last_user_msg = user_messages[-1]

            user_query = last_user_msg.get("content", "")
            if isinstance(user_query, list):
                user_query = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in user_query
                )

            session_id = (request.session_id or "").strip() or str(uuid.uuid4())
            capture = InferenceCapture(
                tab="rag", session_id=session_id, query=user_query, user_id=request.user_id
            )

            yield emit_trace_id_event(capture.request_id)

            collected_content = ""
            retrieve_sent = False

            async for event in graph.astream_events(
                {"messages": [last_user_msg], "context": ""},
                version="v2",
            ):
                kind = event.get("event")
                name = event.get("name", "")

                if kind == "on_chain_start" and name == "retrieve" and not retrieve_sent:
                    retrieve_sent = True
                    capture.mark_retrieval_start()
                    capture.start_span("retrieval", span_type="RETRIEVER", inputs={"query": user_query})
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "retrieve",
                            "content": "Buscando en la base de conocimiento...",
                        }),
                    }

                if kind == "on_chain_end" and name == "retrieve":
                    capture.mark_retrieval_end()
                    output = event["data"].get("output", {})
                    context = output.get("context", "") if isinstance(output, dict) else ""
                    if context:
                        capture.set_context_preview(context)
                        doc_chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
                        capture.set_retrieved_docs([
                            {"content": chunk[:500], "score": 0.0, "metadata": {}}
                            for chunk in doc_chunks
                        ])
                        capture.end_span(outputs={"n_docs": len(doc_chunks), "context": context[:500]})
                        capture.start_span("llm_generation", span_type="LLM", inputs={"query": user_query, "context_length": len(context)})
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "type": "context",
                                "content": context[:500],
                            }),
                        }

                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        token = chunk.content
                        collected_content += token
                        capture.add_token(token)
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "type": "token",
                                "content": token,
                                "done": False,
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

            # tags `prompt.<name>.version|source` antes de cerrar.
            capture.set_prompt_tags(get_prompts_used())

            capture.finalize()

            yield {
                "event": "message",
                "data": json.dumps({"type": "done"}),
            }

        except Exception as e:
            logger.error(f"Error en rag stream: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "token",
                    "content": f"Error: {str(e)}",
                    "done": True,
                }),
            }

    return EventSourceResponse(event_generator())
