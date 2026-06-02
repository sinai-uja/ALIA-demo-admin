"""
Módulo rag_tramites — Router FastAPI para trámites municipales.

Endpoints:
  POST /tramites/stream  — Chat SSE con streaming
  GET  /tramites/health  — Health check de ChromaDB y modelos
"""

import json
import logging
import uuid

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.rag_tramites.api.schemas import TramitesRequest
from backend.config import settings
from backend.rag_tramites.graph.tramites_graph import graph
from backend.inference_logger.middleware import InferenceCapture
from backend.mlflow_tracer import emit_trace_id_event
from backend.prompt_registry.provider import begin_prompt_tracking, get_prompts_used

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/stream")
async def tramites_stream(request: TramitesRequest):
    """Endpoint de streaming SSE para consultas de trámites."""

    async def event_generator():
        # arma la lista de PromptResource consumidos durante la
        # request para que se adjunten como tags a la traza MLflow antes de
        # finalize(). El ContextVar es async-safe y aisla por task.
        begin_prompt_tracking()
        try:
            # Aislamiento: si el cliente no manda session_id, se genera en
            # servidor para que LangGraph/MemorySaver no mezcle checkpoints
            # entre usuarios concurrentes.
            session_id = (request.session_id or "").strip() or str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}

            input_state = {
                "session_id": session_id,
                "query_actual": request.query,
                "messages": [{"role": "user", "content": request.query}],
                "municipio_detectado": None,
                "intent": "especifico",
                "candidatos": [],
                "candidatos_reranked": [],
                "respuesta_final": "",
                "necesita_refinamiento": False,
                "turno": 0,
            }

            capture = InferenceCapture(
                tab="tramites",
                session_id=session_id,
                query=request.query,
                user_id=request.user_id,
            )

            yield emit_trace_id_event(capture.request_id)

            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "retrieve",
                    "content": "Buscando trámites relevantes...",
                }),
            }

            capture.mark_retrieval_start()
            capture.start_span("retrieval", span_type="RETRIEVER", inputs={"query": request.query})
            collected_content = ""

            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")

                if kind == "on_chain_start" and name == "rerank":
                    capture.mark_retrieval_end()
                    capture.end_span(outputs={"status": "retrieval_done"})
                    capture.mark_reranking_start()
                    capture.start_span("reranking", span_type="RERANKER")
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "retrieve",
                            "content": "Reordenando resultados...",
                        }),
                    }

                if kind == "on_chain_end" and name == "rerank":
                    capture.mark_reranking_end()
                    capture.end_span(outputs={"status": "reranking_done"})

                if kind == "on_chain_end" and name == "detect_municipio":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        municipio = output.get("municipio_detectado")
                        capture.set_municipio(municipio)
                        capture.start_span("detect_municipio", span_type="CHAIN", inputs={"query": request.query})
                        capture.end_span(outputs={"municipio": municipio or "none"})

                if kind == "on_chain_end" and name == "detect_intent":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        intent_val = output.get("intent")
                        capture.set_intent(intent_val)
                        capture.start_span("detect_intent", span_type="CHAIN", inputs={"query": request.query})
                        capture.end_span(outputs={"intent": intent_val or "especifico"})

                if kind == "on_chain_end" and name == "guardrail":
                    output = event["data"].get("output", {})
                    respuesta = output.get("respuesta_final", "") if isinstance(output, dict) else ""
                    if respuesta and not output.get("candidatos_reranked"):
                        collected_content = respuesta
                        capture.add_token(respuesta)
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "type": "token",
                                "content": respuesta,
                                "done": False,
                            }),
                        }

                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        if not collected_content:
                            capture.start_span("llm_generation", span_type="LLM", inputs={"query": request.query})
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

            # Obtener resultado final para extraer trámites y scores
            final_state = await graph.aget_state(config)
            tramites = []
            rerank_scores = []
            for r in (final_state.values.get("candidatos_reranked") or []):
                meta = r.get("metadata", {})
                tramites.append({
                    "nombre": meta.get("nombre", ""),
                    "municipio": meta.get("municipio", ""),
                    "url": meta.get("url", ""),
                    "auth": meta.get("auth", ""),
                })
                if "rerank_score" in r:
                    rerank_scores.append(r["rerank_score"])

            if rerank_scores:
                capture.set_rerank_scores(rerank_scores)

            # Documentos recuperados antes del reranking
            candidatos_raw = final_state.values.get("candidatos") or []
            if candidatos_raw:
                capture.set_retrieved_docs([
                    {
                        "content": c.get("documento", "")[:500],
                        "score": c.get("score", 0.0),
                        "metadata": c.get("metadata", {}),
                    }
                    for c in candidatos_raw
                ])

            # Documentos tras reranking
            candidatos_reranked = final_state.values.get("candidatos_reranked") or []
            if candidatos_reranked:
                capture.set_reranked_docs([
                    {
                        "content": r.get("documento", "")[:500],
                        "score": r.get("rerank_score", 0.0),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in candidatos_reranked
                ])

            context_parts = [r.get("documento", "")[:100] for r in candidatos_reranked[:3]]
            if context_parts:
                capture.set_context_preview(" | ".join(context_parts))

            # Adjunta tags `prompt.<name>.version|source` a la traza antes de
            # cerrarla. Resiliente: errores de MLflow no rompen
            # la respuesta SSE.
            capture.set_prompt_tags(get_prompts_used())

            capture.finalize()

            if tramites:
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "tramites",
                        "content": tramites,
                    }),
                }

            yield {
                "event": "message",
                "data": json.dumps({"type": "done"}),
            }

        except Exception as e:
            logger.error(f"Error en tramites stream: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "token",
                    "content": f"Error: {str(e)}",
                    "done": True,
                }),
            }

    return EventSourceResponse(event_generator())

@router.get("/health")
async def tramites_health():
    """Health check: verifica ChromaDB y la colección de trámites."""
    try:
        from backend.rag_tramites.vector_store import TramitesVectorStore
        collection = TramitesVectorStore.get_collection()
        count = collection._collection.count()
        return {
            "status": "ok",
            "coleccion": settings.TRAMITES_CHROMA_COLLECTION,
            "n_tramites": count,
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
        }
