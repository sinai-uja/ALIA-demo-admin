"""
Módulo comparador — Router FastAPI para comparación dual de LLMs.

Endpoint: POST /comparador/stream
Ejecuta retrieval + reranking una vez, luego invoca dos LLMs en paralelo
y devuelve tokens diferenciados por canal SSE.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.comparador.config import llm_a, llm_a_name, llm_b, llm_b_name
from backend.inference_logger.middleware import InferenceCapture
from backend.mlflow_tracer import (
    compute_cost_eur,
    emit_trace_id_event,
    end_span as tracer_end_span,
    end_trace as tracer_end_trace,
    set_trace_tags as tracer_set_trace_tags,
    start_child_span as tracer_start_child,
    start_trace as tracer_start_trace,
)
from backend.prompt_registry.provider import (
    begin_prompt_tracking,
    get_prompts_used,
    get_provider,
)
from backend.rag_tramites.graph.nodes import (
    build_tramites_user_prompt,
    node_detect_intent,
    node_detect_municipio,
    node_rerank,
    node_retrieve,
    node_retrieve_all,
)

logger = logging.getLogger(__name__)

router = APIRouter()

class ComparadorRequest(BaseModel):
    """Modelo de petición para el comparador."""
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

def _build_prompt(query: str, reranked: list[dict], intent: str) -> str:
    """Construye el prompt de usuario para el Comparador.

    El comparador es stateless (no acumula turnos previos), así que llama al
    builder unificado sin `history`. Esta función queda como alias por
    claridad local; cualquier cambio en el formato del prompt vive en
    `build_tramites_user_prompt` (fuente única).
    """
    return build_tramites_user_prompt(query, reranked, intent, history=None)

@router.post("/stream")
async def comparador_stream(request: ComparadorRequest):
    """Endpoint SSE que compara dos LLMs sobre la misma consulta RAG trámites."""
    if llm_b is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Comparador no disponible: define ALIA_LLM_URL_2 y "
                "ALIA_LLM_MODEL_2 en .env para habilitar el segundo LLM."
            ),
        )

    async def event_generator():
        # tracking de prompts usados → tags MLflow al cerrar.
        begin_prompt_tracking()
        try:
            query = request.query.strip()
            if not query:
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "error",
                        "content": "La consulta no puede estar vacía.",
                    }),
                }
                return

            # Si el cliente no manda session_id, generamos uno en servidor
            # para que cada usuario tenga su propio trace MLflow.
            session_id = (request.session_id or "").strip() or str(uuid.uuid4())
            user_id = request.user_id

            # Trace raiz del comparador con session_id. Se inicia antes que
            # cualquier yield para que el primer evento SSE sea el trace_id.
            attrs = {"tab": "comparador", "session_id": session_id}
            if user_id:
                attrs["user_id"] = user_id
            trace_ctx = tracer_start_trace(
                experiment_name="comparador",
                name="comparador_query",
                inputs={"query": query},
                attributes=attrs,
                session_id=session_id,
                user_id=user_id,
            )

            # Comparador comparte un unico trace_ctx entre ambos LLMs; emitimos
            # un trace_id por modelo para que la UI pueda asociar el feedback
            # de preferencia (A/B/empate) al trace correcto.
            request_id = trace_ctx.request_id if trace_ctx else None
            yield emit_trace_id_event(request_id, llm="A")
            yield emit_trace_id_event(request_id, llm="B")

            # ── Retrieval (una sola vez, compartido) ──
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "retrieve",
                    "content": "Buscando trámites relevantes...",
                }),
            }

            state = {
                "query_actual": query,
                "municipio_detectado": None,
                "intent": "especifico",
                "candidatos": [],
                "candidatos_reranked": [],
            }

            sp = tracer_start_child(trace_ctx, "detect_municipio", span_type="CHAIN", inputs={"query": query})
            state.update(node_detect_municipio(state))
            tracer_end_span(trace_ctx, sp, outputs={"municipio": state.get("municipio_detectado") or "none"})

            sp = tracer_start_child(trace_ctx, "detect_intent", span_type="CHAIN", inputs={"query": query})
            state.update(node_detect_intent(state))
            tracer_end_span(trace_ctx, sp, outputs={"intent": state["intent"]})

            sp = tracer_start_child(trace_ctx, "retrieval", span_type="RETRIEVER", inputs={"query": query, "intent": state["intent"]})
            if state["intent"] == "listado":
                state.update(node_retrieve_all(state))
            else:
                state.update(node_retrieve(state))
            n_candidatos = len(state.get("candidatos", []))
            tracer_end_span(trace_ctx, sp, outputs={"n_docs": n_candidatos})

            if state["intent"] != "listado":
                sp = tracer_start_child(trace_ctx, "reranking", span_type="RERANKER", inputs={"n_docs_input": n_candidatos})
                state.update(node_rerank(state))
                tracer_end_span(trace_ctx, sp, outputs={"n_docs_output": len(state.get("candidatos_reranked", []))})

            reranked = state.get("candidatos_reranked", [])
            intent = state["intent"]

            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "retrieve",
                    "content": f"{len(reranked)} trámites encontrados. Generando respuestas...",
                }),
            }

            # ── Construir prompt compartido ──
            # system prompt cargado del registry (paridad estructural
            # con Trámites garantizada por compartir el mismo `uja-tramites-system`).
            user_prompt = _build_prompt(query, reranked, intent)
            system_resource = get_provider().get_prompt("uja-tramites-system")
            messages = [
                SystemMessage(content=system_resource.template),
                HumanMessage(content=user_prompt),
            ]

            # ── InferenceCapture para cada LLM ──
            # create_trace=False: el comparador gestiona su propio trace (trace_ctx)
            capture_a = InferenceCapture(tab="comparador", session_id=session_id, query=query, create_trace=False, user_id=user_id)
            capture_b = InferenceCapture(tab="comparador", session_id=session_id, query=query, create_trace=False, user_id=user_id)

            context_preview = " | ".join(
                r.get("documento", "")[:100] for r in reranked[:3]
            )
            capture_a.set_context_preview(context_preview)
            capture_b.set_context_preview(context_preview)

            municipio = state.get("municipio_detectado")
            capture_a.set_municipio(municipio)
            capture_b.set_municipio(municipio)
            capture_a.set_intent(intent)
            capture_b.set_intent(intent)

            # Documentos recuperados (antes del reranking)
            candidatos_raw = state.get("candidatos", [])
            if candidatos_raw:
                raw_docs = [
                    {
                        "content": c.get("documento", "")[:500],
                        "score": c.get("score", 0.0),
                        "metadata": c.get("metadata", {}),
                    }
                    for c in candidatos_raw
                ]
                capture_a.set_retrieved_docs(raw_docs)
                capture_b.set_retrieved_docs(raw_docs)

            # Documentos tras reranking
            if reranked:
                reranked_docs = [
                    {
                        "content": r.get("documento", "")[:500],
                        "score": r.get("rerank_score", 0.0),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in reranked
                ]
                capture_a.set_reranked_docs(reranked_docs)
                capture_b.set_reranked_docs(reranked_docs)

            rerank_scores = [r["rerank_score"] for r in reranked if "rerank_score" in r]
            if rerank_scores:
                capture_a.set_rerank_scores(rerank_scores)
                capture_b.set_rerank_scores(rerank_scores)

            # ── Invocar ambos LLMs en paralelo ──
            queue: asyncio.Queue = asyncio.Queue()

            async def stream_llm(llm, llm_label, capture):
                """Streama un LLM y pone los eventos en la cola compartida."""
                llm_span = tracer_start_child(trace_ctx, f"llm_generation_{llm_label}", span_type="LLM", inputs={"model": llm_label, "query": query[:200]})
                try:
                    async for chunk in llm.astream(messages):
                        if chunk.content:
                            capture.add_token(chunk.content)
                            await queue.put({
                                "type": "token",
                                "llm": llm_label,
                                "content": chunk.content,
                            })
                        # Capturar usage del último chunk
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            capture.set_tokens(
                                input=chunk.usage_metadata.get("input_tokens"),
                                output=chunk.usage_metadata.get("output_tokens"),
                            )
                        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                            model_name = chunk.response_metadata.get("model_name", "")
                            if model_name:
                                capture.set_model(model_name)
                except Exception as e:
                    logger.error(f"Error en LLM {llm_label}: {e}")
                    await queue.put({
                        "type": "token",
                        "llm": llm_label,
                        "content": f"\n\n[Error: {e}]",
                    })
                finally:
                    t_in = capture._tokens_input or 0
                    t_out = capture._tokens_output or 0
                    tracer_end_span(trace_ctx, llm_span, outputs={
                        "response": "".join(p for p in capture._response_parts)[:500],
                        "tokens_input": t_in,
                        "tokens_output": t_out,
                        "tokens_total": t_in + t_out,
                        "cost_eur": compute_cost_eur(t_in, t_out),
                        "model": capture._model,
                    })
                    entry = capture.finalize()
                    await queue.put({
                        "type": "metrics",
                        "llm": llm_label,
                        "model": entry.model or llm_label,
                        "time_ms": round(entry.time_total_ms, 0),
                        "tokens_output": entry.tokens_output,
                    })
                    await queue.put({"type": f"_done_{llm_label}"})

            task_a = asyncio.create_task(stream_llm(llm_a, "A", capture_a))
            task_b = asyncio.create_task(stream_llm(llm_b, "B", capture_b))

            # Emitir nombres de modelo al inicio
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "models",
                    "model_a": llm_a_name,
                    "model_b": llm_b_name,
                }),
            }

            # ── Consumir cola hasta que ambos terminen ──
            done_count = 0
            while done_count < 2:
                item = await queue.get()
                if item["type"].startswith("_done_"):
                    done_count += 1
                    continue
                yield {
                    "event": "message",
                    "data": json.dumps(item),
                }

            # Asegurar que las tasks están terminadas
            await task_a
            await task_b

            # tags `prompt.<name>.version|source` antes de
            # cerrar la traza. El comparador usa `uja-tramites-system` y
            # `uja-tramites-user-template` (los mismos que Trámites), por
            # lo que la traza correlaciona la versión del prompt con el
            # feedback A/B/empate que pueda venir después.
            prompts_used = get_prompts_used()
            if prompts_used:
                prompt_tags: dict[str, str] = {}
                for resource in prompts_used:
                    prompt_tags[f"prompt.{resource.name}.version"] = str(resource.version)
                    prompt_tags[f"prompt.{resource.name}.source"] = resource.source
                tracer_set_trace_tags(trace_ctx, prompt_tags)

            # Cerrar trace con totales combinados de ambos LLMs
            total_in = (capture_a._tokens_input or 0) + (capture_b._tokens_input or 0)
            total_out = (capture_a._tokens_output or 0) + (capture_b._tokens_output or 0)
            tracer_end_trace(trace_ctx, outputs={
                "status": "done",
                "tokens_input": total_in,
                "tokens_output": total_out,
                "tokens_total": total_in + total_out,
                "cost_eur": compute_cost_eur(total_in, total_out),
            })

            yield {
                "event": "message",
                "data": json.dumps({"type": "done"}),
            }

        except Exception as e:
            logger.error(f"Error en comparador stream: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "content": f"Error: {e}",
                }),
            }

    return EventSourceResponse(event_generator())
