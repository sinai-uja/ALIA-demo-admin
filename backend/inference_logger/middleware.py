"""
Modulo inference_logger -- Middleware para captura de metricas de inferencia.

Provee un context manager que envuelve los event generators SSE
para capturar metricas sin modificar los grafos LangGraph.
"""

import logging
import time
from typing import Optional

from backend.inference_logger.models import LogEntry, RetrievedDoc
from backend.inference_logger.service import inference_logger
from backend.mlflow_tracker import log_to_mlflow
from backend.mlflow_tracer import (
    compute_cost_eur,
    end_span as _tracer_end_span,
    end_trace as _tracer_end_trace,
    set_trace_tags as _tracer_set_trace_tags,
    start_child_span as _tracer_start_child,
    start_trace as _tracer_start_trace,
)

logger = logging.getLogger(__name__)

class InferenceCapture:
    """Captura metricas de una inferencia durante el streaming SSE.

    Uso:
        capture = InferenceCapture(tab="chatbot", session_id="abc", query="hola")
        # Durante el streaming:
        capture.add_token("Hola")
        capture.add_token(", mundo")
        # Al finalizar:
        capture.set_model("alia-40b")
        capture.set_tokens(input=150, output=42)
        capture.finalize()  # registra el LogEntry
    """

    def __init__(
        self,
        tab: str,
        session_id: str,
        query: str,
        create_trace: bool = True,
        user_id: Optional[str] = None,
    ):
        self.tab = tab
        self.session_id = session_id
        self.user_id = user_id
        self.query = query
        self._start_time = time.time()
        self._response_parts: list[str] = []
        self._model: str = ""
        self._tokens_input: Optional[int] = None
        self._tokens_output: int = 0
        self._time_retrieval_ms: Optional[float] = None
        self._time_reranking_ms: Optional[float] = None
        self._time_llm_start: Optional[float] = None
        self._time_llm_ms: float = 0.0
        self._context_preview: Optional[str] = None
        self._context_full: Optional[str] = None
        self._municipio: Optional[str] = None
        self._intent: Optional[str] = None
        self._rerank_scores: Optional[list[float]] = None
        self._retrieved_docs: Optional[list[RetrievedDoc]] = None
        self._reranked_docs: Optional[list[RetrievedDoc]] = None

        # MLflow Tracing (imperativo via MlflowClient)
        # create_trace=False para servicios que gestionan su propio trace (ej. comparador)
        if create_trace:
            attrs = {"tab": tab, "session_id": session_id}
            if user_id:
                attrs["user_id"] = user_id
            self._trace_ctx = _tracer_start_trace(
                experiment_name=tab,
                name=f"{tab}_query",
                inputs={"query": query},
                attributes=attrs,
                session_id=session_id,
                user_id=user_id,
            )
        else:
            self._trace_ctx = None
        self._current_span_id: Optional[str] = None

    @property
    def request_id(self) -> Optional[str]:
        """request_id del trace MLflow activo (None si tracing deshabilitado)."""
        return self._trace_ctx.request_id if self._trace_ctx else None

    def set_prompt_tags(self, prompts_used) -> None:  # noqa: ANN001
        """Adjunta tags `prompt.<name>.version|source` a la traza activa.

        el provider del Prompt Registry expone via ContextVar la
        lista de PromptResource consumidos durante la request. El router
        llama a `capture.set_prompt_tags(get_prompts_used())` antes de
        `finalize()` para que la traza correlacione el prompt usado con el
        feedback humano sobre el mismo trace_id.

        Resiliente a fallos: si MLflow no responde, los tags se pierden pero
        la respuesta SSE no se rompe (delegado en `mlflow_tracer.set_trace_tags`).
        """
        if not prompts_used:
            return
        tags: dict[str, str] = {}
        for resource in prompts_used:
            base = f"prompt.{resource.name}"
            tags[f"{base}.version"] = str(resource.version)
            tags[f"{base}.source"] = resource.source
        _tracer_set_trace_tags(self._trace_ctx, tags)

    def add_token(self, token: str) -> None:
        """Acumula un token de la respuesta."""
        if self._time_llm_start is None:
            self._time_llm_start = time.time()
        self._response_parts.append(token)

    def set_model(self, model: str) -> None:
        self._model = model

    def set_tokens(self, input: Optional[int] = None, output: Optional[int] = None) -> None:
        if input is not None:
            self._tokens_input = input
        if output is not None:
            self._tokens_output = output

    def mark_retrieval_start(self) -> None:
        self._retrieval_start = time.time()

    def mark_retrieval_end(self) -> None:
        if hasattr(self, "_retrieval_start"):
            self._time_retrieval_ms = (time.time() - self._retrieval_start) * 1000

    def mark_reranking_start(self) -> None:
        self._reranking_start = time.time()

    def mark_reranking_end(self) -> None:
        if hasattr(self, "_reranking_start"):
            self._time_reranking_ms = (time.time() - self._reranking_start) * 1000

    def set_context_preview(self, context: str) -> None:
        self._context_preview = context[:300] if context else None
        self._context_full = context or None

    def set_municipio(self, municipio: Optional[str]) -> None:
        self._municipio = municipio

    def set_intent(self, intent: Optional[str]) -> None:
        self._intent = intent

    def set_rerank_scores(self, scores: Optional[list[float]]) -> None:
        self._rerank_scores = scores

    # -- MLflow Tracing: gestion de spans --

    def start_span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        inputs: Optional[dict] = None,
        attributes: Optional[dict] = None,
    ) -> None:
        """Abre un span hijo dentro del trace activo."""
        self._current_span_id = _tracer_start_child(
            self._trace_ctx, name=name, span_type=span_type,
            inputs=inputs, attributes=attributes,
        )

    def end_span(self, outputs: Optional[dict] = None) -> None:
        """Cierra el span activo."""
        if self._current_span_id is not None:
            _tracer_end_span(self._trace_ctx, self._current_span_id, outputs=outputs)
            self._current_span_id = None

    def set_retrieved_docs(self, docs: list[dict]) -> None:
        """Registra documentos devueltos por el retriever."""
        self._retrieved_docs = [
            RetrievedDoc(
                content=d.get("content", "")[:500],
                score=d.get("score", 0.0),
                metadata=d.get("metadata", {}),
                source="retrieval",
            )
            for d in docs
        ]

    def set_reranked_docs(self, docs: list[dict]) -> None:
        """Registra documentos tras reranking."""
        self._reranked_docs = [
            RetrievedDoc(
                content=d.get("content", "")[:500],
                score=d.get("score", 0.0),
                metadata=d.get("metadata", {}),
                source="reranking",
            )
            for d in docs
        ]

    def finalize(self) -> LogEntry:
        """Crea y registra el LogEntry con las metricas capturadas."""
        now = time.time()
        time_total_ms = (now - self._start_time) * 1000

        if self._time_llm_start:
            self._time_llm_ms = (now - self._time_llm_start) * 1000

        response = "".join(self._response_parts)
        if not self._tokens_output and response:
            self._tokens_output = len(response.split())

        entry = LogEntry(
            tab=self.tab,
            session_id=self.session_id,
            query=self.query,
            response=response,
            model=self._model,
            time_total_ms=time_total_ms,
            time_retrieval_ms=self._time_retrieval_ms,
            time_reranking_ms=self._time_reranking_ms,
            time_llm_ms=self._time_llm_ms,
            tokens_input=self._tokens_input,
            tokens_output=self._tokens_output,
            context_preview=self._context_preview,
            retrieved_docs=self._retrieved_docs,
            reranked_docs=self._reranked_docs,
            municipio=self._municipio,
            intent=self._intent,
            rerank_scores=self._rerank_scores,
        )

        inference_logger.add(entry)
        log_to_mlflow(entry, context_full=self._context_full)

        # Cerrar span LLM activo con token usage
        tokens_in = self._tokens_input or 0
        tokens_out = self._tokens_output or 0
        tokens_total = tokens_in + tokens_out
        cost = compute_cost_eur(tokens_in, tokens_out)

        if self._current_span_id is not None:
            _tracer_end_span(self._trace_ctx, self._current_span_id, outputs={
                "response": response[:500],
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "tokens_total": tokens_total,
                "cost_eur": cost,
                "model": self._model,
            })
            self._current_span_id = None

        # Cerrar el trace raiz con totales
        _tracer_end_trace(self._trace_ctx, outputs={
            "response": response[:500],
            "model": self._model,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_total": tokens_total,
            "cost_eur": cost,
            "time_total_ms": round(time_total_ms, 1),
        })

        logger.debug(
            f"Inferencia registrada: tab={self.tab}, "
            f"total={time_total_ms:.0f}ms, tokens_out={self._tokens_output}"
        )
        return entry
