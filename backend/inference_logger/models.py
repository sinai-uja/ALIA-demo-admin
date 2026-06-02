"""
Módulo inference_logger — Modelo de datos LogEntry.

Define la estructura de cada registro de inferencia LLM.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

@dataclass
class RetrievedDoc:
    """Documento recuperado por el retriever o reranker."""
    content: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    source: str = ""  # "retrieval" o "reranking"

@dataclass
class LogEntry:
    """Registro de una inferencia LLM."""
    # Identificación
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tab: str = ""                    # chatbot, react-agent, rag, tramites
    session_id: str = ""

    # Query y respuesta
    query: str = ""
    response: str = ""
    model: str = ""

    # Tiempos (milisegundos)
    time_total_ms: float = 0.0
    time_retrieval_ms: Optional[float] = None
    time_reranking_ms: Optional[float] = None
    time_llm_ms: float = 0.0

    # Tokens (de usage de llama.cpp server)
    tokens_input: Optional[int] = None
    tokens_output: int = 0

    # Contexto RAG
    context_preview: Optional[str] = None   # primeros 300 chars

    # Documentos recuperados (RAG)
    retrieved_docs: Optional[list[RetrievedDoc]] = None
    reranked_docs: Optional[list[RetrievedDoc]] = None

    # Específico de trámites
    municipio: Optional[str] = None
    intent: Optional[str] = None            # listado / especifico
    rerank_scores: Optional[list[float]] = None
