"""
Módulo rag_tramites — Estado del grafo LangGraph.

Define el TypedDict que fluye entre los nodos del grafo de trámites.
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph import add_messages

class TramitesState(TypedDict):
    """Estado del grafo de trámites municipales."""
    session_id: str
    messages: Annotated[list[dict], add_messages]
    query_actual: str
    municipio_detectado: Optional[str]
    intent: str  # "listado" o "especifico"
    candidatos: list[dict]
    candidatos_reranked: list[dict]
    respuesta_final: str
    necesita_refinamiento: bool
    turno: int
