"""
Módulo rag_tramites — Construcción del grafo LangGraph.

Flujo:
  START → detect_municipio → detect_intent
    ├─ (listado)    → retrieve_all → guardrail → generate → END
    └─ (especifico) → retrieve → rerank → guardrail
                        ├─ (sin resultados) → END
                        └─ (con resultados) → generate → END

Usa MemorySaver para aislar memoria por session_id.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from backend.rag_tramites.graph.state import TramitesState
from backend.rag_tramites.graph.nodes import (
    node_detect_municipio,
    node_detect_intent,
    node_retrieve,
    node_retrieve_all,
    node_rerank,
    node_guardrail,
    node_generate,
)

logger = logging.getLogger(__name__)

# Construcción del grafo
graph_builder = StateGraph(TramitesState)

# Nodos
graph_builder.add_node("detect_municipio", node_detect_municipio)
graph_builder.add_node("detect_intent", node_detect_intent)
graph_builder.add_node("retrieve", node_retrieve)
graph_builder.add_node("retrieve_all", node_retrieve_all)
graph_builder.add_node("rerank", node_rerank)
graph_builder.add_node("guardrail", node_guardrail)
graph_builder.add_node("generate", node_generate)

# Edges comunes
graph_builder.add_edge(START, "detect_municipio")
graph_builder.add_edge("detect_municipio", "detect_intent")

def _route_after_intent(state: dict) -> str:
    """Dirige al flujo de listado completo o búsqueda específica."""
    if state.get("intent") == "listado":
        return "retrieve_all"
    return "retrieve"

graph_builder.add_conditional_edges(
    "detect_intent",
    _route_after_intent,
    {"retrieve_all": "retrieve_all", "retrieve": "retrieve"},
)

# Rama listado: retrieve_all → guardrail → generate → END
graph_builder.add_edge("retrieve_all", "guardrail")

# Rama específica: retrieve → rerank → guardrail
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "guardrail")

def _route_after_guardrail(state: dict) -> str:
    """Decide si generar respuesta o terminar directamente."""
    reranked = state.get("candidatos_reranked", [])
    if not reranked:
        return END
    return "generate"

graph_builder.add_conditional_edges(
    "guardrail",
    _route_after_guardrail,
    {END: END, "generate": "generate"},
)
graph_builder.add_edge("generate", END)

# Compilación con checkpointer para memoria por sesión
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

logger.info("Grafo de trámites compilado y listo")

async def invocar_tramites(query: str, session_id: str) -> dict:
    """Invoca el grafo de trámites con aislamiento por sesión.

    Args:
        query: Consulta del usuario.
        session_id: ID de sesión para aislar la memoria.

    Returns:
        Dict con 'respuesta', 'tramites' y 'session_id'.
    """
    config = {"configurable": {"thread_id": session_id}}

    input_state = {
        "session_id": session_id,
        "query_actual": query,
        "messages": [{"role": "user", "content": query}],
        "municipio_detectado": None,
        "intent": "especifico",
        "candidatos": [],
        "candidatos_reranked": [],
        "respuesta_final": "",
        "necesita_refinamiento": False,
        "turno": 0,
    }

    result = await graph.ainvoke(input_state, config=config)

    tramites = []
    for r in result.get("candidatos_reranked", []):
        meta = r.get("metadata", {})
        tramites.append({
            "nombre": meta.get("nombre", ""),
            "municipio": meta.get("municipio", ""),
            "url": meta.get("url", ""),
            "auth": meta.get("auth", ""),
        })

    return {
        "respuesta": result.get("respuesta_final", ""),
        "tramites": tramites,
        "session_id": session_id,
    }
