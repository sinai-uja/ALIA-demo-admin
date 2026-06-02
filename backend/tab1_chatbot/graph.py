"""
Grafo LangGraph para el chatbot simple (Tab 1).

Implementa el grafo más simple posible: [START] → [call_llm] → [END].
Usa StateGraph con MessagesState de LangGraph.

LangGraph versión: >=0.2
API utilizada: langgraph.graph.StateGraph, langgraph.graph.MessagesState
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

from backend.config import settings

llm = ChatOpenAI(
    base_url=settings.ALIA_LLM_URL,
    api_key=settings.ALIA_API_KEY,
    model=settings.ALIA_LLM_MODEL,
    streaming=True,
    max_tokens=1024,
)

async def call_llm(state: MessagesState) -> MessagesState:
    """Nodo que llama al LLM con el historial completo de mensajes."""
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# Construcción del grafo
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_llm", call_llm)
graph_builder.add_edge(START, "call_llm")
graph_builder.add_edge("call_llm", END)

# Compilación del grafo
graph = graph_builder.compile()
