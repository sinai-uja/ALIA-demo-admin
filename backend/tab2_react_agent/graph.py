"""
Grafo LangGraph para el agente ReAct con RAG (Tab 2).

Implementa un agente ReAct usando create_react_agent de langgraph.prebuilt.
El agente tiene acceso a la herramienta search_knowledge_base para buscar
en la base de conocimiento ChromaDB.

LangGraph versión: >=0.2
API utilizada: langgraph.prebuilt.create_react_agent
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from backend.config import settings
from backend.tab2_react_agent.rag_tools import search_knowledge_base

llm = ChatOpenAI(
    base_url=settings.ALIA_LLM_URL,
    api_key=settings.ALIA_API_KEY,
    model=settings.ALIA_LLM_MODEL,
    streaming=True,
    max_tokens=1024,
)

# Lista de herramientas disponibles para el agente
tools = [search_knowledge_base]

# Creación del agente ReAct con LangGraph prebuilt
graph = create_react_agent(llm, tools)
