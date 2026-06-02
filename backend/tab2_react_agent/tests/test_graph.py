"""Tests del agente ReAct y la herramienta de RAG (Tab 2)."""

from unittest.mock import MagicMock

def test_search_knowledge_base_returns_docs_concatenated(
    mock_chroma_vectorstore, monkeypatch
):
    from backend.tab2_react_agent import rag_tools

    monkeypatch.setattr(
        rag_tools, "Chroma", lambda **kwargs: mock_chroma_vectorstore
    )
    monkeypatch.setattr(
        rag_tools, "HuggingFaceEmbeddings", lambda **kwargs: MagicMock()
    )

    result = rag_tools.search_knowledge_base.invoke({"query": "consulta"})

    assert "Documento de prueba 1" in result
    assert "Documento de prueba 2" in result
    assert "test-1.txt" in result

def test_search_knowledge_base_returns_message_when_empty(
    mock_chroma_vectorstore, monkeypatch
):
    mock_chroma_vectorstore.similarity_search.return_value = []
    from backend.tab2_react_agent import rag_tools

    monkeypatch.setattr(
        rag_tools, "Chroma", lambda **kwargs: mock_chroma_vectorstore
    )
    monkeypatch.setattr(
        rag_tools, "HuggingFaceEmbeddings", lambda **kwargs: MagicMock()
    )

    result = rag_tools.search_knowledge_base.invoke({"query": "consulta"})

    assert "No se encontraron" in result

def test_search_knowledge_base_returns_error_message_on_exception(monkeypatch):
    from backend.tab2_react_agent import rag_tools

    def _raise(**kwargs):
        raise RuntimeError("ChromaDB indisponible")

    monkeypatch.setattr(rag_tools, "HuggingFaceEmbeddings", _raise)

    result = rag_tools.search_knowledge_base.invoke({"query": "consulta"})

    assert "Error" in result
    assert "ChromaDB indisponible" in result

def test_graph_exposes_search_knowledge_base_tool():
    from backend.tab2_react_agent.graph import tools

    tool_names = [t.name for t in tools]
    assert "search_knowledge_base" in tool_names

def test_graph_compiled_has_streaming_api():
    from backend.tab2_react_agent.graph import graph

    assert callable(getattr(graph, "astream_events", None))
    assert callable(getattr(graph, "ainvoke", None))
