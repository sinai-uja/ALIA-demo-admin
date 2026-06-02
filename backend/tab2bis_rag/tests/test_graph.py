"""Tests del grafo LangGraph del RAG siempre activo (Tab 2bis)."""

import asyncio
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

def _set_chroma_mock(monkeypatch, mock_chroma_vectorstore):
    """Sustituye Chroma + HuggingFaceEmbeddings por mocks en el módulo."""
    from backend.tab2bis_rag import graph as graph_mod

    monkeypatch.setattr(
        graph_mod, "Chroma", lambda **kwargs: mock_chroma_vectorstore
    )
    monkeypatch.setattr(
        graph_mod, "HuggingFaceEmbeddings", lambda **kwargs: MagicMock()
    )
    return graph_mod

def test_retrieve_returns_context_with_docs(mock_chroma_vectorstore, monkeypatch):
    graph_mod = _set_chroma_mock(monkeypatch, mock_chroma_vectorstore)

    state = {
        "messages": [
            {"role": "user", "content": "¿cómo solicitar empadronamiento?"}
        ]
    }
    result = asyncio.run(graph_mod.retrieve(state))

    assert "context" in result
    assert "Documento de prueba 1" in result["context"]
    assert "Documento de prueba 2" in result["context"]

def test_retrieve_returns_no_results_message_when_empty(
    mock_chroma_vectorstore, monkeypatch
):
    mock_chroma_vectorstore.similarity_search.return_value = []
    graph_mod = _set_chroma_mock(monkeypatch, mock_chroma_vectorstore)

    state = {"messages": [{"role": "user", "content": "consulta inexistente"}]}
    result = asyncio.run(graph_mod.retrieve(state))

    assert "context" in result
    assert "No se encontraron" in result["context"]

def test_retrieve_returns_message_when_query_missing(
    mock_chroma_vectorstore, monkeypatch
):
    _set_chroma_mock(monkeypatch, mock_chroma_vectorstore)
    from backend.tab2bis_rag import graph as graph_mod

    result = asyncio.run(graph_mod.retrieve({"messages": []}))

    assert "context" in result
    assert "consulta" in result["context"].lower()

def test_generate_invokes_llm_with_context(mock_llm, monkeypatch):
    """el nodo `generate` envía SystemMessage + HumanMessage (antes
    enviaba solo HumanMessage). El system viene del registry/fallback con
    la instrucción declarativa; el user incluye contexto + query."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.tab2bis_rag import graph as graph_mod

    monkeypatch.setattr(graph_mod, "llm", mock_llm)
    state = {
        "messages": [{"role": "user", "content": "¿qué dice la doc?"}],
        "context": "Documento clave: respuesta es 42.",
    }

    result = asyncio.run(graph_mod.generate(state))

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    mock_llm.ainvoke.assert_awaited_once()
    sent_messages = mock_llm.ainvoke.await_args.args[0]

    # 2 mensajes — system (declarativo) + user (contexto + query).
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    # System: instrucción declarativa (sin contexto ni query).
    assert "base de conocimiento" in sent_messages[0].content
    assert "Documento clave: respuesta es 42." not in sent_messages[0].content
    # User: contexto + query (sin la instrucción declarativa).
    assert "Documento clave: respuesta es 42." in sent_messages[1].content
    assert "¿qué dice la doc?" in sent_messages[1].content

def test_graph_has_retrieve_and_generate_nodes():
    from backend.tab2bis_rag.graph import graph

    nodes = graph.get_graph().nodes
    assert "retrieve" in nodes
    assert "generate" in nodes

# ---------------------------------------------------------------------------
#  — integración con PromptProvider
# ---------------------------------------------------------------------------

def test_generate_consume_provider_para_system_y_user(mock_llm, monkeypatch):
    """Verifica que `generate` consulta el PromptProvider con los nombres
    correctos (`uja-rag-docs-system` y `uja-rag-docs-user-template`)."""
    from unittest.mock import MagicMock

    from backend.prompt_registry.provider import PromptResource
    from backend.tab2bis_rag import graph as graph_mod

    monkeypatch.setattr(graph_mod, "llm", mock_llm)
    fake_system = PromptResource(
        name="uja-rag-docs-system",
        template="SYSTEM-CUSTOM",
        version=42,
        source="registry",
    )
    fake_user = PromptResource(
        name="uja-rag-docs-user-template",
        template="USER {{contexto}} | {{query}}",
        version=42,
        source="registry",
    )

    fake_provider = MagicMock()

    def fake_get_prompt(name):
        return fake_system if "system" in name else fake_user

    fake_provider.get_prompt.side_effect = fake_get_prompt
    monkeypatch.setattr(graph_mod, "get_provider", lambda: fake_provider)

    state = {
        "messages": [{"role": "user", "content": "consulta de prueba"}],
        "context": "CTX-DEMO",
    }

    asyncio.run(graph_mod.generate(state))

    # Provider consultado con los nombres correctos.
    called_names = [c.args[0] for c in fake_provider.get_prompt.call_args_list]
    assert "uja-rag-docs-system" in called_names
    assert "uja-rag-docs-user-template" in called_names

    # Mensajes enviados al LLM con la plantilla custom aplicada.
    sent_messages = mock_llm.ainvoke.await_args.args[0]
    assert sent_messages[0].content == "SYSTEM-CUSTOM"
    assert sent_messages[1].content == "USER CTX-DEMO | consulta de prueba"

def test_generate_fallback_silencioso_si_mlflow_caido(mock_llm, monkeypatch):
    """Si el PromptProvider no puede contactar MLflow, debe caer al
    fallback empaquetado (`RAG_DOCS_SYSTEM`, `RAG_DOCS_USER_TEMPLATE`)
    y el nodo seguir funcionando. Replica el caso "MLflow caído" sin
    necesidad de levantar nada."""
    from backend.prompt_registry.fallback import (
        RAG_DOCS_SYSTEM,
        RAG_DOCS_USER_TEMPLATE,
    )
    from backend.prompt_registry.provider import PromptProvider, reset_provider
    from backend.tab2bis_rag import graph as graph_mod

    monkeypatch.setattr(graph_mod, "llm", mock_llm)
    # Provider con enabled=False fuerza el camino del fallback sin intentar
    # ninguna llamada a mlflow.
    reset_provider()
    monkeypatch.setattr(
        graph_mod,
        "get_provider",
        lambda: PromptProvider(
            tracking_uri=None,
            default_alias="production",
            cache_ttl_seconds=10,
            enabled=False,
        ),
    )

    state = {
        "messages": [{"role": "user", "content": "consulta fallback"}],
        "context": "ctx-fallback",
    }

    asyncio.run(graph_mod.generate(state))

    sent_messages = mock_llm.ainvoke.await_args.args[0]
    # System es el fallback empaquetado tal cual.
    assert sent_messages[0].content == RAG_DOCS_SYSTEM
    # User template aplicado con los 2 valores → reconstrucción literal.
    expected_user = (
        RAG_DOCS_USER_TEMPLATE
        .replace("{{contexto}}", "ctx-fallback")
        .replace("{{query}}", "consulta fallback")
    )
    assert sent_messages[1].content == expected_user
