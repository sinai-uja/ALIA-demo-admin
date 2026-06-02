"""Tests del grafo LangGraph del chatbot simple (Tab 1)."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

def test_call_llm_invokes_llm_with_messages(mock_llm, monkeypatch):
    from backend.tab1_chatbot import graph as graph_mod

    monkeypatch.setattr(graph_mod, "llm", mock_llm)
    messages = [HumanMessage(content="hola")]

    asyncio.run(graph_mod.call_llm({"messages": messages}))

    mock_llm.ainvoke.assert_awaited_once_with(messages)

def test_call_llm_returns_messages_dict(mock_llm, monkeypatch):
    from backend.tab1_chatbot import graph as graph_mod

    monkeypatch.setattr(graph_mod, "llm", mock_llm)
    result = asyncio.run(
        graph_mod.call_llm({"messages": [HumanMessage(content="hola")]})
    )

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Respuesta simulada"

def test_graph_exposes_call_llm_node():
    from backend.tab1_chatbot.graph import graph

    nodes = graph.get_graph().nodes
    assert "call_llm" in nodes

def test_graph_has_streaming_api():
    from backend.tab1_chatbot.graph import graph

    assert callable(getattr(graph, "astream_events", None))
    assert callable(getattr(graph, "ainvoke", None))
