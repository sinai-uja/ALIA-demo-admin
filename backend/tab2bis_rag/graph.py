"""
Grafo LangGraph para RAG siempre activo (Tab 2bis).

Implementa un flujo RAG que siempre busca en ChromaDB antes de responder:
  [START] → [retrieve] → [generate] → [END]

No depende de tool calling del LLM, compatible con cualquier modelo local.

LangGraph versión: >=0.2
API utilizada: langgraph.graph.StateGraph
"""

import logging
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

from backend.config import settings
from backend.prompt_registry.provider import get_provider

# Logging a fichero para diagnóstico del RAG
_log_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rag.log")
os.makedirs(os.path.dirname(_log_file), exist_ok=True)

logger = logging.getLogger("tab2bis_rag")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_file_handler)

llm = ChatOpenAI(
    base_url=settings.ALIA_LLM_URL,
    api_key=settings.ALIA_API_KEY,
    model=settings.ALIA_LLM_MODEL,
    streaming=True,
    max_tokens=1024,
)
logger.info(f"LLM configurado: model={settings.ALIA_LLM_MODEL}, url={settings.ALIA_LLM_URL}")
logger.info(f"Embeddings configurados: model={settings.EMBEDDING_MODEL_PATH}, device={settings.EMBEDDING_DEVICE}")
logger.info(f"ChromaDB: dir={settings.CHROMA_PERSIST_DIR}")

class RAGState(MessagesState):
    """Estado del grafo RAG con campo para documentos recuperados."""
    context: str

async def retrieve(state: RAGState) -> dict:
    """Nodo de recuperación: busca en ChromaDB los documentos más relevantes."""
    logger.info("=" * 60)
    logger.info("NODO: retrieve — inicio")

    # Extraer la última pregunta del usuario de los mensajes
    query = ""
    for m in reversed(state["messages"]):
        content = None
        if isinstance(m, dict) and m.get("role") == "user":
            content = m["content"]
        elif hasattr(m, "type") and m.type == "human":
            content = m.content

        if content is not None:
            # Gradio puede enviar content como lista de dicts [{"text": "...", "type": "text"}]
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                query = " ".join(text_parts)
            else:
                query = str(content)
            break

    logger.info(f"Query extraída: '{query}'")
    logger.debug(f"Mensajes en estado ({len(state['messages'])}): {[str(m)[:100] for m in state['messages']]}")

    if not query:
        logger.warning("No se encontró query del usuario")
        return {"context": "No se proporcionó una consulta."}

    try:
        logger.info(f"Cargando modelo de embeddings: {settings.EMBEDDING_MODEL_PATH}")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Conectando a ChromaDB: {settings.CHROMA_PERSIST_DIR}, colección=knowledge_base")
        vectorstore = Chroma(
            collection_name="knowledge_base",
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        # k=5 da cobertura suficiente cuando la respuesta requiere
        # consolidar información dispersa en varios chunks del corpus.
        logger.info(f"Ejecutando similarity_search(query='{query}', k=5)")
        docs = vectorstore.similarity_search(query, k=5)
        logger.info(f"Documentos recuperados: {len(docs)}")

        if not docs:
            logger.warning("ChromaDB no devolvió documentos")
            return {"context": "No se encontraron documentos relevantes."}

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconocido")
            question = doc.metadata.get("question", "N/A")
            logger.info(f"  Doc {i}: source={source}, question={question[:80]}")
            logger.info(f"  Doc {i} contenido ({len(doc.page_content)} chars):")
            logger.info(f"  --- CHUNK {i} INICIO ---")
            for line in doc.page_content.splitlines():
                logger.info(f"  | {line}")
            logger.info(f"  --- CHUNK {i} FIN ---")
            context_parts.append(f"[Documento {i} - Fuente: {source}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Contexto total: {len(context)} chars")
        logger.info("NODO: retrieve — fin")
        return {"context": context}

    except Exception as e:
        logger.error(f"Error en retrieve: {e}", exc_info=True)
        return {"context": f"Error al buscar en la base de conocimiento: {str(e)}"}

async def generate(state: RAGState) -> dict:
    """Nodo de generación: responde usando el contexto recuperado de ChromaDB."""
    logger.info("=" * 60)
    logger.info("NODO: generate — inicio")

    context = state.get("context", "")
    logger.info(f"Contexto recibido del retrieve: {len(context)} chars")

    if not context:
        logger.warning("CONTEXTO VACÍO — el LLM no tendrá información del RAG")
    else:
        logger.info("--- CONTEXTO COMPLETO QUE SE ENVÍA AL LLM ---")
        for line in context.splitlines():
            logger.info(f"  CTX | {line}")
        logger.info("--- FIN CONTEXTO ---")

    # Extraer la pregunta del usuario como texto plano
    user_query = ""
    for m in state["messages"]:
        content = None
        if isinstance(m, dict) and m.get("role") == "user":
            content = m["content"]
        elif hasattr(m, "type") and m.type == "human":
            content = m.content

        if content is not None:
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                user_query = " ".join(text_parts)
            else:
                user_query = str(content)

    logger.info(f"Pregunta del usuario (texto plano): '{user_query}'")

    # Prompts cargados del MLflow Prompt Registry. La instrucción
    # declarativa ("A continuación tienes información…") vive en el
    # SystemMessage para que sea editable independientemente del bloque
    # de contexto desde la UI de MLflow.
    from langchain_core.messages import HumanMessage, SystemMessage
    provider = get_provider()
    system_resource = provider.get_prompt("uja-rag-docs-system")
    user_resource = provider.get_prompt("uja-rag-docs-user-template")
    user_prompt = user_resource.format(contexto=context, query=user_query)

    messages = [
        SystemMessage(content=system_resource.template),
        HumanMessage(content=user_prompt),
    ]

    logger.info(
        f"Mensajes enviados al LLM: {len(messages)} "
        f"(prompt_registry sources: sys={system_resource.source}/v{system_resource.version}, "
        f"usr={user_resource.source}/v{user_resource.version})"
    )
    logger.info("--- SYSTEM PROMPT COMPLETO ---")
    for line in system_resource.template.splitlines():
        logger.info(f"  SYS | {line}")
    logger.info("--- FIN SYSTEM PROMPT ---")
    logger.info("--- USER PROMPT COMPLETO ---")
    for line in user_prompt.splitlines():
        logger.info(f"  USR | {line}")
    logger.info("--- FIN USER PROMPT ---")

    logger.info(f"Invocando LLM: {settings.ALIA_LLM_MODEL}")
    response = await llm.ainvoke(messages)

    logger.info(f"--- RESPUESTA LLM ({len(response.content)} chars) ---")
    for line in response.content.splitlines():
        logger.info(f"  LLM | {line}")
    logger.info("--- FIN RESPUESTA LLM ---")
    logger.info("NODO: generate — fin")
    logger.info("=" * 60)

    return {"messages": [response]}

# Construcción del grafo: [START] → [retrieve] → [generate] → [END]
graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Compilación del grafo
graph = graph_builder.compile()
logger.info("Grafo RAG compilado y listo")
