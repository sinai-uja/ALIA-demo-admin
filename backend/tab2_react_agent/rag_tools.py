"""
Herramienta RAG para el agente ReAct (Tab 2).

Define la herramienta search_knowledge_base que busca en ChromaDB
documentos relevantes para una consulta dada.

Usa un modelo de embeddings local (HuggingFace) definido en EMBEDDING_MODEL_PATH.
"""

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings

@tool
def search_knowledge_base(query: str) -> str:
    """Busca en la base de conocimiento documentos relevantes para la consulta dada.

    Utiliza ChromaDB con embeddings locales (HuggingFace) para búsqueda por similitud.
    Devuelve los top-3 documentos más relevantes.

    Args:
        query: La consulta de búsqueda en lenguaje natural.

    Returns:
        Los documentos más relevantes concatenados como string.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            collection_name="knowledge_base",
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        # k=5 da cobertura suficiente cuando la respuesta requiere
        # consolidar información dispersa en varios chunks del corpus.
        docs = vectorstore.similarity_search(query, k=5)

        if not docs:
            return "No se encontraron documentos relevantes en la base de conocimiento."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconocido")
            results.append(f"[Documento {i} - Fuente: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Error al buscar en la base de conocimiento: {str(e)}"
