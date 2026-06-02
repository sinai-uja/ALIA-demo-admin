"""
Módulo rag_tramites — Abstracción de ChromaDB para trámites.

Provee acceso singleton a la colección de trámites en ChromaDB
y métodos de consulta con filtro opcional por municipio.
"""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)

_vectorstore: Optional[Chroma] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el singleton de embeddings (MrBERT)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

class TramitesVectorStore:
    """Acceso a la colección de trámites en ChromaDB."""

    @staticmethod
    def get_collection() -> Chroma:
        """Devuelve la colección ChromaDB (singleton)."""
        global _vectorstore
        if _vectorstore is None:
            _vectorstore = Chroma(
                collection_name=settings.TRAMITES_CHROMA_COLLECTION,
                persist_directory=settings.CHROMA_PERSIST_DIR,
                embedding_function=_get_embeddings(),
            )
            logger.info(
                f"Colección ChromaDB cargada: {settings.TRAMITES_CHROMA_COLLECTION}"
            )
        return _vectorstore

    @staticmethod
    def query(
        query: str,
        municipio: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Busca trámites por similitud semántica.

        Args:
            query: Consulta en lenguaje natural.
            municipio: Filtro opcional por municipio.
            top_k: Número de resultados (por defecto TRAMITES_TOP_K).

        Returns:
            Lista de dicts con 'documento', 'metadata' y 'distancia'.
        """
        k = top_k or settings.TRAMITES_TOP_K
        collection = TramitesVectorStore.get_collection()

        search_kwargs = {"k": k}
        if municipio:
            search_kwargs["filter"] = {"municipio": municipio}

        docs_with_scores = collection.similarity_search_with_score(
            query, **search_kwargs
        )

        results = []
        for doc, score in docs_with_scores:
            results.append({
                "documento": doc.page_content,
                "metadata": doc.metadata,
                "distancia": score,
            })

        logger.info(
            f"Query: '{query[:50]}...' | municipio={municipio} | "
            f"resultados={len(results)}"
        )
        return results

    @staticmethod
    def query_all_by_municipio(municipio: str) -> list[dict]:
        """Recupera TODOS los trámites de un municipio sin límite.

        Args:
            municipio: Nombre del municipio.

        Returns:
            Lista de dicts con 'documento', 'metadata' y 'distancia'.
        """
        collection = TramitesVectorStore.get_collection()

        # Obtener todos los documentos filtrados por municipio
        result = collection.get(
            where={"municipio": municipio},
            include=["documents", "metadatas"],
        )

        results = []
        if result and result.get("documents"):
            for doc, meta in zip(result["documents"], result["metadatas"]):
                results.append({
                    "documento": doc,
                    "metadata": meta,
                    "distancia": 0.0,
                })

        logger.info(
            f"Query ALL: municipio={municipio} | resultados={len(results)}"
        )
        return results
