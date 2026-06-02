"""
Módulo rag_tramites — Retriever con encoder y reranker.

Combina búsqueda vectorial (HuggingFaceEmbeddings/MrBERT) con
reranking mediante CrossEncoder para mejorar la precisión.
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from backend.config import settings
from backend.rag_tramites.vector_store import TramitesVectorStore

logger = logging.getLogger(__name__)

class TramitesRetriever:
    """Retriever con búsqueda vectorial + reranking."""

    def __init__(self):
        self._reranker: Optional[CrossEncoder] = None

    def _get_reranker(self) -> CrossEncoder:
        """Carga lazy del modelo de reranking."""
        if self._reranker is None:
            logger.info(f"Cargando reranker: {settings.TRAMITES_RERANKER_PATH}")
            self._reranker = CrossEncoder(settings.TRAMITES_RERANKER_PATH)
        return self._reranker

    def retrieve(
        self,
        query: str,
        municipio: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Busca trámites por similitud semántica en ChromaDB.

        Args:
            query: Consulta del usuario.
            municipio: Filtro opcional por municipio.
            top_k: Número de candidatos a recuperar.

        Returns:
            Lista de candidatos con documento, metadata y distancia.
        """
        return TramitesVectorStore.query(
            query=query,
            municipio=municipio,
            top_k=top_k or settings.TRAMITES_TOP_K,
        )

    def rerank(
        self,
        query: str,
        candidatos: list[dict],
        top_k_rerank: Optional[int] = None,
        max_unique_tramites: Optional[int] = None,
    ) -> list[dict]:
        """Reordena candidatos usando CrossEncoder y agrupa por trámite.

        Tras puntuar con el cross-encoder, agrupa los chunks por
        `tramite_id` (metadata["id"]) preservando el orden interno
        original (chunk_index) para reconstruir la presentación
        coherente del trámite. Limita el resultado a
        `max_unique_tramites` trámites distintos para no saturar el
        contexto del LLM.

        Args:
            query: Consulta original del usuario.
            candidatos: Lista de candidatos del retrieve (puede contener
                varios chunks del mismo trámite).
            top_k_rerank: Número de chunks que el cross-encoder considera
                "ganadores" antes de la agrupación.
            max_unique_tramites: Número máximo de trámites únicos en el
                resultado final.

        Returns:
            Lista de candidatos ordenada por relevancia del trámite (el
            chunk top de cada trámite marca su posición) y, dentro de
            cada trámite, ordenada por chunk_index. Cada candidato lleva
            su `rerank_score`; el mejor score del grupo se replica en
            `group_score` para depuración.
        """
        if not candidatos:
            return []

        k = top_k_rerank or settings.TRAMITES_TOP_K_RERANK
        max_unique = max_unique_tramites or settings.TRAMITES_MAX_UNIQUE_TRAMITES
        reranker = self._get_reranker()

        pairs = [(query, c["documento"]) for c in candidatos]
        scores = reranker.predict(pairs)
        for candidato, score in zip(candidatos, scores):
            candidato["rerank_score"] = float(score)

        ordered = sorted(candidatos, key=lambda x: x["rerank_score"], reverse=True)
        top_chunks = ordered[:k]

        # Agrupar por tramite_id preservando el orden de aparición del
        # mejor chunk de cada trámite (el primero que vimos = más relevante).
        groups: dict[str, list[dict]] = {}
        group_top_score: dict[str, float] = {}
        for c in top_chunks:
            tid = c.get("metadata", {}).get("id") or ""
            if tid not in groups:
                groups[tid] = []
                group_top_score[tid] = c["rerank_score"]
            groups[tid].append(c)

        # Limitar a max_unique trámites (el orden ya está por relevancia).
        selected_ids = list(groups.keys())[:max_unique]

        # Reconstruir lista plana: por trámite, chunks ordenados por
        # chunk_index para presentación coherente.
        final: list[dict] = []
        for tid in selected_ids:
            chunks_of_tramite = sorted(
                groups[tid],
                key=lambda c: c.get("metadata", {}).get("chunk_index", 0),
            )
            top_score = group_top_score[tid]
            for chunk in chunks_of_tramite:
                chunk["group_score"] = top_score
                final.append(chunk)

        logger.info(
            f"Rerank: {len(candidatos)} candidatos → top {k} chunks → "
            f"{len(selected_ids)} trámites únicos ({len(final)} chunks finales) | "
            f"top group scores: {[f'{group_top_score[t]:.3f}' for t in selected_ids]}"
        )

        return final
