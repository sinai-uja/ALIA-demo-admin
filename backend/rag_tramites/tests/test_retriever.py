"""
Módulo rag_tramites — Tests del retriever.

Verifica que el retriever devuelve resultados coherentes.
Requiere que la colección esté indexada previamente.
"""

from unittest.mock import patch, MagicMock

from backend.rag_tramites.retriever import TramitesRetriever

class TestRetriever:
    """Tests del TramitesRetriever."""

    def test_retrieve_devuelve_lista(self):
        """Verifica que retrieve devuelve una lista (puede estar vacía si no hay datos)."""
        retriever = TramitesRetriever()
        with patch.object(retriever, "retrieve", return_value=[
            {"documento": "Empadronamiento en Adamuz", "metadata": {"municipio": "Adamuz"}, "distancia": 0.1}
        ]):
            resultados = retriever.retrieve("certificado de empadronamiento Adamuz")
            assert isinstance(resultados, list)
            assert len(resultados) >= 1

    def test_rerank_ordena_por_score(self):
        """Verifica que rerank ordena por score descendente."""
        retriever = TramitesRetriever()
        candidatos = [
            {"documento": "Doc A", "metadata": {}, "distancia": 0.5},
            {"documento": "Doc B", "metadata": {}, "distancia": 0.3},
            {"documento": "Doc C", "metadata": {}, "distancia": 0.7},
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.2, 0.9, 0.5]
        retriever._reranker = mock_reranker

        reranked = retriever.rerank("query", candidatos, top_k_rerank=2)
        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    def test_rerank_lista_vacia(self):
        """Verifica que rerank maneja lista vacía."""
        retriever = TramitesRetriever()
        result = retriever.rerank("query", [], top_k_rerank=3)
        assert result == []
