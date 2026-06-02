"""
Módulo rag_tramites — Tests del grafo LangGraph.

Verifica que el grafo responde sin excepción y que
el historial crece en cada turno.
"""

from unittest.mock import patch

from backend.rag_tramites.graph.nodes import node_detect_municipio, node_guardrail

class TestNodeDetectMunicipio:
    """Tests del nodo de detección de municipio."""

    @patch("backend.rag_tramites.graph.nodes._cargar_municipios")
    def test_detecta_municipio(self, mock_cargar):
        mock_cargar.return_value = {"Adamuz", "Córdoba", "Baena"}
        state = {"query_actual": "empadronamiento en Adamuz"}
        result = node_detect_municipio(state)
        assert result["municipio_detectado"] == "Adamuz"

    @patch("backend.rag_tramites.graph.nodes._cargar_municipios")
    def test_no_detecta_municipio(self, mock_cargar):
        mock_cargar.return_value = {"Adamuz", "Córdoba"}
        state = {"query_actual": "trámite genérico"}
        result = node_detect_municipio(state)
        assert result["municipio_detectado"] is None

class TestNodeGuardrail:
    """Tests del nodo guardrail."""

    def test_sin_resultados(self):
        state = {"candidatos_reranked": [], "turno": 0, "respuesta_final": ""}
        result = node_guardrail(state)
        assert "No he encontrado" in result["respuesta_final"]
        assert result["necesita_refinamiento"] is False

    def test_con_resultados(self):
        state = {
            "candidatos_reranked": [{"metadata": {"url": "https://sede.es"}, "documento": "test"}],
            "turno": 1,
            "respuesta_final": "Aquí tienes info",
        }
        result = node_guardrail(state)
        assert result["respuesta_final"] == "Aquí tienes info"

    def test_limpia_urls_inventadas(self):
        state = {
            "candidatos_reranked": [{"metadata": {"url": "https://sede.es"}, "documento": "test"}],
            "turno": 0,
            "respuesta_final": "Visita https://inventada.com para más info",
        }
        result = node_guardrail(state)
        assert "https://inventada.com" not in result["respuesta_final"]
        assert "[URL no verificada]" in result["respuesta_final"]

    def test_muchos_turnos(self):
        state = {
            "candidatos_reranked": [{"metadata": {}, "documento": "test"}],
            "turno": 15,
            "respuesta_final": "Respuesta",
        }
        result = node_guardrail(state)
        assert result["necesita_refinamiento"] is True
