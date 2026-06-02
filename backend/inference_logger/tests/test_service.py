"""
Tests del servicio InferenceLogger.
"""

from backend.inference_logger.models import LogEntry
from backend.inference_logger.service import InferenceLogger

class TestInferenceLogger:
    """Tests del buffer in-memory de logs de inferencia."""

    def _make_entry(self, tab: str = "chatbot", session_id: str = "s1", query: str = "hola") -> LogEntry:
        return LogEntry(tab=tab, session_id=session_id, query=query, response="respuesta")

    def test_add_and_retrieve(self):
        logger = InferenceLogger()
        entry = self._make_entry()
        logger.add(entry)
        result = logger.get_all()
        assert len(result) == 1
        assert result[0].query == "hola"

    def test_max_entries_fifo(self):
        logger = InferenceLogger(max_entries=100)
        for i in range(110):
            logger.add(self._make_entry(query=f"q{i}"))
        assert logger.count() == 100
        # La más reciente debe ser q109
        entries = logger.get_all(limit=1)
        assert entries[0].query == "q109"
        # La más antigua debe ser q10 (las primeras 10 se descartaron)
        entries = logger.get_all(limit=100)
        assert entries[-1].query == "q10"

    def test_filter_by_tab(self):
        logger = InferenceLogger()
        logger.add(self._make_entry(tab="chatbot"))
        logger.add(self._make_entry(tab="tramites"))
        logger.add(self._make_entry(tab="chatbot"))
        result = logger.get_all(tab="chatbot")
        assert len(result) == 2
        assert all(e.tab == "chatbot" for e in result)

    def test_filter_by_session_id(self):
        logger = InferenceLogger()
        logger.add(self._make_entry(session_id="s1"))
        logger.add(self._make_entry(session_id="s2"))
        logger.add(self._make_entry(session_id="s1"))
        result = logger.get_all(session_id="s1")
        assert len(result) == 2

    def test_clear(self):
        logger = InferenceLogger()
        logger.add(self._make_entry())
        logger.add(self._make_entry())
        assert logger.count() == 2
        logger.clear()
        assert logger.count() == 0
        assert logger.get_all() == []

    def test_pagination(self):
        logger = InferenceLogger()
        for i in range(10):
            logger.add(self._make_entry(query=f"q{i}"))
        # Página 1: las 3 más recientes
        page1 = logger.get_all(limit=3, offset=0)
        assert len(page1) == 3
        assert page1[0].query == "q9"
        # Página 2
        page2 = logger.get_all(limit=3, offset=3)
        assert len(page2) == 3
        assert page2[0].query == "q6"

    def test_count_with_filters(self):
        logger = InferenceLogger()
        logger.add(self._make_entry(tab="chatbot"))
        logger.add(self._make_entry(tab="tramites"))
        logger.add(self._make_entry(tab="tramites"))
        assert logger.count() == 3
        assert logger.count(tab="tramites") == 2
        assert logger.count(tab="rag") == 0

    def test_order_most_recent_first(self):
        logger = InferenceLogger()
        logger.add(self._make_entry(query="first"))
        logger.add(self._make_entry(query="second"))
        logger.add(self._make_entry(query="third"))
        result = logger.get_all()
        assert result[0].query == "third"
        assert result[2].query == "first"
