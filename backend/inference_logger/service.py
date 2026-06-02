"""
Módulo inference_logger — Servicio centralizado de logging de inferencias.

Almacena LogEntry en memoria (FIFO, máximo 100 entries).
Singleton importable: `from backend.inference_logger.service import inference_logger`.
"""

import logging
import threading
from typing import Optional

from backend.inference_logger.models import LogEntry

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 100

class InferenceLogger:
    """Buffer in-memory de registros de inferencia LLM."""

    def __init__(self, max_entries: int = _MAX_ENTRIES):
        self._max_entries = max_entries
        self._entries: list[LogEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: LogEntry) -> None:
        """Añade una entrada al buffer. Descarta la más antigua si se supera el límite."""
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
        logger.debug(f"LogEntry añadido: tab={entry.tab}, query='{entry.query[:40]}...'")

    def get_all(
        self,
        tab: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LogEntry]:
        """Devuelve entries filtradas, ordenadas de más reciente a más antigua."""
        with self._lock:
            entries = list(self._entries)

        # Filtrar
        if tab:
            entries = [e for e in entries if e.tab == tab]
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]

        # Ordenar más reciente primero
        entries.reverse()

        # Paginar
        return entries[offset:offset + limit]

    def count(self, tab: Optional[str] = None, session_id: Optional[str] = None) -> int:
        """Devuelve el número de entries (con filtros opcionales)."""
        with self._lock:
            entries = list(self._entries)
        if tab:
            entries = [e for e in entries if e.tab == tab]
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        return len(entries)

    def clear(self) -> None:
        """Vacía el buffer."""
        with self._lock:
            self._entries.clear()
        logger.info("Buffer de logs vaciado")

# Singleton global
inference_logger = InferenceLogger()
