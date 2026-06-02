"""
Módulo inference_logger — Router FastAPI para logs de inferencia.

Endpoints:
  GET    /logs       — Devuelve entries filtradas con paginación.
  GET    /logs/count — Número total de entries (para paginación).
  DELETE /logs       — Vacía el buffer.
"""

import logging
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter

from backend.inference_logger.service import inference_logger

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("")
async def get_logs(
    tab: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Devuelve las entries de log filtradas y paginadas."""
    entries = inference_logger.get_all(
        tab=tab, session_id=session_id, limit=limit, offset=offset
    )
    return {
        "entries": [asdict(e) for e in entries],
        "total": inference_logger.count(tab=tab, session_id=session_id),
        "limit": limit,
        "offset": offset,
    }

@router.get("/count")
async def get_logs_count(
    tab: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Devuelve el número total de entries."""
    return {"count": inference_logger.count(tab=tab, session_id=session_id)}

@router.delete("")
async def clear_logs():
    """Vacía el buffer de logs."""
    inference_logger.clear()
    return {"status": "ok", "message": "Logs vaciados"}
