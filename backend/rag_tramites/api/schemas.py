"""
Módulo rag_tramites — Schemas Pydantic de request/response.
"""

from typing import Optional

from pydantic import BaseModel

class TramitesRequest(BaseModel):
    """Petición de chat sobre trámites."""
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class TramiteInfo(BaseModel):
    """Información resumida de un trámite."""
    nombre: str = ""
    municipio: str = ""
    url: str = ""
    auth: str = ""

class TramitesResponse(BaseModel):
    """Respuesta del chat de trámites."""
    respuesta: str
    tramites: list[TramiteInfo] = []
    session_id: str
