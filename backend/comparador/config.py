"""
Módulo comparador — Configuración de los dos LLMs.

Instancia dos ChatOpenAI apuntando a endpoints distintos para comparación
side-by-side. El segundo LLM es opcional: si `ALIA_LLM_URL_2` o
`ALIA_LLM_MODEL_2` no están definidos, `llm_b` queda a None y el router
debe responder 503 al invocar el endpoint.
"""

import os

from langchain_openai import ChatOpenAI

from backend.config import settings

def _model_display_name(raw: str | None) -> str:
    """Extrae nombre legible del modelo (sin ruta ni extensión)."""
    if not raw:
        return ""
    name = os.path.basename(raw)
    if name.endswith(".gguf"):
        name = name[:-5]
    return name

# Muestreo determinista: mismo (temperature, seed) que `node_generate` en
# `backend/rag_tramites/graph/nodes.py` para que el comparador y la tab
# Trámites converjan a la misma respuesta cuando ambos usan el mismo modelo.
llm_a = ChatOpenAI(
    base_url=settings.ALIA_LLM_URL,
    api_key=settings.ALIA_API_KEY,
    model=settings.ALIA_LLM_MODEL,
    streaming=True,
    max_tokens=1024,
    temperature=0,
    seed=42,
)
llm_a_name = _model_display_name(settings.ALIA_LLM_MODEL)

if settings.ALIA_LLM_URL_2 and settings.ALIA_LLM_MODEL_2:
    llm_b: ChatOpenAI | None = ChatOpenAI(
        base_url=settings.ALIA_LLM_URL_2,
        api_key=settings.ALIA_API_KEY,
        model=settings.ALIA_LLM_MODEL_2,
        streaming=True,
        max_tokens=1024,
        temperature=0,
        seed=42,
    )
    llm_b_name: str | None = _model_display_name(settings.ALIA_LLM_MODEL_2)
else:
    llm_b = None
    llm_b_name = None
