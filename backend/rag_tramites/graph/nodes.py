"""
Módulo rag_tramites — Nodos del grafo LangGraph.

Cada nodo es una función pura (state) -> dict que actualiza
parcialmente el estado del grafo.
"""

import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config import settings
from backend.prompt_registry.fallback import TRAMITES_SYSTEM as _SYSTEM_PROMPT
from backend.prompt_registry.provider import get_provider
from backend.rag_tramites.retriever import TramitesRetriever
from backend.rag_tramites.vector_store import TramitesVectorStore

# Re-export para imports externos (e.g. `backend.comparador.router`):
# la fuente única de verdad del fallback empaquetado vive en
# `backend.prompt_registry.fallback.TRAMITES_SYSTEM`. Mantener este alias
# evita romper código antiguo que importe `_SYSTEM_PROMPT` desde aquí.
__all__ = [
    "_SYSTEM_PROMPT",
    "build_tramites_user_prompt",
    "node_detect_intent",
    "node_detect_municipio",
    "node_generate",
    "node_rerank",
    "node_retrieve",
    "node_retrieve_all",
]

logger = logging.getLogger(__name__)

# Singleton del retriever
_retriever = TramitesRetriever()

# Cache de municipios disponibles (se carga una vez)
_municipios_disponibles: Optional[set[str]] = None

# Patrones que indican intención de listado completo
_LISTADO_PATTERNS = [
    r"\btodos\s+(los\s+)?tr[aá]mites\b",
    r"\bqu[eé]\s+tr[aá]mites\b",
    r"\bcu[aá]les\s+(son\s+)?(los\s+)?tr[aá]mites\b",
    r"\blistado\b.*\btr[aá]mites\b",
    r"\btr[aá]mites\b.*\bdisponibles\b",
    r"\btr[aá]mites\b.*\bexisten\b",
    r"\btr[aá]mites\b.*\bofrec\b",
    r"\btr[aá]mites\b.*\btiene\b",
    r"\btr[aá]mites\b.*\bhay\b",
    r"\bcat[aá]logo\b.*\btr[aá]mites\b",
    r"\btr[aá]mites\b.*\bcat[aá]logo\b",
    r"\blistar\b.*\btr[aá]mites\b",
    r"\benum\w*\b.*\btr[aá]mites\b",
]

def _cargar_municipios() -> set[str]:
    """Carga la lista única de municipios desde ChromaDB."""
    global _municipios_disponibles
    if _municipios_disponibles is not None:
        return _municipios_disponibles

    try:
        collection = TramitesVectorStore.get_collection()
        result = collection.get(include=["metadatas"])
        if result and result.get("metadatas"):
            _municipios_disponibles = {
                m["municipio"] for m in result["metadatas"] if m.get("municipio")
            }
        else:
            _municipios_disponibles = set()
    except Exception as e:
        logger.warning(f"No se pudieron cargar municipios: {e}")
        _municipios_disponibles = set()

    logger.info(f"Municipios disponibles: {len(_municipios_disponibles)}")
    return _municipios_disponibles

def node_detect_municipio(state: dict) -> dict:
    """Detecta si la query menciona algún municipio del catálogo."""
    query = state.get("query_actual", "").lower()
    municipios = _cargar_municipios()

    detectado = None
    for municipio in municipios:
        if municipio.lower() in query:
            detectado = municipio
            break

    logger.info(f"Municipio detectado: {detectado} (query: '{query[:60]}')")
    return {"municipio_detectado": detectado}

def node_detect_intent(state: dict) -> dict:
    """Detecta si el usuario pide un listado completo o una consulta específica."""
    query = state.get("query_actual", "").lower()

    intent = "especifico"
    for pattern in _LISTADO_PATTERNS:
        if re.search(pattern, query):
            intent = "listado"
            break

    logger.info(f"Intent detectado: {intent} (query: '{query[:60]}')")
    return {"intent": intent}

def node_retrieve(state: dict) -> dict:
    """Recupera candidatos de ChromaDB (búsqueda semántica, top-k)."""
    query = state.get("query_actual", "")
    municipio = state.get("municipio_detectado")

    candidatos = _retriever.retrieve(query=query, municipio=municipio)
    logger.info(f"Candidatos recuperados: {len(candidatos)}")
    return {"candidatos": candidatos}

def node_retrieve_all(state: dict) -> dict:
    """Recupera TODOS los trámites de un municipio (sin límite ni rerank)."""
    municipio = state.get("municipio_detectado")

    if municipio:
        candidatos = TramitesVectorStore.query_all_by_municipio(municipio)
    else:
        # Sin municipio detectado, buscar por similitud con top-k alto
        query = state.get("query_actual", "")
        candidatos = _retriever.retrieve(query=query, municipio=None, top_k=20)

    logger.info(f"Retrieve ALL: {len(candidatos)} trámites")
    # En listado completo, los candidatos van directo a reranked (sin rerank)
    return {"candidatos": candidatos, "candidatos_reranked": candidatos}

def node_rerank(state: dict) -> dict:
    """Reordena candidatos con CrossEncoder."""
    query = state.get("query_actual", "")
    candidatos = state.get("candidatos", [])

    reranked = _retriever.rerank(query=query, candidatos=candidatos)
    logger.info(f"Candidatos tras rerank: {len(reranked)}")
    return {"candidatos_reranked": reranked}

def node_guardrail(state: dict) -> dict:
    """Aplica guardrails antes/después de la generación."""
    reranked = state.get("candidatos_reranked", [])
    turno = state.get("turno", 0)
    respuesta = state.get("respuesta_final", "")

    # Sin resultados → respuesta estándar
    if not reranked:
        return {
            "respuesta_final": (
                "No he encontrado trámites que coincidan con tu consulta. "
                "Intenta reformular tu pregunta o especifica el municipio."
            ),
            "necesita_refinamiento": False,
        }

    # Limpiar URLs inventadas de la respuesta
    if respuesta:
        urls_contexto = {r["metadata"].get("url", "") for r in reranked if r["metadata"].get("url")}
        urls_respuesta = set(re.findall(r"https?://\S+", respuesta))
        urls_inventadas = urls_respuesta - urls_contexto - {""}
        for url in urls_inventadas:
            respuesta = respuesta.replace(url, "[URL no verificada]")

    # Resumen del historial si hay demasiados turnos
    necesita_refinamiento = turno > 10

    return {
        "respuesta_final": respuesta,
        "necesita_refinamiento": necesita_refinamiento,
    }

def build_tramites_user_prompt(
    query: str,
    reranked: list[dict],
    intent: str,
    history: list | None = None,
) -> str:
    """Construye el user-prompt unificado para los pipelines de Trámites y Comparador.

    Fuente única de verdad: `node_generate` (graph) y `_build_prompt` (comparador)
    deben llamar a esta función con los mismos argumentos para producir el mismo
    string, garantizando paridad de prompt entre ambos endpoints (P0 del
    discovery `2026-05-08-tramites-eval-url/discovery-style-divergence.md`).

    Args:
        query: pregunta actual del usuario.
        reranked: candidatos rerankeados con metadata + documento.
        intent: "listado" | "especifico".
        history: turnos previos de la conversación (user/assistant), **sin
            incluir el turno actual**. None o lista vacía → no se inyecta
            bloque HISTORIAL RECIENTE. Sólo se incluye cuando hay al menos
            un turno previo real (resuelve el bug del HISTORIAL espurio en
            primer turno).

    el cuerpo carga la plantilla del prompt registry de MLflow vía
    `PromptProvider.get_prompt("uja-tramites-user-template")`. Si MLflow
    no responde, cae al fallback empaquetado en
    `backend/prompt_registry/fallback.py` (mismo contenido). El test
    `test_paridad_md5_con_provider_activado` garantiza que el output
    queda byte-igual al previo.
    """
    # CONTEXTO — render con dos modos según el shape de los reranked:
    #
    # 1. Si los chunks traen metadata `section`, agrupamos por `tramite_id`
    #    y mostramos la info común UNA vez (nombre/municipio/URL/auth)
    #    seguida de las secciones recuperadas como sub-bloques
    #    `── Seccion: ...`. El LLM ve el trámite estructurado y no procesa
    #    el mismo nombre/URL repetido por cada chunk.
    # 2. Si NO traen `section` (caso de fixtures unitarios o de un
    #    upstream que envía documentos enteros sin chunking), render
    #    plano de un bloque por trámite con Descripción truncada a
    #    4000 chars.
    contexto_parts: list[str] = []
    chunked_mode = any(r.get("metadata", {}).get("section") for r in reranked)

    if chunked_mode:
        # Agrupar por tramite_id preservando el orden de aparición.
        from collections import OrderedDict

        groups: "OrderedDict[str, list[dict]]" = OrderedDict()
        for r in reranked:
            tid = r.get("metadata", {}).get("id") or r.get("metadata", {}).get("nombre", "")
            groups.setdefault(tid, []).append(r)

        for i, (_tid, chunks) in enumerate(groups.items(), 1):
            head_meta = chunks[0]["metadata"]
            nombre = head_meta.get("nombre", "Sin nombre")
            municipio = head_meta.get("municipio", "")
            url = head_meta.get("url", "No disponible")
            auth = head_meta.get("auth", "No especificado")
            secciones_lines = []
            for chunk in chunks:
                seccion = chunk.get("metadata", {}).get("section") or "Contenido"
                # Quitar prefijo "{nombre} — {section}." que el indexer añade al
                # page_content para que el LLM no vea redundancia con el header.
                doc_raw = chunk["documento"]
                prefix = f"{nombre} — {seccion}."
                cuerpo = (
                    doc_raw[len(prefix):].lstrip()
                    if doc_raw.startswith(prefix)
                    else doc_raw
                )
                secciones_lines.append(f"  ── {seccion}: {cuerpo[:3000]}")
            secciones_block = "\n".join(secciones_lines)
            contexto_parts.append(
                f"[Trámite {i}] {nombre}\n"
                f"  Municipio: {municipio}\n"
                f"  URL: {url}\n"
                f"  Autenticación: {auth}\n"
                f"{secciones_block}"
            )
        contexto_value = (
            f"CONTEXTO ({len(groups)} trámites encontrados, "
            f"{len(reranked)} secciones):\n" + "\n\n".join(contexto_parts)
        )
    else:
        # Render histórico (fixtures pre-chunking + paridad MD5).
        for i, r in enumerate(reranked, 1):
            meta = r["metadata"]
            nombre = meta.get("nombre", "Sin nombre")
            url = meta.get("url", "No disponible")
            auth = meta.get("auth", "No especificado")
            municipio = meta.get("municipio", "")
            contexto_parts.append(
                f"[Trámite {i}] {nombre}\n"
                f"  Municipio: {municipio}\n"
                f"  URL: {url}\n"
                f"  Autenticación: {auth}\n"
                f"  Descripción: {r['documento'][:4000]}"
            )
        contexto_value = (
            f"CONTEXTO ({len(reranked)} trámites encontrados):\n"
            + "\n\n".join(contexto_parts)
        )

    # Instrucción según intent — incluye `\n` leading + `\n\n` trailing para
    # que, al insertarse entre el bloque contexto y el de PREGUNTA ACTUAL,
    # reproduzca el espaciado previo.
    instruccion_intent_value = ""
    if intent == "listado":
        instruccion_intent_value = (
            "\nIMPORTANTE: El usuario quiere un listado COMPLETO. "
            f"En el contexto hay {len(reranked)} trámites. "
            "Enuméralos TODOS sin omitir ninguno, incluyendo nombre y URL de cada uno.\n\n"
        )

    # HISTORIAL — incluye header "HISTORIAL RECIENTE:" + `\n\n` trailing
    # cuando hay turnos reales.
    historial_value = ""
    if history:
        ultimos = history[-4:]
        partes = []
        for m in ultimos:
            if isinstance(m, dict):
                role = m.get("role", "unknown")
                content = m.get("content", "")
            elif hasattr(m, "type"):
                role = "user" if m.type == "human" else "assistant"
                content = m.content
            else:
                continue
            partes.append(f"{role}: {content}")
        if partes:
            historial_value = "HISTORIAL RECIENTE:\n" + "\n".join(partes) + "\n\n"

    # Carga la plantilla del Prompt Registry (MLflow) o cae al empaquetado.
    resource = get_provider().get_prompt("uja-tramites-user-template")
    return resource.format(
        contexto=contexto_value,
        instruccion_intent=instruccion_intent_value,
        historial=historial_value,
        query=query,
    )

async def node_generate(state: dict) -> dict:
    """Genera respuesta usando el LLM local vía LM Studio."""
    reranked = state.get("candidatos_reranked", [])
    query = state.get("query_actual", "")
    intent = state.get("intent", "especifico")
    messages_hist = state.get("messages", []) or []

    # `messages` siempre lleva el turno actual como último user message.
    # Eliminarlo antes de construir el HISTORIAL evita duplicar la query
    # (bug detectado en discovery-style-divergence.md).
    prior_turns = messages_hist[:-1] if messages_hist else []

    user_prompt = build_tramites_user_prompt(
        query=query,
        reranked=reranked,
        intent=intent,
        history=prior_turns,
    )

    # Muestreo determinista: temperature=0 + seed fija para reproducibilidad
    # entre Trámites y Comparador, y para que la evaluación de  mida
    # calidad del RAG (no varianza del muestreo).
    llm = ChatOpenAI(
        base_url=settings.ALIA_LLM_URL,
        api_key=settings.ALIA_API_KEY,
        model=settings.ALIA_LLM_MODEL,
        streaming=True,
        max_tokens=1024,
        temperature=0,
        seed=42,
    )

    # carga el system prompt del registry. Fallback al empaquetado
    # (`_SYSTEM_PROMPT` = `TRAMITES_SYSTEM`) si MLflow no responde.
    system_resource = get_provider().get_prompt("uja-tramites-system")
    response = await llm.ainvoke([
        SystemMessage(content=system_resource.template),
        HumanMessage(content=user_prompt),
    ])

    logger.info(f"Respuesta generada: {len(response.content)} chars (intent={intent})")
    return {
        "respuesta_final": response.content,
        "turno": state.get("turno", 0) + 1,
    }
