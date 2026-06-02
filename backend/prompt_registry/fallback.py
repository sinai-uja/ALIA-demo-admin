"""Constantes empaquetadas de los prompts (fallback)

Estas constantes son la versión "última conocida buena" de cada prompt usado
por los nodos LangGraph. Se sirven desde aquí en dos casos:

1. Cuando `PROMPT_REGISTRY_ENABLED=False`, el provider devuelve siempre estas
   constantes sin consultar MLflow.
2. Cuando MLflow no responde o el prompt referenciado no existe, el provider
   cae a estas constantes con un `logging.WARNING`.

La paridad bit-exacta Trámites ↔ Comparador A depende de que `TRAMITES_SYSTEM` y
`TRAMITES_USER_TEMPLATE`, al usarse desde el provider con el wrapper
refactorizado de `build_tramites_user_prompt` (tarea 3.2), produzcan
exactamente el mismo string que la implementación previo. El test
extendido `backend/rag_tramites/tests/test_prompt_builder.py` verifica el MD5
`850d013f82162d888aa8176638000c1d`.

"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Trámites — compartido entre tab Trámites y Comparador columna A.
# ---------------------------------------------------------------------------
# Copia literal de `_SYSTEM_PROMPT` en backend/rag_tramites/graph/nodes.py
# (estado post-Mantenimiento 2026-05-11). NO modificar a mano: cualquier cambio
# debe pasar primero por el Prompt Registry de MLflow y luego sincronizarse
# aquí para mantener el fallback al día.
TRAMITES_SYSTEM = (
    "Eres un asistente especializado en trámites administrativos municipales de Andalucía.\n\n"
    "REGLAS:\n"
    "- Responde SIEMPRE en español. Sé conciso y directo.\n"
    "- Si el usuario pide un LISTADO de trámites de un municipio, enumera TODOS "
    "los que aparecen en el contexto. No omitas ninguno.\n"
    "- Si el usuario pregunta por un trámite ESPECÍFICO, céntrate en ese trámite "
    "y da los detalles (URL, autenticación, descripción).\n"
    "- Si no encuentras el trámite exacto, indícalo claramente y sugiere "
    "trámites similares del contexto.\n"
    "- Incluye siempre la URL de acceso cuando esté disponible.\n"
    "- NO inventes trámites ni URLs que no estén en el contexto.\n"
    "- NO inventes pasos, requisitos, plazos ni documentos exigidos: si el "
    "contexto no los detalla, indica que esa información no está disponible "
    "en el catálogo y remite al enlace oficial del trámite.\n"
    "- NO inventes referencias cruzadas a otros trámites (\"ver trámite X\") "
    "salvo que ese trámite esté literalmente en el contexto.\n"
    "- NO mezcles datos entre trámites distintos: cuando cites un trámite, "
    "todos sus campos (nombre, URL, ID, autenticación, descripción) deben "
    "provenir del MISMO bloque [Trámite N] del contexto.\n"
    "\n"
    "DETECCIÓN DE PROCEDIMIENTOS NO CUBIERTOS:\n"
    "- Cuando la pregunta del usuario describe un PROCEDIMIENTO AMPLIO "
    "(p. ej. \"cómo me empadrono\", \"cómo me caso\", \"cómo abro un negocio\", "
    "\"cómo solicito una vivienda social\", \"cómo me doy de alta\"), verifica "
    "si los trámites del CONTEXTO satisfacen literalmente esa pregunta o si "
    "solo cubren PRODUCTOS DERIVADOS del procedimiento (volantes, "
    "certificados, comprobantes, edictos, registros, justificantes).\n"
    "- Si el contexto solo contiene productos derivados, COMIENZA tu "
    "respuesta avisando explícitamente al usuario de que el catálogo NO "
    "incluye el procedimiento completo y REMITE al Ayuntamiento o a la "
    "sede electrónica oficial del municipio.\n"
    "- Tras ese aviso puedes listar los trámites derivados del contexto "
    "como información complementaria, dejando claro que NO son el "
    "procedimiento principal."
)

# Plantilla del user prompt (4 variables: contexto, instruccion_intent,
# historial, query). El caller — `build_tramites_user_prompt` refactorizado
# en tarea 3.2 — construye cada valor con el mismo formato que la versión
# pre-#
#  - `contexto`           = "CONTEXTO ({N} trámites encontrados):\n{items}"
#                           (sin saltos trailing — el template añade `\n\n`).
#  - `instruccion_intent` = "" si intent ≠ "listado",
#                           o el bloque IMPORTANTE (`"\nIMPORTANTE: ...\n\n"`)
#                           si intent == "listado".
#  - `historial`          = "" si no hay turnos previos,
#                           o `"HISTORIAL RECIENTE:\n{turnos}\n\n"` cuando los hay.
#  - `query`              = pregunta del usuario.
#
# Esta composición está cubierta por un test de paridad bit-exacta
# (MD5 `850d013f…`) entre los pipelines de Trámites y Comparador A.
TRAMITES_USER_TEMPLATE = (
    "{{contexto}}\n\n"
    "{{instruccion_intent}}{{historial}}"
    "PREGUNTA ACTUAL:\n{{query}}"
)

# ---------------------------------------------------------------------------
# RAG docs (tab2bis_rag).
# ---------------------------------------------------------------------------
# La instrucción declarativa va en SystemMessage para que sea editable
# independientemente del bloque de contexto desde la UI de MLflow.
RAG_DOCS_SYSTEM = (
    "A continuación tienes información de una base de conocimiento. "
    "Usa ÚNICAMENTE esta información para responder mi pregunta. "
    "Si la información no es suficiente, dilo claramente."
)

# Plantilla del user prompt RAG docs (2 variables: contexto, query).
RAG_DOCS_USER_TEMPLATE = (
    "--- INFORMACIÓN ---\n"
    "{{contexto}}\n"
    "--- FIN INFORMACIÓN ---\n\n"
    "Mi pregunta es: {{query}}"
)

# ---------------------------------------------------------------------------
# Lookup público — consumido por `PromptProvider._build_fallback`.
# ---------------------------------------------------------------------------

# NOTA — convención de naming: MLflow rechaza nombres con `/` interior porque
# las URIs `prompts:/<name>@<alias>` reservan el slash para separar
# `name/version`. Usamos hyphens para emular la jerarquía
# `uja-<dominio>-<rol>` sin colisionar con el parser de URIs.
_FALLBACK_MAP: dict[str, str] = {
    "uja-tramites-system": TRAMITES_SYSTEM,
    "uja-tramites-user-template": TRAMITES_USER_TEMPLATE,
    "uja-rag-docs-system": RAG_DOCS_SYSTEM,
    "uja-rag-docs-user-template": RAG_DOCS_USER_TEMPLATE,
}

def get_fallback_prompt(name: str) -> Optional[str]:
    """Devuelve la plantilla empaquetada para `name`, o `None` si no soporta.

    El provider levanta `KeyError` si recibe `None`: siempre debe existir un
    fallback para los prompts registrados en `_FALLBACK_MAP`. Devolver `None`
    es la forma de indicar "este nombre no está soportado por esta fase".
    """
    return _FALLBACK_MAP.get(name)
