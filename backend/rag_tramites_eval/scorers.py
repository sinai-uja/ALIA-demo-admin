"""Scorers custom para evaluación del RAG de trámites.

Cuatro funciones puras y deterministas (sin red, sin LLM, sin estado global).
Cada una devuelve `float ∈ [0, 1]` para que `mlflow.genai.evaluate` las
agregue de forma uniforme.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from backend.rag_tramites_eval.catalog import (
    Catalog,
    _normalize_municipio,
    extract_urls,
    url_belongs_to,
)

# ---------------------------------------------------------------------------
# 1. Recall de trámites
# ---------------------------------------------------------------------------

def recall_tramites(
    expected_ids: Iterable[str],
    retrieved_ids: Iterable[str],
) -> float:
    """Fracción de trámites esperados que aparecen en los recuperados.

    Convención: si `expected_ids` está vacío (caso "no había nada que
    recuperar"), devuelve 1.0 — el recall es perfecto por vacuidad. Quien
    quiera penalizar ese caso debe usar otro scorer (p. ej. precision).
    """
    expected = {e for e in expected_ids if e}
    if not expected:
        return 1.0
    retrieved = {r for r in retrieved_ids if r}
    hits = expected & retrieved
    return len(hits) / len(expected)

# ---------------------------------------------------------------------------
# 2. Precisión de URLs en la respuesta final
# ---------------------------------------------------------------------------

def precision_urls(
    response_text: str,
    expected_municipio: Optional[str],
    catalog: Catalog,
) -> float:
    """Fracción de URLs de la respuesta que pertenecen al catálogo + municipio.

    - URL del catálogo + municipio correcto → cuenta como acierto.
    - URL inventada o cross-municipio → cuenta como fallo.
    - Sin URLs en la respuesta → 1.0 (sin penalización; el LLM puede legítimamente
      contestar sin enlaces). Recall de URLs es problema de otro scorer.
    """
    urls = extract_urls(response_text)
    if not urls:
        return 1.0
    valid = sum(1 for u in urls if url_belongs_to(catalog, u, expected_municipio))
    return valid / len(urls)

# ---------------------------------------------------------------------------
# 3. Detección correcta de municipio
# ---------------------------------------------------------------------------

def municipio_match(
    extracted: Optional[str],
    expected: Optional[str],
) -> float:
    """1.0 si la detección del nodo `node_detect_municipio` coincide con lo esperado.

    Tabla de verdad (tras normalizar con NFKD + lowercase):

    | extracted | expected | resultado |
    | --------- | -------- | --------- |
    | None      | None     | 1.0       |  (query sin municipio, detectó None)
    | None      | "Adamuz" | 0.0       |  (falso negativo)
    | "Adamuz"  | None     | 0.0       |  (falso positivo)
    | "Adamuz"  | "adamuz" | 1.0       |  (match con normalización)
    | "Adamuz"  | "Cabra"  | 0.0       |  (municipio incorrecto)
    """
    norm_extracted = _normalize_municipio(extracted) if extracted else None
    norm_expected = _normalize_municipio(expected) if expected else None
    if norm_extracted == norm_expected:
        return 1.0
    return 0.0

# ---------------------------------------------------------------------------
# 4. Scope mismatch correcto (regla DETECCIÓN DE PROCEDIMIENTOS NO CUBIERTOS)
# ---------------------------------------------------------------------------

# Patrones que indican que el modelo emitió el disclaimer al inicio de la
# respuesta. Vienen de la sección "DETECCIÓN DE PROCEDIMIENTOS NO CUBIERTOS"
# del system prompt (`backend/prompt_registry/fallback.py::TRAMITES_SYSTEM`):
# "el catálogo NO incluye … REMITE al Ayuntamiento o a la sede electrónica".
_DISCLAIMER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)cat[aá]logo\s+\w*\s*(no|sin)\s+(incluye|cubre|contiene|recoge|tiene)"),
    re.compile(r"(?i)(no|sin)\s+(incluye|cubre|contiene|recoge)\s+.*?cat[aá]logo"),
    re.compile(r"(?i)procedimiento\s+completo\s+no\s+est[aá]\s+(en\s+)?(el\s+)?cat[aá]logo"),
    re.compile(r"(?i)sede\s+electr[oó]nica"),
    re.compile(r"(?i)remit\w+\s+al\s+ayuntamiento"),
    re.compile(r"(?i)acud\w+\s+al\s+ayuntamiento"),
)

# Solo cuentan disclaimers que aparecen al PRINCIPIO de la respuesta
# (el prompt dice "COMIENZA tu respuesta avisando…"). Si aparece a mitad
# o al final, es contexto adicional, no la regla cumplida.
_DISCLAIMER_HEAD_CHARS = 600

def _has_disclaimer(response_text: str) -> bool:
    head = (response_text or "")[:_DISCLAIMER_HEAD_CHARS]
    return any(p.search(head) for p in _DISCLAIMER_PATTERNS)

def scope_mismatch_correct(
    response_text: str,
    expected_scope: str,
) -> float:
    """Verifica si el LLM aplicó (o no) la regla de procedimiento no cubierto.

    `expected_scope`:
      - `"out"` → la pregunta era un procedimiento NO cubierto; se espera
        disclaimer al inicio. 1.0 si aparece, 0.0 si no.
      - `"in"` → la pregunta sí está cubierta; el disclaimer sería un falso
        positivo. 1.0 si NO aparece, 0.0 si aparece indebidamente.

    Cualquier otro valor levanta `ValueError` (no se interpreta silenciosamente).
    """
    if expected_scope not in {"in", "out"}:
        raise ValueError(
            f"expected_scope debe ser 'in' u 'out', recibido: {expected_scope!r}"
        )
    has = _has_disclaimer(response_text)
    if expected_scope == "out":
        return 1.0 if has else 0.0
    return 0.0 if has else 1.0
