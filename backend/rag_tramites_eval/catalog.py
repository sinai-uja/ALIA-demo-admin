"""Loader del catálogo de trámites y utilidades de validación de URLs/municipios.

Parsea `data/export.json` con el mismo schema que el indexer
(`backend/rag_tramites/indexer.py`) y construye un índice en memoria
preparado para los scorers de `backend.rag_tramites_eval.scorers`.

Los `tramite_id` que se exponen aquí coinciden con
`metadata["id"]` que produce el indexer
(`slugify(f"{municipio}_{nombre}")[:64]`), de forma que el output del grafo
RAG (que lee esos mismos metadata desde ChromaDB) es directamente
comparable contra este catálogo sin transformaciones extra.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from slugify import slugify

from backend.config import settings

_URL_RE = re.compile(r"https?://\S+")
_URL_TRAILING_PUNCT = ".,;)]}"

@dataclass(frozen=True)
class TramiteEntry:
    tramite_id: str
    nombre: str
    municipio: str
    urls: frozenset[str]

@dataclass
class Catalog:
    """Catálogo indexado en memoria.

    `by_municipio[municipio_norm][tramite_id] = TramiteEntry`. La clave del
    primer nivel está normalizada (`_normalize_municipio`); el valor
    `TramiteEntry.municipio` conserva el nombre original para mostrar.
    """

    by_municipio: dict[str, dict[str, TramiteEntry]] = field(default_factory=dict)

    @property
    def municipios(self) -> set[str]:
        return {next(iter(t.values())).municipio for t in self.by_municipio.values() if t}

    def count_tramites(self) -> int:
        return sum(len(t) for t in self.by_municipio.values())

def _normalize_municipio(name: str) -> str:
    """Quita tildes, baja a minúsculas y colapsa espacios.

    Mantenemos los artículos ("el", "la") porque el catálogo real no usa el
    patrón "El Tal" en mayúsculas (cf. `data/export.json`).
    """
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_marks.lower().split())

def extract_urls(text: str) -> list[str]:
    """Extrae URLs http(s) de un texto, limpiando puntuación trailing.

    Mismo regex que `node_guardrail` en
    `backend/rag_tramites/graph/nodes.py:170`, con limpieza de puntuación
    final igual que `indexer._extraer_url` para que un URL acabado en `.`
    o `)` se compare bit-a-bit contra el catálogo.
    """
    if not text:
        return []
    raw = _URL_RE.findall(text)
    return [u.rstrip(_URL_TRAILING_PUNCT) for u in raw]

def _make_tramite_id(municipio: str, nombre: str) -> str:
    return slugify(f"{municipio}_{nombre}")[:64]

def _build_entry(municipio: str, nombre: str, texto: str) -> TramiteEntry:
    urls = frozenset(extract_urls(texto))
    return TramiteEntry(
        tramite_id=_make_tramite_id(municipio, nombre),
        nombre=nombre,
        municipio=municipio,
        urls=urls,
    )

def load_catalog(path: Optional[Path] = None) -> Catalog:
    """Carga el JSON y devuelve el catálogo indexado.

    Acepta el mismo shape que el indexer (`dict[municipio, dict[nombre, texto]]`).
    Si `path` es None, usa `settings.TRAMITES_DATA_PATH`.
    """
    src = Path(path) if path else Path(settings.TRAMITES_DATA_PATH)
    with src.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Schema no soportado en {src}: esperaba dict[municipio, dict[nombre, texto]]"
        )

    by_municipio: dict[str, dict[str, TramiteEntry]] = {}
    for municipio, tramites in raw.items():
        if not isinstance(tramites, dict):
            continue
        bucket: dict[str, TramiteEntry] = {}
        for nombre, texto in tramites.items():
            if isinstance(texto, dict):
                texto = json.dumps(texto, ensure_ascii=False)
            entry = _build_entry(municipio, nombre, str(texto))
            bucket[entry.tramite_id] = entry
        if bucket:
            by_municipio[_normalize_municipio(municipio)] = bucket

    return Catalog(by_municipio=by_municipio)

def url_belongs_to(
    catalog: Catalog,
    url: str,
    expected_municipio: Optional[str],
) -> bool:
    """¿La URL aparece en el catálogo y (si se especifica) en el municipio esperado?

    - `expected_municipio = None` o `""` → válida si pertenece a CUALQUIER
      municipio del catálogo. Útil para queries sin municipio. (El bootstrap
      del dataset MLflow convierte `None` a `""` porque MLflow Expectation
      no acepta `None` como `value`; este scorer trata ambos como equivalentes.)
    - `expected_municipio = "Adamuz"` → válida solo si pertenece a Adamuz
      (cross-municipio penaliza).
    """
    if not url:
        return False
    clean = url.rstrip(_URL_TRAILING_PUNCT)
    if not expected_municipio:  # None o ""
        return any(
            clean in entry.urls
            for bucket in catalog.by_municipio.values()
            for entry in bucket.values()
        )
    bucket = catalog.by_municipio.get(_normalize_municipio(expected_municipio))
    if not bucket:
        return False
    return any(clean in entry.urls for entry in bucket.values())

def municipios_normalizados(catalog: Catalog) -> set[str]:
    """Devuelve el set de municipios normalizados del catálogo."""
    return set(catalog.by_municipio.keys())
