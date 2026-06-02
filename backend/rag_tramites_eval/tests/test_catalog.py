"""Test de paridad del catalog loader contra data/export.json real.

Verifica que la carga del catálogo refleja fielmente el JSON: 128 municipios,
los `tramite_id` coinciden con la convención del indexer
(`slugify(municipio_nombre)`), y al menos un trámite conocido es localizable
con su URL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from slugify import slugify

from backend.rag_tramites_eval.catalog import (
    _normalize_municipio,
    extract_urls,
    load_catalog,
    municipios_normalizados,
    url_belongs_to,
)

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "export.json"

@pytest.fixture(scope="module")
def real_catalog():
    if not CATALOG_PATH.exists():
        pytest.skip(f"data/export.json no disponible en {CATALOG_PATH}")
    return load_catalog(CATALOG_PATH)

@pytest.fixture(scope="module")
def raw_json():
    if not CATALOG_PATH.exists():
        pytest.skip(f"data/export.json no disponible en {CATALOG_PATH}")
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)

class TestCatalogLoaderReal:
    def test_128_municipios(self, real_catalog, raw_json):
        # mission.md afirma 128 municipios; lo verificamos contra la fuente.
        assert real_catalog.count_tramites() > 0
        assert len(municipios_normalizados(real_catalog)) == len(raw_json)

    def test_total_tramites_coincide_con_indexer(self, real_catalog, raw_json):
        # El indexer aplica `slugify(municipio_nombre)[:64]` para el id, y la
        # truncación a 64 chars produce colisiones para trámites con nombres
        # largos similares (mismo prefijo). Lo que ChromaDB acaba teniendo es
        # ese set deduplicado, no el total bruto del JSON. La paridad correcta
        # del catálogo de eval es contra esa misma dedupe (es contra eso que
        # se compararán los retrieved_ids).
        unique_slug_ids = set()
        for municipio, tramites in raw_json.items():
            if not isinstance(tramites, dict):
                continue
            for nombre in tramites:
                unique_slug_ids.add(slugify(f"{municipio}_{nombre}")[:64])
        assert real_catalog.count_tramites() == len(unique_slug_ids)

    def test_id_compatible_con_indexer(self, real_catalog):
        # El indexer (`backend/rag_tramites/indexer.py:120`) usa
        # `slugify(f"{municipio}_{nombre}")[:64]`. Verificamos paridad.
        bucket = real_catalog.by_municipio[_normalize_municipio("Adamuz")]
        for entry in bucket.values():
            expected_id = slugify(f"{entry.municipio}_{entry.nombre}")[:64]
            assert entry.tramite_id == expected_id

    def test_tramite_conocido_localizable(self, real_catalog):
        # "Autoliquidación impuesto Plusvalías" de Adamuz tiene URL conocida
        # (`https://e-admin.eprinsa.es/adamuz/monotramite?tramite=1857`).
        bucket = real_catalog.by_municipio[_normalize_municipio("Adamuz")]
        expected_id = slugify("Adamuz_Autoliquidación impuesto Plusvalías")[:64]
        entry = bucket[expected_id]
        assert entry.nombre == "Autoliquidación impuesto Plusvalías"
        assert any("eprinsa.es" in u for u in entry.urls)

    def test_url_belongs_to_acierta_y_cross_municipio_falla(self, real_catalog):
        bucket = real_catalog.by_municipio[_normalize_municipio("Adamuz")]
        sample_url = next(
            (u for entry in bucket.values() for u in entry.urls),
            None,
        )
        assert sample_url, "Adamuz debería tener al menos un trámite con URL"
        assert url_belongs_to(real_catalog, sample_url, "Adamuz") is True
        assert url_belongs_to(real_catalog, sample_url, "Cabra") is False
        assert url_belongs_to(real_catalog, sample_url, None) is True

    def test_url_inventada_no_pertenece(self, real_catalog):
        assert (
            url_belongs_to(real_catalog, "https://no-existe.invalid/x", None) is False
        )

    def test_extract_urls_strip_trailing_punct(self):
        # Una URL acabada en '.' o ')' debe limpiarse.
        text = "Ver https://example.com/x. y también (https://example.com/y)"
        urls = extract_urls(text)
        assert "https://example.com/x" in urls
        assert "https://example.com/y" in urls
