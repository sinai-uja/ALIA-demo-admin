"""Tests de regresión del seed file y del transform a records MLflow.

Verifica el contrato del seed (count + schema + distribución mínima) y la
conversión a records `inputs/expectations/tags` sin tocar MLflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.rag_tramites_eval.catalog import load_catalog
from backend.rag_tramites_eval.seed_queries import EXPECTED_ALL_TRAMITES, SEED_QUERIES
from scripts.bootstrap_eval_dataset import _expand_expected_ids, _seed_to_record

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "export.json"

@pytest.fixture(scope="module")
def real_catalog():
    if not CATALOG_PATH.exists():
        pytest.skip(f"data/export.json no disponible en {CATALOG_PATH}")
    return load_catalog(CATALOG_PATH)

REQUIRED_KEYS = {
    "query",
    "expected_municipio",
    "expected_tramite_ids",
    "expected_response",
    "expected_scope",
}

class TestSeedSchema:
    def test_count_en_rango(self):
        # plan.md §2.1: 20 ≤ N ≤ 30.
        assert 20 <= len(SEED_QUERIES) <= 30

    def test_campos_requeridos(self):
        for i, seed in enumerate(SEED_QUERIES):
            assert REQUIRED_KEYS.issubset(seed), (
                f"seed #{i} le faltan campos: {REQUIRED_KEYS - set(seed)}"
            )

    def test_tipos_correctos(self):
        for i, seed in enumerate(SEED_QUERIES):
            assert isinstance(seed["query"], str) and seed["query"], f"#{i}: query vacía"
            assert seed["expected_municipio"] is None or isinstance(
                seed["expected_municipio"], str
            ), f"#{i}: expected_municipio no es str|None"
            ids = seed["expected_tramite_ids"]
            assert ids == EXPECTED_ALL_TRAMITES or isinstance(
                ids, list
            ), f"#{i}: expected_tramite_ids no es list|sentinel"
            assert isinstance(seed["expected_response"], str) and seed["expected_response"]
            assert seed["expected_scope"] in {"in", "out"}, (
                f"#{i}: expected_scope inválido: {seed['expected_scope']!r}"
            )

class TestSeedDistribucion:
    def test_minimo_5_listados(self):
        n = sum(1 for s in SEED_QUERIES if s["expected_tramite_ids"] == EXPECTED_ALL_TRAMITES)
        assert n >= 5, f"se esperaban ≥5 listados, hay {n}"

    def test_minimo_10_especificas(self):
        # "Específica" = scope_in + lista no vacía de IDs + municipio identificado.
        n = sum(
            1
            for s in SEED_QUERIES
            if s["expected_scope"] == "in"
            and isinstance(s["expected_tramite_ids"], list)
            and len(s["expected_tramite_ids"]) >= 1
            and s["expected_municipio"]
        )
        assert n >= 10, f"se esperaban ≥10 específicas, hay {n}"

    def test_minimo_3_scope_out(self):
        n = sum(1 for s in SEED_QUERIES if s["expected_scope"] == "out")
        assert n >= 3, f"se esperaban ≥3 scope_out, hay {n}"

    def test_listados_tienen_municipio(self):
        for i, s in enumerate(SEED_QUERIES):
            if s["expected_tramite_ids"] == EXPECTED_ALL_TRAMITES:
                assert s["expected_municipio"], (
                    f"#{i}: listado sin municipio (ALL exige municipio)"
                )

class TestIdsRealesContraCatalogo:
    """Los IDs específicos del seed deben existir realmente en data/export.json."""

    def test_ids_existen_en_catalogo(self, real_catalog):
        all_ids = {
            tid for bucket in real_catalog.by_municipio.values() for tid in bucket
        }
        for i, s in enumerate(SEED_QUERIES):
            if s["expected_tramite_ids"] == EXPECTED_ALL_TRAMITES:
                continue
            for tid in s["expected_tramite_ids"]:
                assert tid in all_ids, (
                    f"#{i} query={s['query']!r}: tramite_id {tid!r} no existe en el catálogo"
                )

class TestExpandirSentinel:
    def test_expand_listado(self, real_catalog):
        ids = _expand_expected_ids(EXPECTED_ALL_TRAMITES, "Adamuz", real_catalog)
        assert len(ids) == 8  # Adamuz tiene 8 trámites tras dedupe del indexer
        assert all(tid.startswith("adamuz-") for tid in ids)

    def test_expand_lista_explicita(self, real_catalog):
        ids = _expand_expected_ids(["adamuz-volante-de-empadronamiento"], "Adamuz", real_catalog)
        assert ids == ["adamuz-volante-de-empadronamiento"]

    def test_expand_all_sin_municipio_falla(self, real_catalog):
        with pytest.raises(ValueError):
            _expand_expected_ids(EXPECTED_ALL_TRAMITES, None, real_catalog)

class TestSeedToRecord:
    """Verifica el contrato del transform a record MLflow."""

    def test_record_tiene_inputs_y_expectations(self, real_catalog):
        record = _seed_to_record(SEED_QUERIES[0], real_catalog)
        assert "inputs" in record
        assert "expectations" in record
        assert "tags" in record
        assert record["inputs"]["query"] == SEED_QUERIES[0]["query"]
        assert record["expectations"]["expected_response"] == SEED_QUERIES[0]["expected_response"]

    def test_record_listado_expande_ids(self, real_catalog):
        # Primer seed es el listado de Adamuz.
        record = _seed_to_record(SEED_QUERIES[0], real_catalog)
        assert isinstance(record["expectations"]["expected_tramite_ids"], list)
        assert len(record["expectations"]["expected_tramite_ids"]) == 8
        assert record["tags"]["category"] == "listado"

    def test_record_scope_out_tagged(self, real_catalog):
        scope_out_seed = next(s for s in SEED_QUERIES if s["expected_scope"] == "out")
        record = _seed_to_record(scope_out_seed, real_catalog)
        assert record["tags"]["category"] == "scope_out"

    def test_record_sin_municipio_tagged(self, real_catalog):
        sin_mun = next(
            s
            for s in SEED_QUERIES
            if s["expected_municipio"] is None and s["expected_scope"] == "in"
        )
        record = _seed_to_record(sin_mun, real_catalog)
        assert record["tags"]["category"] == "sin_municipio"
        assert record["expectations"]["expected_municipio"] is None
