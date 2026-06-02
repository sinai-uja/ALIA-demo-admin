"""Tests unitarios de los 4 scorers custom. Sin red, sin LLM.
"""

from __future__ import annotations

import pytest

from backend.rag_tramites_eval.catalog import Catalog, TramiteEntry
from backend.rag_tramites_eval.scorers import (
    municipio_match,
    precision_urls,
    recall_tramites,
    scope_mismatch_correct,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog_simple() -> Catalog:
    """Catálogo con 2 municipios y 2 trámites cada uno."""
    return Catalog(
        by_municipio={
            "adamuz": {
                "adamuz-padron": TramiteEntry(
                    tramite_id="adamuz-padron",
                    nombre="Padrón",
                    municipio="Adamuz",
                    urls=frozenset({"https://e-admin.eprinsa.es/adamuz/x?t=1"}),
                ),
                "adamuz-plusvalia": TramiteEntry(
                    tramite_id="adamuz-plusvalia",
                    nombre="Plusvalía",
                    municipio="Adamuz",
                    urls=frozenset({"https://e-admin.eprinsa.es/adamuz/x?t=2"}),
                ),
            },
            "cabra": {
                "cabra-padron": TramiteEntry(
                    tramite_id="cabra-padron",
                    nombre="Padrón",
                    municipio="Cabra",
                    urls=frozenset({"https://e-admin.eprinsa.es/cabra/x?t=9"}),
                ),
            },
        }
    )

# ---------------------------------------------------------------------------
# recall_tramites
# ---------------------------------------------------------------------------

class TestRecallTramites:
    def test_happy_path(self):
        assert recall_tramites({"a", "b"}, ["a", "b", "c"]) == 1.0

    def test_partial(self):
        assert recall_tramites({"a", "b"}, ["a", "z"]) == 0.5

    def test_zero(self):
        assert recall_tramites({"a", "b"}, ["x", "y"]) == 0.0

    def test_expected_vacio_devuelve_uno(self):
        # Por vacuidad: no hay nada que recuperar → recall perfecto.
        assert recall_tramites([], ["a", "b"]) == 1.0

    def test_ignora_ids_vacios(self):
        assert recall_tramites({"a", ""}, ["a", None]) == 1.0

# ---------------------------------------------------------------------------
# precision_urls
# ---------------------------------------------------------------------------

class TestPrecisionUrls:
    def test_sin_urls_es_perfecta(self, catalog_simple):
        # Sin URLs no hay nada que penalizar.
        assert precision_urls("Lorem ipsum sin enlaces.", "Adamuz", catalog_simple) == 1.0

    def test_url_valida(self, catalog_simple):
        text = "Accede aquí: https://e-admin.eprinsa.es/adamuz/x?t=1"
        assert precision_urls(text, "Adamuz", catalog_simple) == 1.0

    def test_url_inventada_penaliza(self, catalog_simple):
        text = "Url falsa: https://no-existe.invalid/x"
        assert precision_urls(text, "Adamuz", catalog_simple) == 0.0

    def test_cross_municipio_penaliza(self, catalog_simple):
        # URL de Cabra contestando una query de Adamuz → 0/1.
        text = "Enlace: https://e-admin.eprinsa.es/cabra/x?t=9"
        assert precision_urls(text, "Adamuz", catalog_simple) == 0.0

    def test_mezcla_valida_invalida(self, catalog_simple):
        text = (
            "Buena: https://e-admin.eprinsa.es/adamuz/x?t=1 "
            "Mala: https://no-existe.invalid/x"
        )
        assert precision_urls(text, "Adamuz", catalog_simple) == 0.5

    def test_municipio_none_acepta_cualquier_catalogo(self, catalog_simple):
        # Query sin municipio: URL válida en cualquier municipio cuenta.
        text = "Enlace: https://e-admin.eprinsa.es/cabra/x?t=9"
        assert precision_urls(text, None, catalog_simple) == 1.0

    def test_municipio_empty_string_equivalente_a_none(self, catalog_simple):
        # El bootstrap convierte None → "" para MLflow; precision_urls debe
        # tratar ambos como "cualquier municipio del catálogo vale".
        text = "Enlace: https://e-admin.eprinsa.es/cabra/x?t=9"
        assert precision_urls(text, "", catalog_simple) == 1.0
        assert precision_urls(text, None, catalog_simple) == precision_urls(text, "", catalog_simple)

    def test_url_con_punto_final(self, catalog_simple):
        # Trailing punctuation no debe romper la validación.
        text = "Ver https://e-admin.eprinsa.es/adamuz/x?t=1."
        assert precision_urls(text, "Adamuz", catalog_simple) == 1.0

# ---------------------------------------------------------------------------
# municipio_match
# ---------------------------------------------------------------------------

class TestMunicipioMatch:
    def test_match_exacto(self):
        assert municipio_match("Adamuz", "Adamuz") == 1.0

    def test_match_normalizado_tildes(self):
        assert municipio_match("Córdoba", "cordoba") == 1.0

    def test_ambos_none(self):
        # Query sin municipio + el nodo detectó None → acierto.
        assert municipio_match(None, None) == 1.0

    def test_falso_negativo(self):
        assert municipio_match(None, "Adamuz") == 0.0

    def test_falso_positivo(self):
        assert municipio_match("Adamuz", None) == 0.0

    def test_municipio_incorrecto(self):
        assert municipio_match("Adamuz", "Cabra") == 0.0

# ---------------------------------------------------------------------------
# scope_mismatch_correct
# ---------------------------------------------------------------------------

class TestScopeMismatch:
    def test_out_con_disclaimer(self):
        text = (
            "El catálogo no incluye el procedimiento completo de empadronamiento. "
            "Te remitimos al Ayuntamiento de Adamuz para iniciarlo."
        )
        assert scope_mismatch_correct(text, "out") == 1.0

    def test_out_con_sede_electronica(self):
        text = (
            "Aviso: este procedimiento debes iniciarlo en la sede electrónica del "
            "municipio. Como información complementaria, los trámites del catálogo "
            "relacionados son…"
        )
        assert scope_mismatch_correct(text, "out") == 1.0

    def test_out_sin_disclaimer_es_fallo(self):
        text = "Para empadronarte solicita el Certificado de Empadronamiento Individual."
        assert scope_mismatch_correct(text, "out") == 0.0

    def test_in_sin_disclaimer(self):
        text = "El trámite 'Padrón' está disponible en https://example.com/adamuz/p"
        assert scope_mismatch_correct(text, "in") == 1.0

    def test_in_con_disclaimer_es_falso_positivo(self):
        text = (
            "El catálogo no incluye este procedimiento. "
            "(Aunque sí está, pero el LLM se equivocó.)"
        )
        assert scope_mismatch_correct(text, "in") == 0.0

    def test_disclaimer_al_final_no_cuenta(self):
        # 700 chars de relleno antes del disclaimer → fuera de la ventana head.
        text = "X" * 700 + " sede electrónica"
        assert scope_mismatch_correct(text, "out") == 0.0

    def test_scope_invalido_levanta(self):
        with pytest.raises(ValueError):
            scope_mismatch_correct("foo", "maybe")
