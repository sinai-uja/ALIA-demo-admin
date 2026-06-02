"""Tests del builder unificado de user-prompt.

Garantiza que `node_generate` (Trámites) y `_build_prompt` (Comparador)
producen exactamente el mismo string cuando no hay historial previo, de
forma que el bloque HISTORIAL solo aparece cuando hay turnos reales.

`build_tramites_user_prompt` carga la plantilla del PromptProvider. Sin
MLflow alcanzable, el provider cae al fallback empaquetado en
`backend/prompt_registry/fallback.py:TRAMITES_USER_TEMPLATE`. Los tests
de paridad MD5 al final de este archivo blindan la regresión.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

from backend.comparador.router import _build_prompt as comparador_build_prompt
from backend.prompt_registry.fallback import TRAMITES_USER_TEMPLATE
from backend.prompt_registry.provider import PromptResource
from backend.rag_tramites.graph.nodes import build_tramites_user_prompt

def _fake_reranked() -> list[dict]:
    return [
        {
            "documento": "Volante de Empadronamiento. Descripción del trámite.",
            "metadata": {
                "nombre": "Volante de Empadronamiento",
                "municipio": "Adamuz",
                "url": "https://e-admin.eprinsa.es/adamuz/monotramite?tramite=7204",
                "auth": "no especificado",
            },
        },
        {
            "documento": "Certificado de Empadronamiento Individual.",
            "metadata": {
                "nombre": "Certificado de Empadronamiento Individual",
                "municipio": "Adamuz",
                "url": "https://e-admin.eprinsa.es/adamuz/monotramite?tramite=7303",
                "auth": "no especificado",
            },
        },
    ]

QUERY = "¿Cómo me empadrono en Adamuz?"

# ---------------------------------------------------------------------------
# Paridad entre pipelines
# ---------------------------------------------------------------------------

def test_parity_no_history_matches_comparador():
    """Sin historial, builder y comparador deben producir EL MISMO string."""
    reranked = _fake_reranked()
    p_builder = build_tramites_user_prompt(QUERY, reranked, "especifico")
    p_comparador = comparador_build_prompt(QUERY, reranked, "especifico")
    assert p_builder == p_comparador, (
        "Trámites y Comparador deben usar el mismo prompt sin historial. "
        "Si esto falla, alguien rompió la fuente única de verdad."
    )

def test_parity_empty_history_matches_no_history():
    reranked = _fake_reranked()
    a = build_tramites_user_prompt(QUERY, reranked, "especifico", history=None)
    b = build_tramites_user_prompt(QUERY, reranked, "especifico", history=[])
    assert a == b

# ---------------------------------------------------------------------------
# HISTORIAL — solo cuando hay turnos previos REALES
# ---------------------------------------------------------------------------

def test_first_turn_does_not_inject_historial():
    """Primer turno: messages = [current_query]. El builder debe excluirlo."""
    reranked = _fake_reranked()
    # Simula lo que hace node_generate: messages[:-1] elimina el turno actual.
    messages_simulated_first_turn = [{"role": "user", "content": QUERY}]
    prior = messages_simulated_first_turn[:-1]
    prompt = build_tramites_user_prompt(QUERY, reranked, "especifico", history=prior)
    assert "HISTORIAL RECIENTE" not in prompt
    # La query aparece UNA sola vez (en PREGUNTA ACTUAL).
    assert prompt.count(QUERY) == 1

def test_second_turn_injects_historial():
    reranked = _fake_reranked()
    history = [
        {"role": "user", "content": "¿Qué trámites hay en Adamuz?"},
        {"role": "assistant", "content": "Hay varios trámites disponibles..."},
    ]
    prompt = build_tramites_user_prompt(QUERY, reranked, "especifico", history=history)
    assert "HISTORIAL RECIENTE:" in prompt
    assert "user: ¿Qué trámites hay en Adamuz?" in prompt
    assert "assistant: Hay varios trámites disponibles..." in prompt
    # La query actual sigue apareciendo UNA sola vez (no se duplica en historial).
    assert prompt.count(QUERY) == 1

def test_history_limited_to_last_four():
    reranked = _fake_reranked()
    history = [
        {"role": "user", "content": f"msg-{i}"} for i in range(10)
    ]
    prompt = build_tramites_user_prompt(QUERY, reranked, "especifico", history=history)
    assert "msg-9" in prompt
    assert "msg-6" in prompt
    assert "msg-5" not in prompt  # solo últimos 4

# ---------------------------------------------------------------------------
# Intent listado
# ---------------------------------------------------------------------------

def test_intent_listado_adds_instruction():
    reranked = _fake_reranked()
    prompt = build_tramites_user_prompt(QUERY, reranked, "listado")
    assert "IMPORTANTE: El usuario quiere un listado COMPLETO" in prompt
    assert f"En el contexto hay {len(reranked)} trámites" in prompt

def test_intent_especifico_omits_listado_instruction():
    reranked = _fake_reranked()
    prompt = build_tramites_user_prompt(QUERY, reranked, "especifico")
    assert "listado COMPLETO" not in prompt

# ---------------------------------------------------------------------------
# Paridad MD5: blinda contra regresiones del refactor que mete
# `PromptProvider` en `build_tramites_user_prompt`. Si alguien rompe la
# plantilla del fallback o la lógica de los 4 valores (contexto,
# instruccion_intent, historial, query), estos MD5 cambian y el test falla.
#
# Para regenerar (solo cuando el cambio sea INTENCIONAL):
#   python -c "import hashlib; from backend.rag_tramites.tests.test_prompt_builder \
#     import _fake_reranked, QUERY; \
#     from backend.rag_tramites.graph.nodes import build_tramites_user_prompt; \
#     p = build_tramites_user_prompt(QUERY, _fake_reranked(), 'especifico'); \
#     print(hashlib.md5(p.encode()).hexdigest())"
# ---------------------------------------------------------------------------

# MD5 de referencia, verificados byte-a-byte en los 3 escenarios
# (específico, listado, específico+historial) con el provider activado
# en modo fallback.
_MD5_REF = {
    "especifico_sin_historial": "4c4892512029899f301e9cd0f5e6349e",
    "listado_sin_historial": "b095d7b7dbb47c34752ffb3975001589",
    "especifico_con_historial": "6530bab7aa759a13a8357061c35e196e",
}

def test_md5_especifico_sin_historial():
    """MD5 del prompt completo para la query de referencia (intent=especifico)."""
    prompt = build_tramites_user_prompt(QUERY, _fake_reranked(), "especifico")
    md5 = hashlib.md5(prompt.encode()).hexdigest()
    assert md5 == _MD5_REF["especifico_sin_historial"], (
        f"Prompt MD5 cambió: {md5}. Si el cambio es intencional, "
        "actualiza _MD5_REF en este archivo y documenta el motivo en el commit."
    )

def test_md5_listado_sin_historial():
    prompt = build_tramites_user_prompt(QUERY, _fake_reranked(), "listado")
    md5 = hashlib.md5(prompt.encode()).hexdigest()
    assert md5 == _MD5_REF["listado_sin_historial"], f"Prompt MD5 cambió: {md5}"

def test_md5_especifico_con_historial():
    history = [
        {"role": "user", "content": "¿Qué trámites hay en Adamuz?"},
        {"role": "assistant", "content": "Hay varios trámites disponibles..."},
    ]
    prompt = build_tramites_user_prompt(QUERY, _fake_reranked(), "especifico", history=history)
    md5 = hashlib.md5(prompt.encode()).hexdigest()
    assert md5 == _MD5_REF["especifico_con_historial"], f"Prompt MD5 cambió: {md5}"

def test_paridad_md5_con_provider_activado_y_registry_simulado():
    """Paridad bit-exacta cuando MLflow SÍ responde con el contenido del
    fallback (v.gr. tras bootstrap inicial). Mockea `PromptProvider.get_prompt`
    devolviendo un `PromptResource(source='registry', version=1)` con la
    plantilla del fallback → el output debe quedar idéntico al modo fallback.
    """
    fake_resource = PromptResource(
        name="uja-tramites-user-template",
        template=TRAMITES_USER_TEMPLATE,
        version=1,
        source="registry",
    )
    with patch(
        "backend.rag_tramites.graph.nodes.get_provider"
    ) as get_provider_mock:
        get_provider_mock.return_value.get_prompt.return_value = fake_resource
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked(), "especifico")

    md5 = hashlib.md5(prompt.encode()).hexdigest()
    assert md5 == _MD5_REF["especifico_sin_historial"], (
        "Con el provider devolviendo la plantilla del registry (idéntica al "
        "fallback), el output debe ser bit-exacto al modo fallback. Si esto "
        "falla, hay un bug en el wiring del provider o en la plantilla."
    )
    # Verifica que el provider fue invocado con el nombre correcto (sin slashes).
    get_provider_mock.return_value.get_prompt.assert_called_once_with(
        "uja-tramites-user-template"
    )

# ---------------------------------------------------------------------------
# Render chunked — modo del builder cuando los reranked traen metadata
# `section` (resultado del chunking por secciones del trámite).
# ---------------------------------------------------------------------------

def _fake_reranked_chunked() -> list[dict]:
    """3 chunks del mismo trámite + 2 chunks de otro distinto."""
    return [
        {
            "documento": "Padrón — Objeto. Inscripción en el padrón municipal.",
            "metadata": {
                "id": "adamuz-padron",
                "nombre": "Padrón",
                "municipio": "Adamuz",
                "url": "https://e-admin.eprinsa.es/adamuz/padron",
                "auth": "Cl@ve",
                "section": "Objeto",
                "chunk_index": 1,
                "total_chunks": 3,
            },
        },
        {
            "documento": "Padrón — Documentación. DNI y comprobante de domicilio.",
            "metadata": {
                "id": "adamuz-padron",
                "nombre": "Padrón",
                "municipio": "Adamuz",
                "url": "https://e-admin.eprinsa.es/adamuz/padron",
                "auth": "Cl@ve",
                "section": "Documentación",
                "chunk_index": 2,
                "total_chunks": 3,
            },
        },
        {
            "documento": "Plusvalías — Plazo. 30 días hábiles desde la transmisión.",
            "metadata": {
                "id": "adamuz-plusvalias",
                "nombre": "Plusvalías",
                "municipio": "Adamuz",
                "url": "https://e-admin.eprinsa.es/adamuz/plusvalias",
                "auth": "certificado digital",
                "section": "Plazo",
                "chunk_index": 1,
                "total_chunks": 2,
            },
        },
    ]

class TestRenderChunkedMode:
    """Render reagrupado por trámite cuando reranked traen `section`."""

    def test_agrupa_chunks_por_tramite(self):
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked_chunked(), "especifico")
        # 2 trámites únicos (Padrón con 2 chunks, Plusvalías con 1).
        assert "[Trámite 1] Padrón" in prompt
        assert "[Trámite 2] Plusvalías" in prompt
        # Header dice 2 trámites, 3 secciones.
        assert "2 trámites encontrados, 3 secciones" in prompt

    def test_info_comun_no_se_repite(self):
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked_chunked(), "especifico")
        # La URL del trámite Padrón aparece una sola vez (no por chunk).
        assert prompt.count("https://e-admin.eprinsa.es/adamuz/padron") == 1

    def test_secciones_renderizan_como_subbloques(self):
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked_chunked(), "especifico")
        assert "── Objeto: Inscripción en el padrón" in prompt
        assert "── Documentación: DNI y comprobante" in prompt
        assert "── Plazo: 30 días hábiles" in prompt

    def test_prefijo_nombre_section_no_aparece_en_render(self):
        """El prefijo `Padrón — Objeto.` del page_content se quita en el render
        para no duplicar info con el header de la sección."""
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked_chunked(), "especifico")
        # El prefijo limpio no debe aparecer literalmente.
        assert "Padrón — Objeto." not in prompt
        # Sí debe aparecer el cuerpo después del prefijo.
        assert "Inscripción en el padrón municipal" in prompt

    def test_fallback_render_si_no_hay_section(self):
        """Sin `section` en metadata, mantiene el render histórico (paridad MD5)."""
        prompt = build_tramites_user_prompt(QUERY, _fake_reranked(), "especifico")
        # No usa el formato chunked.
        assert "── " not in prompt
        # Mantiene el formato "Descripción: ..." del render histórico.
        assert "Descripción:" in prompt
