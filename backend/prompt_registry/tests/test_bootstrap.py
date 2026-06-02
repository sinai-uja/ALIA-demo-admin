"""Tests unit del bootstrap del Prompt Registry.

Mockean `mlflow.genai.load_prompt`, `mlflow.genai.register_prompt`,
`mlflow.genai.set_prompt_alias` y `mlflow.set_tracking_uri` para verificar
las 3 ramas de idempotencia sin tocar un MLflow real:

1. v1 no existe → registra + crea aliases.
2. v1 existe y coincide con fallback → no registra, crea solo aliases faltantes.
3. v1 existe pero contenido diverge → log WARNING, no toca nada.

La integración real contra el servicio MLflow Docker vive en
`test_integration_mlflow.py` (tarea 3.6).

"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.prompt_registry import bootstrap
from backend.prompt_registry.fallback import (
    RAG_DOCS_SYSTEM,
    RAG_DOCS_USER_TEMPLATE,
    TRAMITES_SYSTEM,
    TRAMITES_USER_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Helpers — mocks de mlflow.genai
# ---------------------------------------------------------------------------

def _fake_prompt_v1(template: str) -> SimpleNamespace:
    """Simula lo que devuelve `mlflow.genai.load_prompt` para un v1 existente."""
    return SimpleNamespace(template=template, version=1)

# ---------------------------------------------------------------------------
# Caso 1: MLflow vacío → registra todo + crea aliases.
# ---------------------------------------------------------------------------

def test_bootstrap_mlflow_vacio_registra_los_4_prompts_y_aliases():
    """Si load_prompt devuelve None para todo (allow_missing=True), el
    bootstrap llama a register_prompt 4 veces y set_prompt_alias 8 veces
    (2 aliases x 4 prompts)."""

    def load_side_effect(uri, allow_missing=False, **kwargs):
        assert allow_missing is True  # bootstrap siempre lo pide así
        return None  # nada existe

    register_calls: list[dict] = []

    def register_side_effect(*, name, template, commit_message=None, tags=None, **kwargs):
        register_calls.append({"name": name, "template": template, "tags": tags})
        return SimpleNamespace(version=1, name=name)

    alias_calls: list[tuple] = []

    def alias_side_effect(*, name, alias, version):
        alias_calls.append((name, alias, version))

    with patch("mlflow.set_tracking_uri") as set_uri, patch(
        "mlflow.genai.load_prompt", side_effect=load_side_effect
    ), patch(
        "mlflow.genai.register_prompt", side_effect=register_side_effect
    ), patch(
        "mlflow.genai.set_prompt_alias", side_effect=alias_side_effect
    ):
        results = bootstrap.register_initial_prompts(
            tracking_uri="http://mock:5000"
        )

    set_uri.assert_called_once_with("http://mock:5000")
    # 4 prompts registrados con sus contenidos del fallback.
    assert len(register_calls) == 4
    by_name = {c["name"]: c for c in register_calls}
    assert by_name["uja-tramites-system"]["template"] == TRAMITES_SYSTEM
    assert by_name["uja-tramites-user-template"]["template"] == TRAMITES_USER_TEMPLATE
    assert by_name["uja-rag-docs-system"]["template"] == RAG_DOCS_SYSTEM
    assert by_name["uja-rag-docs-user-template"]["template"] == RAG_DOCS_USER_TEMPLATE
    # Tags incluyen marker de fase.
    for call in register_calls:
        assert call["tags"]["phase"] == "13"
        assert call["tags"]["bootstrap"] == "true"

    # 8 aliases creados (2 por cada prompt).
    assert len(alias_calls) == 8
    expected_aliases = {
        (name, alias, 1)
        for name in by_name
        for alias in ("staging", "production")
    }
    assert set(alias_calls) == expected_aliases

    # Resultado estructurado.
    for name in by_name:
        r = results[name]
        assert r.registry_action == "created"
        assert r.aliases == {"staging": "created", "production": "created"}
        assert r.version == 1

# ---------------------------------------------------------------------------
# Caso 2: v1 ya existe y coincide → no registra, crea solo aliases faltantes.
# ---------------------------------------------------------------------------

def test_bootstrap_v1_existe_y_coincide_no_registra_pero_completa_aliases():
    """v1 existe con contenido idéntico al fallback. El alias `staging` ya
    existe; `production` falta. Bootstrap NO debe registrar v1; debe crear
    solo el alias `production`."""

    fallback_by_name = {
        "uja-tramites-system": TRAMITES_SYSTEM,
        "uja-tramites-user-template": TRAMITES_USER_TEMPLATE,
        "uja-rag-docs-system": RAG_DOCS_SYSTEM,
        "uja-rag-docs-user-template": RAG_DOCS_USER_TEMPLATE,
    }

    def load_side_effect(uri, allow_missing=False, **kwargs):
        # uri patterns: prompts:/<name>/1   o   prompts:/<name>@<alias>
        for name, template in fallback_by_name.items():
            if uri == f"prompts:/{name}/1":
                return _fake_prompt_v1(template)
            if uri == f"prompts:/{name}@staging":
                # staging ya existe.
                return _fake_prompt_v1(template)
            if uri == f"prompts:/{name}@production":
                # production no existe.
                return None
        return None

    register_calls: list[str] = []
    alias_calls: list[tuple] = []

    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt", side_effect=load_side_effect
    ), patch(
        "mlflow.genai.register_prompt",
        side_effect=lambda **kw: register_calls.append(kw["name"]),
    ), patch(
        "mlflow.genai.set_prompt_alias",
        side_effect=lambda *, name, alias, version: alias_calls.append((name, alias, version)),
    ):
        results = bootstrap.register_initial_prompts(tracking_uri="http://mock:5000")

    # Ningún register: v1 ya existe en todos.
    assert register_calls == []
    # 4 set_prompt_alias (solo production, staging se mantiene).
    assert len(alias_calls) == 4
    assert all(alias == "production" for _, alias, _ in alias_calls)

    for name in fallback_by_name:
        r = results[name]
        assert r.registry_action == "already_matches"
        assert r.aliases == {
            "staging": "already_exists",
            "production": "created",
        }

# ---------------------------------------------------------------------------
# Caso 3: v1 diverge → WARNING, no se sobreescribe, no se crean aliases nuevos.
# ---------------------------------------------------------------------------

def test_bootstrap_v1_diverge_warning_y_no_toca_nada(caplog):
    """v1 existe en MLflow pero con contenido editado por el operador (no
    coincide con fallback). Bootstrap debe:
    - Emitir WARNING identificando el prompt afectado.
    - NO llamar a register_prompt.
    - NO crear aliases nuevos (aunque falten — el operador debe asignarlos).
    """

    def load_side_effect(uri, allow_missing=False, **kwargs):
        if uri == "prompts:/uja-tramites-system/1":
            return _fake_prompt_v1("CONTENIDO MODIFICADO POR EL OPERADOR")
        if uri.startswith("prompts:/uja-tramites-system@"):
            return None  # aliases no existen
        # El resto de prompts: simulamos que ya están bien (skip).
        if "/1" in uri:
            for name, template in {
                "uja-tramites-user-template": TRAMITES_USER_TEMPLATE,
                "uja-rag-docs-system": RAG_DOCS_SYSTEM,
                "uja-rag-docs-user-template": RAG_DOCS_USER_TEMPLATE,
            }.items():
                if uri == f"prompts:/{name}/1":
                    return _fake_prompt_v1(template)
        # aliases del resto: ya existen.
        return _fake_prompt_v1("anything")

    register_calls: list[str] = []
    alias_calls: list[tuple] = []

    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt", side_effect=load_side_effect
    ), patch(
        "mlflow.genai.register_prompt",
        side_effect=lambda **kw: register_calls.append(kw["name"]),
    ), patch(
        "mlflow.genai.set_prompt_alias",
        side_effect=lambda *, name, alias, version: alias_calls.append((name, alias, version)),
    ), caplog.at_level("WARNING", logger="backend.prompt_registry.bootstrap"):
        results = bootstrap.register_initial_prompts(tracking_uri="http://mock:5000")

    # No registrado: respetamos la versión modificada por el operador.
    assert "uja-tramites-system" not in register_calls
    # No se crean aliases nuevos para el prompt divergente.
    assert all(name != "uja-tramites-system" for name, _, _ in alias_calls)
    # WARNING emitido identificando el prompt.
    assert any(
        "uja-tramites-system" in rec.message and "DIFIERE" in rec.message
        for rec in caplog.records
    )

    r = results["uja-tramites-system"]
    assert r.registry_action == "diverged_skipped"
    # Aliases reportados como "already_exists" (semánticamente: no acción).
    assert all(a == "already_exists" for a in r.aliases.values())

# ---------------------------------------------------------------------------
# Caso 4: dry_run no efectúa cambios.
# ---------------------------------------------------------------------------

def test_bootstrap_dry_run_no_llama_a_register_ni_set_alias():
    """Con `dry_run=True`, las funciones de escritura (register_prompt y
    set_prompt_alias) NO se invocan. Sí se invocan las lecturas y
    set_tracking_uri."""

    def load_side_effect(uri, allow_missing=False, **kwargs):
        return None  # nada existe; el plan sería "crear todo"

    with patch("mlflow.set_tracking_uri") as set_uri, patch(
        "mlflow.genai.load_prompt", side_effect=load_side_effect
    ), patch("mlflow.genai.register_prompt") as register, patch(
        "mlflow.genai.set_prompt_alias"
    ) as set_alias:
        results = bootstrap.register_initial_prompts(
            tracking_uri="http://mock:5000", dry_run=True
        )

    set_uri.assert_called_once()
    register.assert_not_called()
    set_alias.assert_not_called()
    # El plan reportado es "crear todo".
    for r in results.values():
        assert r.registry_action == "created"
        assert r.aliases == {"staging": "created", "production": "created"}

# ---------------------------------------------------------------------------
# Edge case: falta tracking_uri en settings y no se pasa explícito.
# ---------------------------------------------------------------------------

def test_bootstrap_falla_si_no_hay_tracking_uri(monkeypatch):
    """El bootstrap requiere MLflow alcanzable. Sin URI configurada, debe
    fallar explícitamente en vez de intentar contra el default local."""
    from backend.config import settings

    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", None)
    with pytest.raises(RuntimeError) as excinfo:
        bootstrap.register_initial_prompts()  # sin tracking_uri explícito
    assert "MLFLOW_TRACKING_URI" in str(excinfo.value)

# ---------------------------------------------------------------------------
# CLI: main() devuelve 0 en éxito, 1 en error.
# ---------------------------------------------------------------------------

def test_main_devuelve_0_en_exito(capsys):
    """`main([])` ejecuta el bootstrap (dry_run mockeado) y devuelve 0."""
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt", return_value=None
    ), patch("mlflow.genai.register_prompt", return_value=SimpleNamespace(version=1)), patch(
        "mlflow.genai.set_prompt_alias"
    ):
        # Forzar tracking_uri vía settings para el CLI.
        from backend.config import settings
        original = settings.MLFLOW_TRACKING_URI
        try:
            object.__setattr__(settings, "MLFLOW_TRACKING_URI", "http://mock:5000")
            exit_code = bootstrap.main([])
        finally:
            object.__setattr__(settings, "MLFLOW_TRACKING_URI", original)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Resumen del bootstrap" in captured.out

def test_main_devuelve_1_si_bootstrap_falla(monkeypatch, capsys):
    """`main()` captura excepciones y devuelve exit code 1 con log de error."""
    from backend.config import settings
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", None)
    exit_code = bootstrap.main([])
    assert exit_code == 1
