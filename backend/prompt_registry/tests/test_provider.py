"""Tests unit del PromptProvider.

Cobertura:
- `PromptResource.format`: sustitución `{{var}}`, error de variable faltante,
  tolerancia a variables sobrantes.
- `PromptProvider.get_prompt`: carga desde MLflow, hit caché, refresh tras TTL,
  fallback en error de red, fallback con flag desactivado, KeyError cuando
  no hay fallback registrado.

Todos los tests mockean `mlflow.genai.load_prompt` y `mlflow.set_tracking_uri`
para evitar llamadas reales. La integración contra MLflow real vive en
`test_integration_mlflow.py` (tarea 3.6).

"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.prompt_registry.fallback import TRAMITES_SYSTEM
from backend.prompt_registry.provider import (
    PromptProvider,
    PromptResource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_mlflow_prompt(template: str, version: int = 1) -> SimpleNamespace:
    """Simula el objeto que devuelve `mlflow.genai.load_prompt`."""
    return SimpleNamespace(template=template, version=version)

def _make_provider(**overrides) -> PromptProvider:
    """PromptProvider con defaults de test (caché corta, MLflow habilitado)."""
    defaults = dict(
        tracking_uri="http://mock-mlflow:5000",
        default_alias="production",
        cache_ttl_seconds=60,
        enabled=True,
    )
    defaults.update(overrides)
    return PromptProvider(**defaults)

# ---------------------------------------------------------------------------
# PromptResource.format
# ---------------------------------------------------------------------------

def test_promptresource_format_substituye_variables():
    """`format` sustituye todos los `{{var}}` por sus valores."""
    r = PromptResource(
        name="uja-test-x",
        template="Hola {{nombre}}, tu pedido {{id}} está {{estado}}.",
        version=1,
        source="registry",
    )
    out = r.format(nombre="Félix", id="42", estado="entregado")
    assert out == "Hola Félix, tu pedido 42 está entregado."

def test_promptresource_format_levanta_keyerror_si_falta_variable():
    """`format` levanta `KeyError` si la plantilla referencia una variable
    no provista. Mensaje incluye nombre del prompt + variables faltantes."""
    r = PromptResource(
        name="uja-test-x",
        template="A {{a}} B {{b}} C {{c}}",
        version=1,
        source="registry",
    )
    with pytest.raises(KeyError) as excinfo:
        r.format(a="1", c="3")  # falta `b`
    assert "uja-test-x" in str(excinfo.value)
    assert "'b'" in str(excinfo.value)

def test_promptresource_format_ignora_variables_sobrantes():
    """Variables extra en `values` se ignoran silenciosamente.

    Comportamiento permisivo: si el operador retira una variable del
    template en el registry, el runtime sigue funcionando hasta el
    siguiente despliegue del refactor del caller.
    """
    r = PromptResource(
        name="uja-test-x",
        template="Solo {{a}}.",
        version=1,
        source="registry",
    )
    out = r.format(a="1", b="2", c="3")
    assert out == "Solo 1."

def test_promptresource_format_sin_variables_devuelve_template_literal():
    """Plantilla sin `{{...}}` no requiere `values` y se devuelve tal cual."""
    r = PromptResource(
        name="uja-test-x",
        template="texto sin marcadores",
        version=1,
        source="registry",
    )
    assert r.format() == "texto sin marcadores"

# ---------------------------------------------------------------------------
# PromptProvider.get_prompt — carga desde MLflow
# ---------------------------------------------------------------------------

def test_get_prompt_carga_desde_mlflow_en_primer_acceso():
    """Primer get sin caché → llama a `mlflow.genai.load_prompt` con la URI
    `prompts:/<name>@<alias>` y devuelve `PromptResource(source='registry')`."""
    provider = _make_provider()
    with patch("mlflow.set_tracking_uri") as set_uri, patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.return_value = _fake_mlflow_prompt("Hola {{nombre}}", version=3)
        r = provider.get_prompt("uja-tramites-system")

    set_uri.assert_called_once_with("http://mock-mlflow:5000")
    load_prompt.assert_called_once_with("prompts:/uja-tramites-system@production")
    assert r.name == "uja-tramites-system"
    assert r.template == "Hola {{nombre}}"
    assert r.version == 3
    assert r.source == "registry"

def test_get_prompt_usa_alias_override_si_se_pasa():
    """Si el caller pasa `alias=`, se usa ese alias en lugar del default."""
    provider = _make_provider(default_alias="production")
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.return_value = _fake_mlflow_prompt("hola", version=7)
        provider.get_prompt("uja-tramites-system", alias="staging")

    load_prompt.assert_called_once_with("prompts:/uja-tramites-system@staging")

# ---------------------------------------------------------------------------
# PromptProvider.get_prompt — caché
# ---------------------------------------------------------------------------

def test_get_prompt_hit_cache_no_relanza_mlflow():
    """Segunda llamada con misma `(name, alias)` antes del TTL: devuelve la
    instancia cacheada y NO vuelve a llamar a `mlflow.genai.load_prompt`."""
    provider = _make_provider(cache_ttl_seconds=60)
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.return_value = _fake_mlflow_prompt("template-v1", version=1)
        r1 = provider.get_prompt("uja-tramites-system")
        r2 = provider.get_prompt("uja-tramites-system")

    assert load_prompt.call_count == 1
    assert r1 is r2  # misma instancia exacta (caché devuelve la referencia)

def test_get_prompt_miss_tras_ttl_relanza_mlflow():
    """Tras vencer el TTL, la siguiente llamada refresca contra MLflow y
    devuelve la nueva versión (verifica el camino de "alias movido")."""
    provider = _make_provider(cache_ttl_seconds=60)

    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt, patch("time.monotonic") as monotonic:
        load_prompt.side_effect = [
            _fake_mlflow_prompt("v1-template", version=1),
            _fake_mlflow_prompt("v2-template", version=2),
        ]
        # t=0: primera carga.
        monotonic.return_value = 0.0
        r1 = provider.get_prompt("uja-tramites-system")
        # t=30: dentro del TTL → hit.
        monotonic.return_value = 30.0
        r1_again = provider.get_prompt("uja-tramites-system")
        # t=70: TTL vencido (>60) → miss → refresh.
        monotonic.return_value = 70.0
        r2 = provider.get_prompt("uja-tramites-system")

    assert load_prompt.call_count == 2
    assert r1.version == 1
    assert r1_again.version == 1  # mismo cache hit
    assert r2.version == 2
    assert r2.template == "v2-template"

def test_caches_distintos_alias_por_separado():
    """`(name, 'staging')` y `(name, 'production')` son entradas independientes."""
    provider = _make_provider()
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.side_effect = [
            _fake_mlflow_prompt("staging-v1", version=10),
            _fake_mlflow_prompt("production-v1", version=20),
        ]
        rs = provider.get_prompt("uja-tramites-system", alias="staging")
        rp = provider.get_prompt("uja-tramites-system", alias="production")

    assert load_prompt.call_count == 2
    assert rs.template == "staging-v1"
    assert rp.template == "production-v1"

# ---------------------------------------------------------------------------
# PromptProvider.get_prompt — fallback
# ---------------------------------------------------------------------------

def test_get_prompt_fallback_silencioso_al_fallar_mlflow(caplog):
    """MLflow lanza excepción → provider devuelve el fallback empaquetado
    con `source='fallback'`, `version=0`, y emite `logging.WARNING`."""
    provider = _make_provider()
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.side_effect = ConnectionError("MLflow no responde")
        with caplog.at_level("WARNING", logger="backend.prompt_registry.provider"):
            r = provider.get_prompt("uja-tramites-system")

    assert r.source == "fallback"
    assert r.version == 0
    assert r.template == TRAMITES_SYSTEM
    # WARNING emitido con identificador del prompt + razón.
    assert any(
        "uja-tramites-system" in rec.message and "ConnectionError" in rec.message
        for rec in caplog.records
    )

def test_get_prompt_fallback_si_flag_desactivado_no_llama_mlflow():
    """Con `enabled=False`, el provider NO consulta MLflow y devuelve fallback."""
    provider = _make_provider(enabled=False)
    with patch("mlflow.genai.load_prompt") as load_prompt:
        r = provider.get_prompt("uja-tramites-system")

    load_prompt.assert_not_called()
    assert r.source == "fallback"
    assert r.version == 0
    assert r.template == TRAMITES_SYSTEM

def test_fallback_se_cachea_para_no_martillear_mlflow():
    """Cuando MLflow falla, el provider cachea el fallback con el mismo TTL
    para no reintentar en cada request si el servidor sigue caído."""
    provider = _make_provider(cache_ttl_seconds=60)
    with patch("mlflow.set_tracking_uri"), patch(
        "mlflow.genai.load_prompt"
    ) as load_prompt:
        load_prompt.side_effect = ConnectionError("down")
        r1 = provider.get_prompt("uja-tramites-system")
        r2 = provider.get_prompt("uja-tramites-system")

    assert load_prompt.call_count == 1  # NO se reintenta dentro del TTL
    assert r1 is r2
    assert r1.source == "fallback"

def test_get_prompt_keyerror_si_no_hay_fallback_registrado():
    """Si el prompt pedido no existe ni en MLflow ni en `_FALLBACK_MAP`,
    el provider levanta `KeyError` (no hay forma de servir nada coherente)."""
    provider = _make_provider(enabled=False)
    with pytest.raises(KeyError) as excinfo:
        provider.get_prompt("uja-desconocido-system")
    assert "uja-desconocido-system" in str(excinfo.value)
    assert "fallback.py" in str(excinfo.value)

# ---------------------------------------------------------------------------
# Sanity: el fallback empaquetado de Trámites coincide byte-a-byte con el
# `_SYSTEM_PROMPT` activo. Este test es complementario al de paridad MD5
# en `backend/rag_tramites/tests/test_prompt_builder.py` y existe aquí para
# que la suite del módulo capture la regresión sin depender de otra carpeta.
# ---------------------------------------------------------------------------

def test_fallback_tramites_system_es_byte_igual_a_codigo_actual():
    from backend.rag_tramites.graph.nodes import _SYSTEM_PROMPT

    assert TRAMITES_SYSTEM == _SYSTEM_PROMPT, (
        "TRAMITES_SYSTEM (fallback) divergió de _SYSTEM_PROMPT (código). "
        "Esto romperá la paridad bit-exacta Trámites↔Comparador A en cuanto "
        "el provider entre en uso. Sincronízalos."
    )

# ---------------------------------------------------------------------------
# Sanity test for `time.monotonic` patching — defensivo: si en otro test
# olvidamos restaurar el patch, este test detecta la fuga.
# ---------------------------------------------------------------------------

def test_time_monotonic_no_esta_pacheado_globalmente():
    """Defensivo: asegura que `time.monotonic` devuelve un float real."""
    assert isinstance(time.monotonic(), float)

# ---------------------------------------------------------------------------
# Tracking de prompts usados — ContextVar async-safe
# ---------------------------------------------------------------------------

def test_tracking_inactivo_si_no_se_llama_begin():
    """Sin `begin_prompt_tracking()`, `get_prompts_used()` devuelve lista vacía
    y `get_prompt()` no rompe."""
    from backend.prompt_registry.provider import get_prompts_used

    provider = _make_provider(enabled=False)
    provider.get_prompt("uja-tramites-system")
    assert get_prompts_used() == []

def test_tracking_registra_cada_get_prompt():
    """`begin_prompt_tracking` activa el ContextVar; cada `get_prompt`
    posterior añade su `PromptResource` a la lista. Las llamadas con caché
    hit también deben registrarse (la traza necesita saber qué prompt se
    USÓ en la request, no si vino de caché)."""
    from backend.prompt_registry.provider import (
        begin_prompt_tracking,
        get_prompts_used,
    )

    begin_prompt_tracking()
    provider = _make_provider(enabled=False, cache_ttl_seconds=60)

    provider.get_prompt("uja-tramites-system")
    provider.get_prompt("uja-tramites-user-template")
    # Segunda llamada al mismo prompt → caché hit, pero también registra.
    provider.get_prompt("uja-tramites-system")

    used = get_prompts_used()
    assert len(used) == 3
    names = [r.name for r in used]
    assert names == [
        "uja-tramites-system",
        "uja-tramites-user-template",
        "uja-tramites-system",
    ]
    # Todos son del fallback porque enabled=False.
    assert all(r.source == "fallback" for r in used)

def test_tracking_aislado_entre_contexts_asyncio():
    """ContextVar aisla por task: dos contextos paralelos no se mezclan."""
    import asyncio

    from backend.prompt_registry.provider import (
        begin_prompt_tracking,
        get_prompts_used,
    )

    async def request_handler(prompt_name: str) -> list[str]:
        begin_prompt_tracking()
        provider = _make_provider(enabled=False)
        provider.get_prompt(prompt_name)
        return [r.name for r in get_prompts_used()]

    async def run_both():
        # Las dos tasks corren con contextos asyncio aislados.
        a, b = await asyncio.gather(
            request_handler("uja-tramites-system"),
            request_handler("uja-tramites-user-template"),
        )
        return a, b

    a, b = asyncio.run(run_both())
    assert a == ["uja-tramites-system"]
    assert b == ["uja-tramites-user-template"]

def test_begin_prompt_tracking_resetea_lista_previa():
    """Una segunda llamada a `begin_prompt_tracking()` deja la lista vacía
    (la request anterior no contamina la siguiente cuando se reutiliza el
    mismo contexto, p.ej. en tests secuenciales)."""
    from backend.prompt_registry.provider import (
        begin_prompt_tracking,
        get_prompts_used,
    )

    begin_prompt_tracking()
    provider = _make_provider(enabled=False)
    provider.get_prompt("uja-tramites-system")
    assert len(get_prompts_used()) == 1

    begin_prompt_tracking()
    assert get_prompts_used() == []
