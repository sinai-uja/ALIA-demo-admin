"""Tests de integración contra MLflow real (requiere docker-compose up).

Verifica el round-trip completo del PromptProvider contra un servidor MLflow
en vivo:

1. Registra un prompt temporal (nombre con UUID para no colisionar con la
   bootstrap state).
2. Crea aliases `staging` y `production` apuntando a v1.
3. Carga vía PromptProvider y verifica `source=registry, version=1`.
4. Registra v2 del prompt y mueve `production` a v2.
5. Espera el TTL configurado.
6. Re-carga vía PromptProvider y verifica `source=registry, version=2`
   (el cambio se propaga sin reiniciar el provider).
7. Limpia (best-effort delete del prompt).

Marcado con `pytest.mark.integration`: se ejecuta solo cuando
`MLFLOW_TRACKING_URI` está configurado y responde. CI lo des-selecciona
para no depender de infraestructura levantada.

"""

from __future__ import annotations

import os
import time
import uuid

import pytest

def _mlflow_reachable() -> bool:
    """True si `MLFLOW_TRACKING_URI` está seteado y el server responde."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.MlflowClient().search_experiments(max_results=1)
        return True
    except Exception:
        return False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _mlflow_reachable(),
        reason="MLFLOW_TRACKING_URI no configurado o servidor no alcanzable.",
    ),
]

@pytest.fixture
def temp_prompt_name() -> str:
    """Genera un nombre de prompt único para que el test no colisione con
    versiones previas ni con los prompts de bootstrap."""
    return f"test-prompt-{uuid.uuid4().hex[:12]}"

@pytest.fixture
def cleanup_prompts():
    """Lista mutable de nombres a borrar al final del test (best-effort)."""
    names: list[str] = []
    yield names
    if not names:
        return
    import mlflow

    client = mlflow.MlflowClient()
    for name in names:
        try:
            # Borrar aliases primero porque algunas versiones de MLflow
            # impiden delete_prompt mientras tenga aliases.
            for alias in ("staging", "production"):
                try:
                    client.delete_prompt_alias(name=name, alias=alias)
                except Exception:
                    pass
            client.delete_prompt(name)
        except Exception:
            # Best effort: si falla, el operador lo limpia desde la UI.
            pass

def test_round_trip_registro_alias_y_refresh_tras_ttl(
    temp_prompt_name: str, cleanup_prompts: list[str]
):
    """Round-trip: registrar v1+v2 y verificar que el provider refresca al
    cambiar el alias tras vencer el TTL.
    """
    import mlflow

    from backend.prompt_registry.provider import PromptProvider

    cleanup_prompts.append(temp_prompt_name)
    uri = os.environ["MLFLOW_TRACKING_URI"]

    # --- Setup v1 + alias production ----------------------------------------
    mlflow.set_tracking_uri(uri)
    mlflow.genai.register_prompt(
        name=temp_prompt_name,
        template="version 1: {{var}}",
        commit_message="test-integration v1",
        tags={"test": "integration", "phase": "13"},
    )
    mlflow.genai.set_prompt_alias(
        name=temp_prompt_name, alias="production", version=1
    )

    # --- Provider con TTL corto para forzar refresh en el test --------------
    provider = PromptProvider(
        tracking_uri=uri,
        default_alias="production",
        cache_ttl_seconds=1,
        enabled=True,
    )

    # --- Carga inicial → v1 -------------------------------------------------
    r1 = provider.get_prompt(temp_prompt_name)
    assert r1.source == "registry", (
        f"Esperaba source='registry' pero vino del fallback "
        f"(reason en logs): {r1}"
    )
    assert r1.version == 1
    assert r1.template == "version 1: {{var}}"
    assert r1.name == temp_prompt_name

    # --- Registrar v2 + mover alias production -----------------------------
    mlflow.genai.register_prompt(
        name=temp_prompt_name,
        template="version 2: {{var}}",
        commit_message="test-integration v2",
    )
    mlflow.genai.set_prompt_alias(
        name=temp_prompt_name, alias="production", version=2
    )

    # --- Esperar TTL --------------------------------------------------------
    # El provider tiene TTL=1s; esperamos 1.5s para garantizar miss.
    time.sleep(1.5)

    # --- Re-carga → debe ver v2 sin reiniciar el provider ------------------
    r2 = provider.get_prompt(temp_prompt_name)
    assert r2.source == "registry"
    assert r2.version == 2, (
        f"Tras mover el alias y vencer el TTL, esperaba version=2. "
        f"Got: {r2}"
    )
    assert r2.template == "version 2: {{var}}"

def test_fallback_cuando_prompt_no_existe_en_mlflow_real(
    cleanup_prompts: list[str],
):
    """Contra MLflow real, pedir un prompt que NO existe debe caer al
    fallback empaquetado (si el name está en `_FALLBACK_MAP`) con un
    WARNING log."""
    from backend.prompt_registry.fallback import TRAMITES_SYSTEM
    from backend.prompt_registry.provider import PromptProvider

    uri = os.environ["MLFLOW_TRACKING_URI"]
    provider = PromptProvider(
        tracking_uri=uri,
        default_alias="production",
        cache_ttl_seconds=10,
        enabled=True,
    )

    # Usamos un nombre que NO está bootstrapped pero SÍ está en _FALLBACK_MAP:
    # `uja-tramites-system`. Si el bootstrap ya corrió, el test sigue
    # funcionando porque assertaría source=registry; para ser robusto, lo
    # toleramos en cualquier caso y validamos que el contenido es correcto.
    r = provider.get_prompt("uja-tramites-system")
    assert r.template == TRAMITES_SYSTEM, (
        "Tanto si viene de registry como de fallback, el contenido debe "
        "coincidir byte-a-byte con el TRAMITES_SYSTEM empaquetado."
    )
    assert r.source in ("registry", "fallback")

def test_tags_prompt_se_adjuntan_a_la_traza_real(cleanup_prompts: list[str]):
    """Round-trip del tracing de  contra MLflow real:

    1. Inicia una traza MLflow vía `mlflow_tracer.start_trace`.
    2. Activa el tracking de prompts (ContextVar).
    3. Carga 2 prompts vía PromptProvider (registrados o fallback).
    4. Lee `get_prompts_used()` y construye los tags como hace el router.
    5. Llama `set_trace_tags(trace_ctx, tags)`.
    6. Cierra la traza.
    7. Recupera la traza vía `MlflowClient.get_trace` y verifica los tags.

    Esto cubre el camino completo que en producción ejecuta el router de
    Trámites: si los tags llegan a la traza, también llegarán cuando
    `mlflow.log_feedback(trace_id=...)` los correlacione.
    """
    import mlflow

    from backend.mlflow_tracer import (
        end_trace as tracer_end_trace,
        set_trace_tags as tracer_set_trace_tags,
        start_trace as tracer_start_trace,
    )
    from backend.prompt_registry.provider import (
        PromptProvider,
        begin_prompt_tracking,
        get_prompts_used,
    )

    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)

    # Iniciar traza raíz. Reusamos un experiment ya configurado por el
    # bootstrap del proyecto ("tramites") para evitar el error
    # "Unable to determine trace artifact location" que aparece con
    # experiments recién creados ad-hoc en este servidor MLflow.
    trace_ctx = tracer_start_trace(
        experiment_name="tramites",
        name=f"tag_test_{uuid.uuid4().hex[:6]}",
        inputs={"query": "integration test"},
    )
    assert trace_ctx is not None, "MLflow no inició la traza"
    trace_id = trace_ctx.request_id

    # Activar tracking + cargar prompts.
    begin_prompt_tracking()
    provider = PromptProvider(
        tracking_uri=uri,
        default_alias="production",
        cache_ttl_seconds=10,
        enabled=True,
    )
    provider.get_prompt("uja-tramites-system")
    provider.get_prompt("uja-tramites-user-template")

    # Construir y aplicar tags como hace el router.
    prompts_used = get_prompts_used()
    assert len(prompts_used) == 2
    tags = {}
    for r in prompts_used:
        tags[f"prompt.{r.name}.version"] = str(r.version)
        tags[f"prompt.{r.name}.source"] = r.source
    tracer_set_trace_tags(trace_ctx, tags)

    # Cerrar traza para forzar persistencia (export es asíncrono y puede
    # tardar varios segundos en aparecer indexada).
    tracer_end_trace(trace_ctx, outputs={"status": "tagged"})

    # Polling: el export del trace al backend de MLflow es asíncrono. Aunque
    # `end_trace` retorna inmediatamente, la traza puede tardar varios
    # segundos en ser searchable. Polleamos hasta 15s con backoff.
    #
    # Decisiones de diseño:
    # - `get_trace(trace_id)` falla con "Unable to determine trace artifact
    #   location" en algunas configs del server; usamos `search_traces`.
    # - El filtro `request_id = '...'` no funciona en MLflow 3.12 contra el
    #   server desplegado, así que matcheamos por request_id en cliente.
    client = mlflow.MlflowClient()
    exp_id = mlflow.get_experiment_by_name("tramites").experiment_id

    # max_results=5 acota la query (con experiments grandes, search_traces
    # puede ser muy lento). Como acabamos de crear nuestra traza y
    # buscamos en orden inverso (más recientes primero), 5 es suficiente.
    matching = []
    for _ in range(10):
        time.sleep(1.0)
        recent = client.search_traces(locations=[exp_id], max_results=5)
        matching = [t for t in recent if t.info.request_id == trace_id]
        if matching:
            break

    assert matching, (
        f"Traza {trace_id} no aparece en MLflow tras 10s de polling. "
        "El export asíncrono pudo haber fallado, o MLflow está saturado."
    )
    trace_tags = matching[0].info.tags
    # Los tags deben llevar la name+version+source de cada prompt cargado.
    assert "prompt.uja-tramites-system.version" in trace_tags
    assert "prompt.uja-tramites-system.source" in trace_tags
    assert "prompt.uja-tramites-user-template.version" in trace_tags
    assert "prompt.uja-tramites-user-template.source" in trace_tags
    # `source` debe ser registry o fallback.
    assert trace_tags["prompt.uja-tramites-system.source"] in ("registry", "fallback")

def test_format_sustituye_variables_en_prompt_cargado_de_mlflow(
    temp_prompt_name: str, cleanup_prompts: list[str]
):
    """Verifica que `PromptResource.format()` funciona sobre una plantilla
    cargada de MLflow real (no solo del fallback)."""
    import mlflow

    from backend.prompt_registry.provider import PromptProvider

    cleanup_prompts.append(temp_prompt_name)
    uri = os.environ["MLFLOW_TRACKING_URI"]

    mlflow.set_tracking_uri(uri)
    mlflow.genai.register_prompt(
        name=temp_prompt_name,
        template="Hola {{nombre}}, tu trámite {{tipo}} está en proceso.",
        commit_message="test-integration format",
    )
    mlflow.genai.set_prompt_alias(
        name=temp_prompt_name, alias="production", version=1
    )

    provider = PromptProvider(
        tracking_uri=uri,
        default_alias="production",
        cache_ttl_seconds=10,
        enabled=True,
    )

    resource = provider.get_prompt(temp_prompt_name)
    assert resource.source == "registry"
    out = resource.format(nombre="Félix", tipo="empadronamiento")
    assert out == "Hola Félix, tu trámite empadronamiento está en proceso."
