"""PromptProvider — carga prompts de MLflow Prompt Registry con caché TTL y fallback.

Punto de entrada público del módulo prompt_registry. Encapsula:

- Resolución de referencias `prompts:/<name>@<alias>` contra MLflow Prompt Registry.
- Caché in-memory con TTL configurable (delegada en `cache.TTLCache`).
- Fallback silencioso al prompt empaquetado en `fallback.py` cuando MLflow no
  responde, el prompt no existe o el flag `PROMPT_REGISTRY_ENABLED` está en
  False. Cada fallback emite `logging.WARNING` identificando prompt y razón.

Uso típico (desde un nodo LangGraph o un router):

    from backend.prompt_registry.provider import get_provider

    provider = get_provider()
    system_resource = provider.get_prompt("uja-tramites-system")
    system_text = system_resource.template

    user_resource = provider.get_prompt("uja-tramites-user-template")
    user_text = user_resource.format(contexto=ctx, query=q, ...)

Convención de naming: hyphens, no slashes. MLflow rechaza nombres con `/`
porque las URIs `prompts:/<name>@<alias>` reservan el slash como separador.

"""

from __future__ import annotations

import contextvars
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

from backend.config import settings
from backend.prompt_registry.cache import TTLCache
from backend.prompt_registry.fallback import get_fallback_prompt

logger = logging.getLogger(__name__)

PromptSource = Literal["registry", "fallback"]

_TEMPLATE_VAR_RE = re.compile(r"\{\{(\w+)\}\}")

# ---------------------------------------------------------------------------
# Tracking de prompts usados durante una request (ContextVar async-safe).
#
# Permite que los routers SSE adjunten tags `prompt.<name>.version|source` a la
# traza MLflow ya emitida. Sin esto, el provider serviría prompts
# sin dejar rastro de qué versión usó cada request.
#
# Por defecto el ContextVar es None (= "no tracking activo"); el router llama
# a `begin_prompt_tracking()` al inicio de la request y a `get_prompts_used()`
# al final para leer la lista. ContextVar aisla por task asyncio: dos
# requests concurrentes no se contaminan.
# ---------------------------------------------------------------------------

_prompts_used_var: contextvars.ContextVar[Optional[list["PromptResource"]]] = (
    contextvars.ContextVar("prompt_registry_used", default=None)
)

def begin_prompt_tracking() -> None:
    """Inicializa una lista vacía en el ContextVar de la task actual.

    Llamada típica: al inicio del handler de una request SSE. Sin esta
    llamada, `get_prompt` no registra nada (la lista queda como None).
    """
    _prompts_used_var.set([])

def get_prompts_used() -> list["PromptResource"]:
    """Devuelve los `PromptResource` cargados durante esta request.

    Lista vacía si no se llamó a `begin_prompt_tracking()` o si ningún
    prompt fue cargado.
    """
    current = _prompts_used_var.get()
    return list(current) if current else []

def _record_prompt_used(resource: "PromptResource") -> None:
    """Interno: agrega un PromptResource a la lista del ContextVar si está
    activo. Llamado por `PromptProvider.get_prompt`."""
    current = _prompts_used_var.get()
    if current is not None:
        current.append(resource)

@dataclass(frozen=True)
class PromptResource:
    """Plantilla cargada (de MLflow o del fallback) lista para usar."""

    name: str
    template: str
    version: int  # 0 cuando viene del fallback empaquetado.
    source: PromptSource

    def format(self, **values: object) -> str:
        """Sustituye marcadores `{{var}}` en la plantilla por los `values` dados.

        Levanta `KeyError` si la plantilla referencia una variable no provista.
        Variables sobrantes en `values` se ignoran (comportamiento permisivo
        para que un cambio en el registry que retire una variable no rompa el
        runtime hasta el siguiente despliegue).
        """
        required = set(_TEMPLATE_VAR_RE.findall(self.template))
        missing = required - set(values.keys())
        if missing:
            raise KeyError(
                f"Faltan variables para plantilla '{self.name}': {sorted(missing)}"
            )
        result = self.template
        for key, value in values.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result

class PromptProvider:
    """Resolutor de prompts con caché TTL y fallback al empaquetado.

    Es seguro instanciarlo varias veces (cada instancia tiene su propia caché).
    Para uso compartido en runtime, usa el singleton expuesto por
    `get_provider()`.
    """

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        default_alias: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._tracking_uri = (
            tracking_uri if tracking_uri is not None
            else settings.MLFLOW_TRACKING_URI
        )
        self._default_alias = (
            default_alias if default_alias is not None
            else settings.PROMPT_REGISTRY_ALIAS
        )
        self._enabled = (
            enabled if enabled is not None
            else settings.PROMPT_REGISTRY_ENABLED
        )
        ttl = (
            cache_ttl_seconds if cache_ttl_seconds is not None
            else settings.PROMPT_REGISTRY_CACHE_TTL_SECONDS
        )
        self._cache: TTLCache = TTLCache(ttl_seconds=ttl)

    def get_prompt(self, name: str, *, alias: Optional[str] = None) -> PromptResource:
        """Devuelve la plantilla `name@alias` desde la caché, MLflow o fallback.

        Resolución en orden:
        1. Caché si `(name, alias)` no ha expirado.
        2. MLflow si `enabled` y la carga tiene éxito.
        3. Fallback empaquetado en cualquier otro caso.

        El resultado siempre se cachea con el TTL configurado, incluido el
        fallback (evita martillear MLflow si está caído).
        """
        effective_alias = alias or self._default_alias
        cache_key = (name, effective_alias)

        cached = self._cache.get(cache_key)
        if cached is not None:
            _record_prompt_used(cached)
            return cached

        if not self._enabled:
            resource = self._build_fallback(name, reason="registry_disabled")
        else:
            try:
                resource = self._load_from_mlflow(name, effective_alias)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "prompt_registry: fallback para %s@%s (load falló: %s: %s)",
                    name, effective_alias, type(exc).__name__, exc,
                )
                resource = self._build_fallback(
                    name, reason=f"mlflow_error:{type(exc).__name__}"
                )

        self._cache.set(cache_key, resource)
        _record_prompt_used(resource)
        return resource

    def _load_from_mlflow(self, name: str, alias: str) -> PromptResource:
        # Import perezoso: permite que el módulo se importe en tests
        # unitarios que mocken mlflow sin necesidad de tenerlo configurado.
        import mlflow

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        prompt_uri = f"prompts:/{name}@{alias}"
        prompt_version = mlflow.genai.load_prompt(prompt_uri)
        return PromptResource(
            name=name,
            template=prompt_version.template,
            version=int(prompt_version.version),
            source="registry",
        )

    def _build_fallback(self, name: str, *, reason: str) -> PromptResource:
        template = get_fallback_prompt(name)
        if template is None:
            raise KeyError(
                f"No hay fallback empaquetado para prompt '{name}'. "
                f"Añádelo a backend/prompt_registry/fallback.py "
                f"(reason: {reason})."
            )
        logger.warning(
            "prompt_registry: usando fallback empaquetado para %s (reason=%s)",
            name, reason,
        )
        return PromptResource(
            name=name,
            template=template,
            version=0,
            source="fallback",
        )

# ---------------------------------------------------------------------------
# Singleton perezoso para uso desde nodos LangGraph y routers.
# Tarea 3.1 del plan exige inicialización perezosa a nivel de módulo.
# Los tests deben instanciar `PromptProvider(...)` directo con overrides
# en lugar de tocar el singleton.
# ---------------------------------------------------------------------------

_provider: Optional[PromptProvider] = None

def get_provider() -> PromptProvider:
    """Devuelve la instancia singleton del PromptProvider (lazy init)."""
    global _provider
    if _provider is None:
        _provider = PromptProvider()
    return _provider

def reset_provider() -> None:
    """Resetea el singleton — útil entre tests."""
    global _provider
    _provider = None
