"""Script de bootstrap del Prompt Registry.

Registra los 4 prompts iniciales en MLflow desde las constantes empaquetadas
en `fallback.py` y crea los aliases `staging` y `production` apuntando a v1.

Idempotente: re-ejecutar el script en un MLflow ya inicializado no machaca
versiones existentes ni mueve aliases que ya estén puestos.

Reglas de idempotencia:

- v1 del prompt no existe en MLflow → se registra con el contenido del
  fallback empaquetado (`fallback._FALLBACK_MAP`).
- v1 existe y su template coincide byte-a-byte con el fallback → no se
  hace nada con el contenido (caso re-run sin cambios).
- v1 existe pero el contenido difiere → se EMITE WARNING y se DEJA EN PAZ.
  El operador modificó v1 directamente en MLflow UI; el script no
  sobreescribe (regla "el operador manda en MLflow").
- Alias staging/production no existe → se crea apuntando a v1.
- Alias existe → se deja como está (el operador puede haberlo movido).

Uso CLI:

    python -m backend.prompt_registry.bootstrap

Lee `MLFLOW_TRACKING_URI` de los settings; si no está, falla con un
mensaje explícito (no tiene sentido bootstrappear contra el server por
defecto de MLflow, que sería local efímero).

"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Literal, Optional

from backend.config import settings
from backend.prompt_registry.fallback import _FALLBACK_MAP

logger = logging.getLogger(__name__)

RegistryAction = Literal["created", "already_matches", "diverged_skipped"]
AliasAction = Literal["created", "already_exists"]

@dataclass(frozen=True)
class PromptBootstrapResult:
    """Resultado de bootstrappear un prompt concreto."""

    name: str
    version: int  # versión a la que apuntan los aliases tras el bootstrap.
    registry_action: RegistryAction
    aliases: dict[str, AliasAction]

# Alias por entorno que crea el bootstrap.
_BOOTSTRAP_ALIASES: tuple[str, ...] = ("staging", "production")

def register_initial_prompts(
    *,
    tracking_uri: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, PromptBootstrapResult]:
    """Registra los 4 prompts iniciales y sus aliases en MLflow.

    Args:
        tracking_uri: si se pasa, se aplica antes de cualquier llamada. Si es
            None, se usa `settings.MLFLOW_TRACKING_URI`. Falla si ambos son None.
        dry_run: si True, no llama a register/set_prompt_alias; solo inspecciona
            y devuelve el plan. Útil para tests y para verificar idempotencia
            sin efectos secundarios.

    Returns:
        Mapa `{name: PromptBootstrapResult}` por cada prompt del fallback.

    Raises:
        RuntimeError: si no hay `tracking_uri` resoluble.
    """
    import mlflow

    effective_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    if not effective_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI no está configurado. Define la variable de "
            "entorno o pasa tracking_uri explícito a register_initial_prompts()."
        )
    mlflow.set_tracking_uri(effective_uri)
    logger.info("Bootstrap contra MLflow %s (dry_run=%s)", effective_uri, dry_run)

    results: dict[str, PromptBootstrapResult] = {}
    for name, fallback_template in _FALLBACK_MAP.items():
        results[name] = _bootstrap_one(name, fallback_template, dry_run=dry_run)
    return results

def _bootstrap_one(
    name: str, fallback_template: str, *, dry_run: bool
) -> PromptBootstrapResult:
    import mlflow

    # 1. ¿Existe v1?
    existing_v1 = mlflow.genai.load_prompt(
        f"prompts:/{name}/1", allow_missing=True
    )

    if existing_v1 is None:
        # No existe → registrar.
        if dry_run:
            logger.info("[dry_run] registraría %s v1 (len=%d)", name, len(fallback_template))
            registry_action: RegistryAction = "created"
            version = 1
        else:
            prompt_version = mlflow.genai.register_prompt(
                name=name,
                template=fallback_template,
                commit_message=(
                    "Initial bootstrap from packaged fallback "
                    "(backend/prompt_registry/fallback.py)."
                ),
                tags={"phase": "13", "bootstrap": "true"},
            )
            registry_action = "created"
            version = int(prompt_version.version)
            logger.info("Registrado %s v%d", name, version)
    else:
        # v1 existe — chequear contenido.
        if existing_v1.template == fallback_template:
            registry_action = "already_matches"
            version = 1
            logger.info("%s v1 ya existe y coincide con el fallback (skip)", name)
        else:
            registry_action = "diverged_skipped"
            version = 1
            logger.warning(
                "%s v1 existe pero su contenido DIFIERE del fallback empaquetado. "
                "No se sobreescribe (el operador editó v1 en MLflow). Sincroniza "
                "backend/prompt_registry/fallback.py si el cambio es intencional.",
                name,
            )

    # 2. Aliases staging y production.
    aliases_result: dict[str, AliasAction] = {}
    for alias in _BOOTSTRAP_ALIASES:
        existing_alias = mlflow.genai.load_prompt(
            f"prompts:/{name}@{alias}", allow_missing=True
        )
        if existing_alias is not None:
            aliases_result[alias] = "already_exists"
            logger.info("Alias %s@%s ya existe (skip)", name, alias)
            continue
        if registry_action == "diverged_skipped":
            # No tocamos aliases si el contenido de v1 no es el esperado:
            # crear un alias pointing a v1 puede ser indeseable para el operador.
            aliases_result[alias] = "already_exists"  # tratamos como "no acción"
            logger.warning(
                "Alias %s@%s no existe pero v1 diverge — NO se crea. "
                "Asígnalo manualmente desde MLflow UI.",
                name, alias,
            )
            continue
        if dry_run:
            logger.info("[dry_run] crearía alias %s@%s -> v%d", name, alias, version)
            aliases_result[alias] = "created"
        else:
            mlflow.genai.set_prompt_alias(name=name, alias=alias, version=version)
            aliases_result[alias] = "created"
            logger.info("Alias %s@%s -> v%d creado", name, alias, version)

    return PromptBootstrapResult(
        name=name,
        version=version,
        registry_action=registry_action,
        aliases=aliases_result,
    )

def _print_summary(results: dict[str, PromptBootstrapResult]) -> None:
    """Imprime un resumen humano de lo hecho por el bootstrap."""
    print()
    print("Resumen del bootstrap del Prompt Registry:")
    print("=" * 72)
    for name, r in results.items():
        print(f"  {name}")
        print(f"    versión: v{r.version}")
        print(f"    registro: {r.registry_action}")
        for alias, action in r.aliases.items():
            print(f"    alias {alias}: {action}")
    print("=" * 72)
    print()

def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint CLI: `python -m backend.prompt_registry.bootstrap [--dry-run]`."""
    argv = sys.argv[1:] if argv is None else argv
    dry_run = "--dry-run" in argv

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        results = register_initial_prompts(dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.error("Bootstrap falló: %s: %s", type(exc).__name__, exc)
        return 1

    _print_summary(results)
    return 0

if __name__ == "__main__":
    sys.exit(main())
