"""Bootstrap del dataset MLflow GenAI para la evaluación del RAG de trámites.

Uso:
    MLFLOW_TRACKING_URI=http://localhost:5001 \\
        python -m scripts.bootstrap_eval_dataset
    # opcional: --suffix custom-tag para sobreescribir el sufijo de fecha
    # opcional: --force para re-mergear records si el dataset ya existe

Lee `backend/rag_tramites_eval/seed_queries.SEED_QUERIES`, expande los
listados ALL → IDs reales del catálogo, y registra cada entrada como un
record (inputs + expectations + tags) en el dataset GenAI
`rag-tramites-eval-baseline-YYYY-MM-DD` dentro del experimento dedicado
`rag-tramites-eval`.

Idempotente: si el dataset ya existe, log warning y termina sin sobrescribir.
Pasa `--force` para re-mergear (la API de MLflow es upsert).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from typing import Any

from backend.rag_tramites_eval.catalog import Catalog, _normalize_municipio, load_catalog
from backend.rag_tramites_eval.seed_queries import EXPECTED_ALL_TRAMITES, SEED_QUERIES

EXPERIMENT_NAME = "rag-tramites-eval"
DATASET_PREFIX = "rag-tramites-eval-baseline"
DATASET_TAGS = {"team": "uja", "stage": "validation", "domain": "tramites"}

logger = logging.getLogger(__name__)

def _expand_expected_ids(
    expected: list[str] | str,
    municipio: str | None,
    catalog: Catalog,
) -> list[str]:
    """Resuelve el sentinel `EXPECTED_ALL_TRAMITES` a todos los IDs del municipio."""
    if expected == EXPECTED_ALL_TRAMITES:
        if not municipio:
            raise ValueError(
                "EXPECTED_ALL_TRAMITES exige expected_municipio no nulo"
            )
        bucket = catalog.by_municipio.get(_normalize_municipio(municipio), {})
        return sorted(bucket.keys())
    return list(expected)

def _seed_to_record(seed: dict, catalog: Catalog) -> dict[str, Any]:
    """Convierte una entrada de SEED_QUERIES al schema de merge_records."""
    expected_ids = _expand_expected_ids(
        seed["expected_tramite_ids"], seed["expected_municipio"], catalog
    )
    is_listado = seed["expected_tramite_ids"] == EXPECTED_ALL_TRAMITES
    is_no_municipio = seed["expected_municipio"] is None
    if seed["expected_scope"] == "out":
        category = "scope_out"
    elif is_listado:
        category = "listado"
    elif is_no_municipio:
        category = "sin_municipio"
    else:
        category = "especifica"
    return {
        "inputs": {"query": seed["query"]},
        "expectations": {
            "expected_response": seed["expected_response"],
            "expected_tramite_ids": expected_ids,
            "expected_municipio": seed["expected_municipio"],
            "expected_scope": seed["expected_scope"],
        },
        "tags": {"category": category},
    }

def _dataset_name(suffix: str | None) -> str:
    return f"{DATASET_PREFIX}-{suffix or date.today().isoformat()}"

def bootstrap(suffix: str | None, force: bool) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI no está configurado.")

    import mlflow
    import mlflow.genai.datasets as datasets

    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("Experimento: %s (id=%s)", EXPERIMENT_NAME, exp.experiment_id)

    name = _dataset_name(suffix)
    try:
        existing = datasets.get_dataset(name=name)
    except Exception:
        existing = None

    if existing and not force:
        logger.warning(
            "Dataset %r ya existe (id=%s). Pasa --force para re-mergear.",
            name, existing.dataset_id,
        )
        return existing.dataset_id

    catalog = load_catalog()
    records = [_seed_to_record(s, catalog) for s in SEED_QUERIES]
    logger.info("Records preparados: %d", len(records))

    if existing:
        dataset = existing
        logger.warning("--force: re-mergeando sobre dataset existente %s", dataset.dataset_id)
    else:
        dataset = datasets.create_dataset(
            name=name,
            experiment_id=exp.experiment_id,
            tags=DATASET_TAGS,
        )
        logger.info("Dataset creado: id=%s name=%s", dataset.dataset_id, dataset.name)

    dataset.merge_records(records)
    logger.info(
        "Merge completo: %d records en dataset %s (id=%s)",
        len(records), name, dataset.dataset_id,
    )
    return dataset.dataset_id

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix", default=None,
        help="Sufijo del dataset (default: fecha de hoy en YYYY-MM-DD).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-mergear records si el dataset ya existe (upsert).",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    dataset_id = bootstrap(args.suffix, args.force)
    print(dataset_id)
    return 0

if __name__ == "__main__":
    sys.exit(main())
