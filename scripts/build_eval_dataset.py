"""
Construye un Evaluation Dataset MLflow a partir de traces anotados con expected_response.

Uso:
    python -m scripts.build_eval_dataset --experiment chatbot --output-name eval-chatbot-2026-05

Selecciona los traces del experiment indicado que tengan registrado un assessment
`expected_response` (ground truth humano o LLM judge), crea un dataset
versionable via `mlflow.genai.datasets.create_dataset` y le mergea los traces.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

DATASET_TAGS = {"team": "uja", "stage": "validation"}
# Excluimos trazas que ademas tienen expected_facts: el scorer Correctness exige
# uno u otro, no ambos. Si necesitas evaluar solo facts, ajusta el filtro.
EXPECTED_RESPONSE_FILTER = (
    "expectation.expected_response IS NOT NULL AND expectation.expected_facts IS NULL"
)

def _resolve_experiment_id(experiment_name: str) -> str:
    import mlflow

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"Experimento no encontrado en MLflow: {experiment_name!r}")
    return exp.experiment_id

def build_dataset(experiment: str, output_name: str) -> str:
    """Construye el dataset y devuelve el dataset_id."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI no esta configurado.")

    import mlflow
    import mlflow.genai.datasets

    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _resolve_experiment_id(experiment)

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=EXPECTED_RESPONSE_FILTER,
        return_type="list",
    )
    if not traces:
        raise SystemExit(
            f"No se encontraron traces con expected_response en el experiment {experiment!r}."
        )
    logger.info("Encontrados %d traces anotados en %s.", len(traces), experiment)

    dataset = mlflow.genai.datasets.create_dataset(
        name=output_name,
        experiment_id=experiment_id,
        tags=DATASET_TAGS,
    )
    dataset.merge_records(traces)
    logger.info(
        "Dataset creado: id=%s name=%s records=%d",
        dataset.dataset_id, dataset.name, len(traces),
    )
    return dataset.dataset_id

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True,
        help="Nombre del experimento MLflow (p.ej. chatbot, react-agent, rag, tramites, comparador).",
    )
    parser.add_argument(
        "--output-name", required=True,
        help="Nombre del dataset a crear en MLflow GenAI.",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    dataset_id = build_dataset(args.experiment, args.output_name)
    print(dataset_id)
    return 0

if __name__ == "__main__":
    sys.exit(main())
