"""Runner de evaluación del RAG de trámites — invoca el grafo real y mide.

Uso:
    MLFLOW_TRACKING_URI=http://localhost:5001 \\
        python -m scripts.run_tramites_eval \\
        --dataset-name rag-tramites-eval-baseline-2026-05-30

A diferencia de `scripts/run_evaluation.py`, aquí `predict_fn`
NO llama a `/v1/chat/completions` directamente, sino que invoca el grafo
completo de `backend.rag_tramites.graph.tramites_graph` para obtener:

- `response`: respuesta final del LLM (tras guardrail).
- `retrieved_ids`: IDs de los trámites devueltos por retrieve+rerank.
- `extracted_municipio`: lo que el `node_detect_municipio` detectó.

Esos tres campos alimentan los 4 scorers custom de
`backend.rag_tramites_eval.scorers`. Recall y precisión de URLs se miden
contra `data/export.json`.

Cada ejecución crea un run nuevo en el experimento `rag-tramites-eval`
con tags `prompt.uja-tramites-system.version` y `prompt.uja-tramites-system.source`
para trazabilidad de la versión de prompt usada (heredado de ).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Any

from backend.rag_tramites_eval.catalog import load_catalog
from backend.rag_tramites_eval.scorers import (
    municipio_match,
    precision_urls,
    recall_tramites,
    scope_mismatch_correct,
)

EXPERIMENT_NAME = "rag-tramites-eval"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scorers custom envueltos para mlflow.genai.evaluate
# ---------------------------------------------------------------------------

def _build_scorers():
    """Construye los 4 scorers custom envueltos con @scorer y cierre del catálogo."""
    from mlflow.genai.scorers import scorer

    catalog = load_catalog()

    @scorer
    def recall_tramites_score(outputs: Any, expectations: dict) -> float:
        return recall_tramites(
            expected_ids=expectations.get("expected_tramite_ids") or [],
            retrieved_ids=(outputs or {}).get("retrieved_ids") or [],
        )

    @scorer
    def precision_urls_score(outputs: Any, expectations: dict) -> float:
        return precision_urls(
            response_text=(outputs or {}).get("response", ""),
            expected_municipio=expectations.get("expected_municipio"),
            catalog=catalog,
        )

    @scorer
    def municipio_match_score(outputs: Any, expectations: dict) -> float:
        return municipio_match(
            extracted=(outputs or {}).get("extracted_municipio"),
            expected=expectations.get("expected_municipio"),
        )

    @scorer
    def scope_mismatch_score(outputs: Any, expectations: dict) -> float:
        return scope_mismatch_correct(
            response_text=(outputs or {}).get("response", ""),
            expected_scope=expectations.get("expected_scope", "in"),
        )

    return [
        recall_tramites_score,
        precision_urls_score,
        municipio_match_score,
        scope_mismatch_score,
    ]

# ---------------------------------------------------------------------------
# predict_fn: invoca el grafo real
# ---------------------------------------------------------------------------

def _make_real_predict_fn():
    """predict_fn que invoca el grafo completo de rag_tramites por consulta."""
    from backend.rag_tramites.graph.tramites_graph import graph

    def predict(query: str) -> dict[str, Any]:
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": session_id}}
        input_state = {
            "session_id": session_id,
            "query_actual": query,
            "messages": [{"role": "user", "content": query}],
            "municipio_detectado": None,
            "intent": "especifico",
            "candidatos": [],
            "candidatos_reranked": [],
            "respuesta_final": "",
            "necesita_refinamiento": False,
            "turno": 0,
        }
        result = asyncio.run(graph.ainvoke(input_state, config=config))
        retrieved_ids = [
            r.get("metadata", {}).get("id")
            for r in result.get("candidatos_reranked", [])
            if r.get("metadata", {}).get("id")
        ]
        return {
            "response": result.get("respuesta_final", ""),
            "retrieved_ids": retrieved_ids,
            "extracted_municipio": result.get("municipio_detectado"),
        }

    return predict

# ---------------------------------------------------------------------------
# Tagging del run con prompt_version
# ---------------------------------------------------------------------------

def _tag_run_with_prompt_version(run_id: str) -> None:
    """Lee la versión activa del prompt Trámites y la setea como tag del run.

     ya hace este tagging por traza dentro del flujo de chat. En el
    contexto de evaluación, lo replicamos a nivel de run para que el report
    pueda referenciar la versión de prompt sin tener que abrir cada traza.
    """
    try:
        from mlflow import MlflowClient

        from backend.prompt_registry.provider import get_provider

        resource = get_provider().get_prompt("uja-tramites-system")
        client = MlflowClient()
        client.set_tag(run_id, "prompt.uja-tramites-system.version", str(resource.version))
        client.set_tag(run_id, "prompt.uja-tramites-system.source", resource.source)
        logger.info(
            "Run %s tagged: version=%s source=%s",
            run_id, resource.version, resource.source,
        )
    except Exception as e:
        logger.warning("No se pudo etiquetar el run con prompt_version: %s", e)

# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict[str, Any]) -> None:
    if not metrics:
        print("(sin métricas)")
        return
    width = max(len(k) for k in metrics) + 2
    print()
    print(f"{'Métrica'.ljust(width)} Valor")
    print(f"{'-' * width} -----")
    for k in sorted(metrics):
        v = metrics[k]
        value_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k.ljust(width)} {value_str}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(dataset_name: str) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI no está configurado.")

    import mlflow
    import mlflow.genai
    import mlflow.genai.datasets as datasets

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("Experimento: %s (id=%s)", EXPERIMENT_NAME, exp.experiment_id)

    dataset = datasets.get_dataset(name=dataset_name)
    logger.info("Dataset: %s (id=%s)", dataset.name, dataset.dataset_id)

    scorers = _build_scorers()
    predict_fn = _make_real_predict_fn()

    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    logger.info("Run de evaluación creado: %s", result.run_id)
    _tag_run_with_prompt_version(result.run_id)
    _print_summary(result.metrics or {})
    return result.run_id

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name", required=True,
        help="Nombre del dataset MLflow GenAI (creado por bootstrap_eval_dataset).",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    run_id = run(args.dataset_name)
    print()
    print(f"run_id: {run_id}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
