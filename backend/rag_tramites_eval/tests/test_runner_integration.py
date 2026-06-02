"""Test de integración del runner contra MLflow real (skip-on-missing).

Verifica end-to-end (sin LLM real) que:

1. El bootstrap puede crear un dataset MLflow GenAI con records custom.
2. `mlflow.genai.evaluate` con `predict_fn` stub corre los 4 scorers custom
   y produce un run con metrics por scorer.
3. `_tag_run_with_prompt_version` deja el tag `prompt.uja-tramites-system.version`
   en el run (heredado de ).

Marcado con `pytest.mark.integration`: solo se ejecuta cuando
`MLFLOW_TRACKING_URI` está configurado y el server responde. CI lo skipea
automáticamente porque la env no está seteada.

"""

from __future__ import annotations

import os
import uuid

import pytest

def _mlflow_reachable() -> bool:
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
def temp_dataset_name() -> str:
    return f"rag-tramites-eval-itest-{uuid.uuid4().hex[:10]}"

@pytest.fixture
def cleanup_datasets():
    """Best-effort delete de datasets al final del test."""
    names: list[str] = []
    yield names
    if not names:
        return
    import mlflow.genai.datasets as datasets

    for name in names:
        try:
            ds = datasets.get_dataset(name=name)
            datasets.delete_dataset(dataset_id=ds.dataset_id)
        except Exception:
            pass

def _stub_predict(query: str) -> dict:
    """predict_fn determinista que SIEMPRE devuelve la misma respuesta perfecta
    para el primer record del dataset y vacía para el segundo. Sirve para
    verificar que los scorers ven datos reales sin invocar LLM."""
    if "adamuz" in query.lower():
        return {
            "response": "El trámite Volante de Empadronamiento de Adamuz: https://e-admin.eprinsa.es/adamuz/x",
            "retrieved_ids": ["adamuz-volante-de-empadronamiento"],
            "extracted_municipio": "Adamuz",
        }
    return {
        "response": "",
        "retrieved_ids": [],
        "extracted_municipio": None,
    }

def test_runner_evaluate_con_scorers_custom_y_tag_prompt_version(
    temp_dataset_name: str, cleanup_datasets: list[str]
):
    import mlflow
    import mlflow.genai
    import mlflow.genai.datasets as datasets

    from scripts.run_tramites_eval import (
        EXPERIMENT_NAME,
        _build_scorers,
        _tag_run_with_prompt_version,
    )

    cleanup_datasets.append(temp_dataset_name)

    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(EXPERIMENT_NAME)

    # Dataset de 2 records.
    dataset = datasets.create_dataset(
        name=temp_dataset_name,
        experiment_id=exp.experiment_id,
        tags={"team": "uja", "stage": "itest"},
    )
    dataset.merge_records([
        {
            "inputs": {"query": "Volante de empadronamiento en Adamuz"},
            "expectations": {
                "expected_response": "Adamuz tiene Volante de Empadronamiento",
                "expected_tramite_ids": ["adamuz-volante-de-empadronamiento"],
                "expected_municipio": "Adamuz",
                "expected_scope": "in",
            },
            "tags": {"category": "especifica"},
        },
        {
            "inputs": {"query": "Trámites en un municipio desconocido"},
            "expectations": {
                "expected_response": "No hay trámites",
                "expected_tramite_ids": [],
                "expected_municipio": None,
                "expected_scope": "in",
            },
            "tags": {"category": "sin_municipio"},
        },
    ])

    # Ejecutar evaluate con predict_fn stub.
    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=_stub_predict,
        scorers=_build_scorers(),
    )
    assert result.run_id, "evaluate no devolvió run_id"

    # Tag de prompt_version (replicar lo que hace el runner real).
    _tag_run_with_prompt_version(result.run_id)

    # Verificar tags en el run.
    client = mlflow.MlflowClient()
    run = client.get_run(result.run_id)
    assert "prompt.uja-tramites-system.version" in run.data.tags, (
        f"Esperaba tag prompt.uja-tramites-system.version en el run; tags: {run.data.tags}"
    )
    assert "prompt.uja-tramites-system.source" in run.data.tags

    # Verificar que los 4 scorers produjeron métricas en el run.
    metric_keys = set(run.data.metrics.keys())
    expected_scorer_names = {
        "recall_tramites_score",
        "precision_urls_score",
        "municipio_match_score",
        "scope_mismatch_score",
    }
    # MLflow agrega los nombres con sufijos (/mean, /min, /max...). Aceptamos
    # cualquier métrica que contenga el nombre del scorer como prefijo.
    for name in expected_scorer_names:
        assert any(name in k for k in metric_keys), (
            f"Esperaba métrica con prefijo {name!r} en el run; "
            f"metric_keys observadas: {sorted(metric_keys)}"
        )
