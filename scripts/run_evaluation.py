"""
Ejecuta `mlflow.genai.evaluate` sobre un dataset de evaluacion previamente creado.

Uso:
    python -m scripts.run_evaluation \\
        --dataset-name eval-chatbot-2026-05-06 \\
        --target-uri http://host.docker.internal:12021/v1 \\
        --model alia-40b-instruct

Para cada registro del dataset, llama al endpoint OpenAI-compat indicado por
`--target-uri` con la `query` como mensaje de usuario. Despues puntua cada
respuesta con los scorers integrados `Correctness` y `RelevanceToQuery` y
imprime un resumen tabular con las metricas medias.

NOTA sobre el juez LLM:
    Los scorers necesitan un LLM juez fiable que produzca el esquema JSON
    estructurado que MLflow espera. ALIA local NO funciona de forma fiable
    como juez (genera prosa libre en lugar del JSON requerido). Para obtener
    metricas validas, configurar un juez GPT-class:

        --judge-uri https://api.openai.com/v1 \\
        --judge-model gpt-4o-mini \\
        --judge-api-key sk-...

    Para detalles y alternativas sobre el juez LLM, ver la sección
    "Configuración del juez LLM" en este mismo módulo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0

def _make_predict_fn(
    target_uri: str,
    model: Optional[str],
    timeout_s: float,
    api_key: Optional[str],
):
    """Crea una predict_fn sincrona que llama a un endpoint OpenAI-compatible.

    Acepta un parametro `query` (extraido de los `inputs` del dataset) y
    devuelve el texto de la respuesta del LLM.
    """
    base = target_uri.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def predict(query: str) -> str:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        if model:
            payload["model"] = model
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    return predict

def _print_summary(metrics: dict[str, Any]) -> None:
    """Imprime un resumen tabular de las metricas devueltas por evaluate."""
    if not metrics:
        print("(sin metricas)")
        return
    width = max(len(k) for k in metrics) + 2
    print()
    print(f"{'Metric'.ljust(width)} Value")
    print(f"{'-' * width} -----")
    for k in sorted(metrics):
        v = metrics[k]
        if isinstance(v, float):
            value_str = f"{v:.4f}"
        else:
            value_str = str(v)
        print(f"{k.ljust(width)} {value_str}")

def _configure_judge_endpoint(judge_uri: Optional[str], judge_api_key: Optional[str]) -> None:
    """Redirige las llamadas del juez LLM al endpoint indicado.

    Los scorers integrados (`Correctness`, `RelevanceToQuery`) usan el provider
    `openai:/` por defecto, que en MLflow 3.x apunta a `https://api.openai.com`
    sin honorar `OPENAI_API_BASE`. Para apuntarlo a un servidor local OpenAI-
    compat (p.ej. ALIA via llama.cpp), monkey-patcheamos
    `_get_provider_instance` para inyectar `openai_api_base` en la config.
    """
    if judge_api_key:
        os.environ["OPENAI_API_KEY"] = judge_api_key

    if not judge_uri:
        return

    base = judge_uri.rstrip("/")
    os.environ["OPENAI_API_BASE"] = base
    os.environ["OPENAI_BASE_URL"] = base

    from mlflow.metrics.genai import model_utils

    original = model_utils._get_provider_instance

    def patched(provider: str, model: str):
        if provider == "openai":
            from mlflow.gateway.config import EndpointConfig
            from mlflow.gateway.providers.openai import OpenAIConfig, OpenAIProvider

            config = OpenAIConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_base=base,
            )
            endpoint = EndpointConfig(
                name=provider,
                endpoint_type="llm/v1/chat",
                model={
                    "provider": provider,
                    "name": model,
                    "config": config.model_dump(),
                },
            )
            return OpenAIProvider(endpoint)
        return original(provider, model)

    model_utils._get_provider_instance = patched
    logger.info("Juez LLM redirigido a %s", base)

def run(
    dataset_name: str,
    target_uri: str,
    model: Optional[str],
    timeout_s: float,
    api_key: Optional[str],
    judge_uri: Optional[str],
    judge_model: Optional[str],
    judge_api_key: Optional[str],
) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI no esta configurado.")

    _configure_judge_endpoint(judge_uri, judge_api_key)

    import mlflow
    import mlflow.genai
    import mlflow.genai.datasets as datasets
    from mlflow.genai.scorers import Correctness, RelevanceToQuery

    mlflow.set_tracking_uri(tracking_uri)
    dataset = datasets.get_dataset(name=dataset_name)
    logger.info("Dataset %s (id=%s)", dataset.name, dataset.dataset_id)

    predict_fn = _make_predict_fn(target_uri, model, timeout_s, api_key)

    judge_model_uri = f"openai:/{judge_model}" if judge_model else None
    if judge_model_uri:
        logger.info("Judge LLM: %s", judge_model_uri)

    scorer_kwargs: dict[str, Any] = {}
    if judge_model_uri:
        scorer_kwargs["model"] = judge_model_uri

    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[Correctness(**scorer_kwargs), RelevanceToQuery(**scorer_kwargs)],
    )
    logger.info("Run de evaluacion creado: %s", result.run_id)
    _print_summary(result.metrics or {})
    return result.run_id

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name", required=True,
        help="Nombre del dataset de evaluacion (creado con build_eval_dataset.py).",
    )
    parser.add_argument(
        "--target-uri", required=True,
        help="Endpoint OpenAI-compatible del modelo a evaluar (p.ej. http://host:12021/v1).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Nombre del modelo (opcional; algunos servidores llama.cpp lo ignoran).",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Timeout por peticion al LLM en segundos (default {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("ALIA_API_KEY", ""),
        help="Bearer token para el endpoint LLM (default: $ALIA_API_KEY).",
    )
    parser.add_argument(
        "--judge-uri", default=None,
        help="Endpoint OpenAI-compat del LLM juez (default: mismo que --target-uri).",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Nombre del modelo juez. Si se omite, los scorers usan el default de MLflow (api.openai.com).",
    )
    parser.add_argument(
        "--judge-api-key", default=None,
        help="Bearer token para el endpoint del juez (default: --api-key).",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    judge_uri = args.judge_uri or args.target_uri
    judge_api_key = args.judge_api_key or args.api_key
    run_id = run(
        args.dataset_name,
        args.target_uri,
        args.model,
        args.timeout,
        args.api_key or None,
        judge_uri,
        args.judge_model,
        judge_api_key or None,
    )
    print()
    print(f"run_id: {run_id}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
