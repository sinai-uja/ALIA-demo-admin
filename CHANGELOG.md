# Changelog

Todas las versiones notables de este proyecto se documentan en este fichero.

Sigue el formato de [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-01

### Released

Primer release público de UJAenAgent, agente conversacional sobre el catálogo
de trámites municipales de la Diputación de Córdoba (1 091 trámites de
128 municipios) con modelos de lenguaje locales (ALIA-40B vía llama.cpp).

#### Capacidades

- **8 pestañas funcionales end-to-end**: ChatBot simple, Agente ReAct,
  RAG sobre documentos, RAG especializado en Trámites, Comparador dual
  de LLMs, Tracking MLflow, Logs de inferencia y Parámetros runtime.
- **URL de evaluación pública** dedicada (`frontend/eval_app.py`) con
  identificación firmada del evaluador en cada feedback registrado en MLflow.
- **MLflow Prompt Registry** integrado: edición de prompts del LLM por
  personas no técnicas vía UI de MLflow, sin redeploy ni tocar código.
- **Feedback humano** (thumbs up/down) capturado por traza con
  `mlflow.log_feedback` y panel para anotar ground truth
  (`expected_response` / `expected_facts`).
- **Evaluación cuantitativa del RAG de Trámites** con 4 scorers custom
  (recall de trámites, precisión de URLs, detección de municipio,
  scope mismatch correcto) ejecutables vía `mlflow.genai.evaluate`.
- **Stack Docker** con servicios MLflow, backend, frontend y
  frontend-eval, y perfiles separados `dev`/`pre`/`pro`.

#### Stack técnico

- **Backend**: FastAPI 0.135+ con SSE (Server-Sent Events) para streaming
  de tokens del LLM.
- **Frontend**: Gradio 6.9+ (8 tabs en un fichero, sin JS personalizado).
- **Orquestación de agentes**: LangGraph 1.1+.
- **LLM runtime**: llama.cpp server con API compatible OpenAI (ALIA-40B
  como modelo principal, soporta segundo modelo para Comparador).
- **Embeddings**: MrBERT (modelo interno UJA) vía HuggingFace.
- **Vector DB**: ChromaDB on-disk, una colección para trámites y otra
  para documentos.
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2.
- **Observabilidad**: MLflow Tracking Server con UI integrada en el frontend.

#### Tests

- 166 tests unitarios + integración (skip-on-missing para los que requieren
  MLflow real).
- CI con lint (ruff) y suite completa.
