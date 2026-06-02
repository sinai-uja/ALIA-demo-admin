# UJAenAgent

> Asistente conversacional sobre el catálogo de trámites municipales de la
> Diputación de Córdoba (**1 091 trámites, 128 municipios**) con modelos de
> lenguaje **locales**, sin enviar datos a APIs cloud.

UJAenAgent es una aplicación experimental de la **Universidad de Jaén** que
demuestra la viabilidad de construir agentes conversacionales con
[LangGraph](https://github.com/langchain-ai/langgraph) y modelos GGUF locales
servidos por [llama.cpp](https://github.com/ggerganov/llama.cpp) sobre datos
de administración pública andaluza.

- 🔒 **Soberanía de datos**: ningún componente requiere APIs externas (LLM,
  embeddings, vector DB y reranker corren en local).
- 🧩 **Modular**: ocho pestañas independientes en el frontend Gradio, cada
  una con su grafo LangGraph y su router FastAPI.
- 📊 **Observabilidad de primera clase**: integración con MLflow para
  tracking, feedback humano y evaluación cuantitativa del RAG.
- ⚙️ **Editable sin redeploy**: los system prompts del LLM se editan desde
  la UI de MLflow gracias al Prompt Registry integrado.

> **Estado: prototipo de investigación.** No es un producto en producción ni
> sustituye a la sede electrónica oficial de ningún ayuntamiento.

---

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Pestañas del frontend](#pestañas-del-frontend)
- [Stack técnico](#stack-técnico)
- [Setup local](#setup-local)
- [Setup Docker](#setup-docker)
- [Indexación del catálogo](#indexación-del-catálogo)
- [Variables de entorno](#variables-de-entorno)
- [Modo evaluación pública (Trámites)](#modo-evaluación-pública-trámites)
- [Edición de prompts sin redeploy](#edición-de-prompts-sin-redeploy)
- [Feedback humano y evaluación cuantitativa](#feedback-humano-y-evaluación-cuantitativa)
- [Tests](#tests)
- [Licencia y créditos](#licencia-y-créditos)

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          GRADIO FRONTEND                                 │
│  ChatBot │ ReAct │ RAG │ Trámites │ Comparador │ Tracking │ Logs │ Param │
└──────┬────────────┬───────────┬──────────────┬─────────────────┬─────────┘
       │ SSE        │ SSE       │ SSE          │ SSE             │ HTTP
       ▼            ▼           ▼              ▼                 │
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐         │
│/chatbot  │ │/react-agent│ │/rag      │ │/tramites   │         │
│ stream   │ │   stream   │ │ stream   │ │   stream   │         │
└────┬─────┘ └─────┬──────┘ └────┬─────┘ └──────┬─────┘         │
     │             │             │               │               │
     │  ┌──────────▼─────────────▼───────────────▼──────┐        │
     │  │              ChromaDB                         │        │
     │  │  knowledge_base │ tramites_municipales        │        │
     │  └──────────────────────────────────────────────┘        │
     │                        │                                  │
     │              ┌─────────▼─────────┐                        │
     │              │  llama.cpp server │                        │
     │              │  :12021/v1        │                        │
     │              └───────────────────┘                        │
     │                                                           │
     └──────────────────────┬────────────────────────────────────┘
                            ▼
                  ┌───────────────────┐
                  │  MLflow Tracking  │
                  │  Server :5001     │
                  └───────────────────┘
```

Patrones clave:

- **Streaming-first**: todas las respuestas se entregan token a token vía
  SSE (`EventSourceResponse`). No hay endpoints síncronos de generación.
- **Encoder único**: el modelo MrBERT genera todos los embeddings de la
  aplicación, garantizando compatibilidad vectorial entre colecciones.
- **API compatible OpenAI**: el backend habla con cualquier runtime LLM que
  exponga `/v1/chat/completions` (llama.cpp, Ollama, vLLM, etc.).
- **Aditivo, no destructivo**: cada pestaña vive en su módulo bajo
  `backend/`; añadir/quitar tabs no afecta a las demás.

---

## Pestañas del frontend

| # | Pestaña | Propósito | Endpoint backend |
|---|---|---|---|
| 1 | **ChatBot** | Chat conversacional simple con grafo LangGraph mínimo (`START → call_llm → END`). | `/chatbot/stream` |
| 2 | **Agente ReAct** | Agente con razonamiento y herramienta de búsqueda en ChromaDB; muestra pasos intermedios. | `/react-agent/stream` |
| 3 | **RAG** | Question-Answering sobre documentos sueltos en `./docs/` (TXT, PDF, MD). | `/rag/stream` |
| 4 | **Trámites** | RAG especializado sobre el catálogo municipal: chunking por secciones del trámite (`## Objeto`, `## Plazo`, etc.), detección de municipio, reranking cross-encoder, agrupación de chunks por trámite tras el rerank. | `/tramites/stream` |
| 5 | **Comparador** | Mismo prompt en paralelo a dos LLMs locales con respuestas side-by-side. | `/comparador/stream` |
| 6 | **Tracking** | Dashboard MLflow embebido (experimentos, runs, traces, feedback). | `/mlflow/*` + proxy `/mlflow-ui` |
| 7 | **Logs** | Últimos N registros de inferencia (query, contexto, respuesta, métricas). | `/logs`, `/logs/count` |
| 8 | **Parámetros** | Configuración runtime (API key, BACKEND_URL) compartida entre tabs. | — |

Adicionalmente, existe un **entrypoint paralelo** (`frontend/eval_app.py`)
con solo la tab Trámites + identificación firmada del evaluador, pensado
para evaluación pública del RAG por personas externas — ver
[Modo evaluación pública](#modo-evaluación-pública-trámites).

---

## Stack técnico

| Componente | Tecnología | Notas |
|---|---|---|
| API | **FastAPI 0.135+** | Async nativo, SSE con `sse-starlette`, OpenAPI autogenerada |
| Frontend | **Gradio 6.9+** | 8 tabs en un único fichero, sin JS personalizado |
| Orquestación de agentes | **LangGraph 1.1+** | Grafos con streaming de eventos, checkpointer por sesión |
| LLM runtime | **llama.cpp server** | Cualquier runtime con API OpenAI-compat sirve (Ollama, vLLM…) |
| Embeddings | **MrBERT** | Modelo multilingüe interno UJA — solicítalo al equipo del proyecto |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Reordenamiento de candidatos RAG |
| Vector DB | **ChromaDB 1.5+** | Persistencia on-disk, dos colecciones (`knowledge_base`, `tramites_municipales`) |
| Tracking | **MLflow 3.x** | Server en Docker, GenAI evaluation, Prompt Registry |
| Config | **pydantic-settings + python-dotenv** | Settings tipadas, fail-fast en arranque si faltan mandatorias |
| Contenedores | **Docker + docker-compose** | 4 servicios + perfiles `dev`/`pre`/`pro` |
| Tests | **pytest** | 166 tests (160 passed + 6 skipped; los skipped requieren MLflow real) |

---

## Setup local

### Requisitos

- **Python 3.11+** (CI y Docker usan 3.11).
- **Un runtime LLM con API compatible OpenAI** corriendo en local, p. ej.
  [llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
  en `:12021` con un modelo GGUF cargado (recomendado: ALIA-40B-Instruct).
- **MrBERT** y **cross-encoder de reranking**, modelos internos UJA.

### Instalación

```bash
# 1. Clonar e instalar dependencias.
git clone <repo-url> ujaenagent
cd ujaenagent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar variables de entorno.
cp .env.example .env
# Edita .env: ALIA_LLM_URL, ALIA_LLM_MODEL, EMBEDDING_MODEL_PATH,
# TRAMITES_RERANKER_PATH son obligatorias (el backend falla al arrancar si
# faltan, con un mensaje claro).

# 3. Indexar el catálogo (una sola vez, ~30s en MPS, 5–15 min en CPU).
python -m backend.rag_tramites.indexer

# 4. Arrancar backend y frontend en dos terminales.
uvicorn backend.main:app --reload --port 8000  # terminal A
python frontend/app.py                          # terminal B
```

Frontend en <http://localhost:7860>. Backend OpenAPI en
<http://localhost:8000/docs>.

---

## Setup Docker

El stack completo (backend + frontend + MLflow) corre con `docker compose`:

```bash
cp .env.example .env
# Edita .env. Para Docker, define también MODELS_HOST_PATH apuntando
# al directorio que contiene MrBERT/ y cross-encoder-ms-marco-MiniLM-L6/.

docker compose up -d
# Frontend principal: http://localhost:7862
# Frontend evaluación: http://localhost:7863
# MLflow UI:           http://localhost:5001/mlflow-ui
# Backend OpenAPI:     http://localhost:8002/docs
```

Existen tres perfiles adicionales en `docker/` para entornos de
desarrollo, pre-producción y producción:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.dev.yml up -d
# (análogos: docker-compose.pre.yml, docker-compose.pro.yml)
```

Detalles en [`docker/README.md`](./docker/README.md).

---

## Indexación del catálogo

El proyecto incluye dos pipelines de indexación independientes:

```bash
# Catálogo de trámites municipales (1 091 trámites, 128 municipios).
# Fuente: data/export.json — versionado en el repo.
python -m backend.rag_tramites.indexer

# Documentos sueltos para la pestaña RAG (./docs/*.txt|.pdf|.md).
python -m backend.tab2_react_agent.ingest
```

### Estrategia de chunking de Trámites

Cada trámite se parte en chunks por los **headers markdown `## `** del
catálogo (`## Objeto`, `## Plazo`, `## Documentación`, etc.). El preámbulo
del trámite (campos `PROCEDIMIENTO`, `Entidad`, `ID`) se preserva como
section `Cabecera`. Si una sección sigue siendo >3 500 chars, se sub-split
con `RecursiveCharacterTextSplitter(chunk_size=3500, overlap=200)`.

Cada chunk se indexa con metadata: `tramite_id`, `section`, `chunk_index`,
`total_chunks` (más los comunes `municipio`, `nombre`, `url`, `auth`). El
`page_content` se prefija con `nombre — section.` para reforzar la
representación semántica del chunk.

Sobre el catálogo real: 1 070 trámites únicos → **~11 900 chunks** (~11
chunks por trámite de media). Permite recuperar información que antes
quedaba invisible al estar en secciones posteriores de trámites largos.

### Recuperación

El retriever de Trámites combina búsqueda vectorial + reranking
cross-encoder + agrupación por trámite:

1. `similarity_search` con `TRAMITES_TOP_K=20` chunks candidatos.
2. Reranking con CrossEncoder MiniLM, top `TRAMITES_TOP_K_RERANK=8`.
3. Agrupación por `tramite_id`, limitada a `TRAMITES_MAX_UNIQUE_TRAMITES=5`
   trámites únicos. Dentro de cada trámite, los chunks se ordenan por
   `chunk_index` para presentación coherente.
4. Render del prompt: cada trámite aparece UNA vez con su info común
   (`nombre`, `URL`, `auth`, `municipio`), seguido de los chunks recuperados
   como sub-bloques `── Section: contenido`.

La pestaña **RAG** usa un retriever más simple: `similarity_search(query,
k=5)` sin reranking.

MrBERT puede emitir warnings esperables al cargar
(`No modules.json found`, `head.dense.weight UNEXPECTED`). Son ignorables
y los marca el propio output de HuggingFace.

---

## Variables de entorno

Definición completa con comentarios en [`.env.example`](./.env.example).
Resumen rápido:

| Bloque | Variables | Obligatoria |
|---|---|---|
| LLM principal | `ALIA_LLM_URL`, `ALIA_LLM_MODEL`, `ALIA_API_KEY` | ✅ (URL + MODEL) |
| LLM secundario (Comparador) | `ALIA_LLM_URL_2`, `ALIA_LLM_MODEL_2` | — |
| Embeddings y reranker | `EMBEDDING_MODEL_PATH`, `TRAMITES_RERANKER_PATH`, `EMBEDDING_DEVICE` (`cpu`/`mps`/`cuda`) | ✅ (paths) |
| Vector store | `CHROMA_PERSIST_DIR` | — |
| Trámites | `TRAMITES_DATA_PATH`, `TRAMITES_CHROMA_COLLECTION`, `TRAMITES_TOP_K`, `TRAMITES_TOP_K_RERANK` | — |
| MLflow | `MLFLOW_TRACKING_URI`, `MLFLOW_EXTERNAL_URL` | — |
| Feedback | `FEEDBACK_ENABLED` (default `true`) | — |
| Prompt Registry | `PROMPT_REGISTRY_ENABLED`, `PROMPT_REGISTRY_DEFAULT_ALIAS`, `PROMPT_REGISTRY_CACHE_TTL_SECONDS` | — |
| Red | `BACKEND_URL`, `GRADIO_SERVER_PORT`, `GRADIO_EVAL_SERVER_PORT` | — |
| Docker | `MODELS_HOST_PATH` | ✅ (solo Docker) |

Las variables obligatorias se validan en el arranque con
[`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/);
si faltan, el backend falla con `ValidationError` y mensaje específico.

---

## Modo evaluación pública (Trámites)

Para que evaluadores externos (técnicos de administraciones públicas, por
ejemplo) puedan probar el RAG de Trámites de forma controlada y con cada
valoración firmada en MLflow, existe un segundo entrypoint Gradio dedicado:

```bash
python frontend/eval_app.py        # local, por defecto en :7861
# o en Docker:
docker compose up -d frontend-eval # expuesto en :7863
```

Características:

- Una única tab (Trámites), sin ChatBot/ReAct/RAG/Comparador/Tracking.
- Bloque "Identifícate como evaluador" (API key + alias) — el alias
  introducido firma cada thumbs up/down enviado a MLflow.
- Disclaimer institucional ("no sustituye la sede electrónica oficial").
- Tema UJA y header/footer institucional.

Pensado para compartir una URL pública (vía ngrok o equivalente) sin
exponer el resto de tabs.

---

## Edición de prompts sin redeploy

Los system y user prompts de las tabs **Trámites**, **Comparador A** y
**RAG docs** están externalizados al **Prompt Registry de MLflow 3.x**.
Personas no técnicas pueden editarlos desde la UI de MLflow sin tocar
código ni redeployar el backend.

### Bootstrap inicial

```bash
docker compose up -d mlflow
python -m backend.prompt_registry.bootstrap
```

Registra v1 de los 4 prompts iniciales con el contenido empaquetado en
`backend/prompt_registry/fallback.py` y crea aliases `staging` y
`production` apuntando a v1. **Idempotente**: si v1 ya existe con el
mismo contenido, no machaca nada.

Vista previa sin efectos secundarios:
`python -m backend.prompt_registry.bootstrap --dry-run`.

### Flujo de edición (operador no técnico)

1. Abrir MLflow UI (<http://localhost:5001>) → pestaña **Prompts**.
2. Seleccionar el prompt (`uja-tramites-system`, `uja-tramites-user-template`,
   `uja-rag-docs-system`, `uja-rag-docs-user-template`).
3. Crear nueva versión con el texto modificado.
4. Mover el alias `production` (o `staging` para probar) a la nueva versión.
5. Esperar a que expire la caché (`PROMPT_REGISTRY_CACHE_TTL_SECONDS`,
   default 300 s) — el siguiente request usará la nueva versión.

### Trazabilidad

Cada traza MLflow emitida por las tabs afectadas lleva tags
`prompt.<name>.version` y `prompt.<name>.source` (`registry` o `fallback`)
con la versión exacta usada. Esto permite filtrar trazas en MLflow UI por
versión de prompt y correlacionar feedback humano con la versión específica
que el usuario evaluó.

### Visibilidad del prompt completo

`mlflow.langchain.autolog(log_traces=True)` está activado al arrancar el
backend (`backend/main.py`). Cada invocación a `ChatOpenAI` (es decir, cada
consulta que llega al LLM en las 5 tabs de chat) queda capturada como span
LLM en la traza con la **lista completa de mensajes** (`SystemMessage` +
`HumanMessage`) en `inputs` y la respuesta en `outputs`. Así puedes auditar
desde la UI de MLflow, byte-a-byte y sin reconstruir nada, exactamente lo
que recibió el modelo: instrucciones del sistema, contexto RAG renderizado
con los trámites recuperados, historial conversacional y pregunta del
usuario, todo concatenado tal cual.

La auto-instrumentación es silenciosa si `MLFLOW_TRACKING_URI` no está
configurado.

### Rollback (tres niveles)

1. **Cambio de versión sin redeploy** (operador en UI): mover el alias
   `production` a la versión anterior. Efectivo tras expirar el TTL.
2. **Desactivar el registry sin redeploy** (devops):
   `PROMPT_REGISTRY_ENABLED=false` y reiniciar el backend → sirve siempre
   el prompt empaquetado.
3. **Si MLflow no responde** (automático): el provider cae al prompt
   empaquetado con `WARNING` en logs. Sin downtime para el usuario final.

---

## Feedback humano y evaluación cuantitativa

### Feedback humano (thumbs up/down)

Bajo cada respuesta del LLM/agente hay thumbs up/down con rationale
opcional. Las valoraciones se envían a `POST /feedback` y se almacenan
como `Assessment` en la traza MLflow correspondiente. Para deshabilitarlo
en frontend + endpoints: `FEEDBACK_ENABLED=false`.

Para anotar ground truth (`expected_response`, `expected_facts`) hay un
panel en la tab **Tracking**.

### Evaluación cuantitativa del RAG de Trámites

Cuatro métricas custom específicas del dominio que los scorers genéricos
de MLflow (`Correctness`, `RelevanceToQuery`) no cubren:

- **Recall de trámites** — fracción de `tramite_id` esperados que aparecen
  en `candidatos_reranked` del grafo (calidad de retrieval).
- **Precisión de URLs** — fracción de URLs de la respuesta final que
  pertenecen al catálogo del municipio esperado (`data/export.json`).
- **Detección de municipio** — el nodo `node_detect_municipio` coincide
  con lo esperado (incluido `None` cuando la query no lleva municipio).
- **Scope mismatch correcto** — el LLM aplica el disclaimer de
  procedimiento no cubierto solo cuando procede.

Bootstrap del dataset (idempotente):

```bash
docker compose up -d mlflow
MLFLOW_TRACKING_URI=http://localhost:5001 \
    python -m scripts.bootstrap_eval_dataset
```

Crea el experimento `rag-tramites-eval` y el dataset
`rag-tramites-eval-baseline-YYYY-MM-DD` con 25 queries curadas:
5 listados completos por municipio, 14 consultas específicas (incluyendo
edge cases con minúsculas y multi-trámite), 3 queries de scope mismatch
y 3 sin municipio explícito. Fuente reproducible en
[`backend/rag_tramites_eval/seed_queries.py`](./backend/rag_tramites_eval/seed_queries.py).

Ejecución de la evaluación:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 \
    python -m scripts.run_tramites_eval \
    --dataset-name rag-tramites-eval-baseline-YYYY-MM-DD
```

El runner invoca el grafo `backend/rag_tramites/graph/tramites_graph.py`
end-to-end (no `/v1/chat/completions` directo) para extraer `response`,
`retrieved_ids` y `extracted_municipio` que alimentan los scorers custom.
Cada ejecución deja un run en `rag-tramites-eval` con tags
`prompt.uja-tramites-system.{version,source}` para trazabilidad de versión
de prompt.

### Datasets ad-hoc desde trazas anotadas

Para evaluar contra cualquier subset de trazas anotadas con
`expected_response`:

```bash
# Construir dataset desde trazas
python -m scripts.build_eval_dataset \
    --experiment chatbot \
    --output-name eval-chatbot-2026-06

# Evaluar con scorers genéricos de MLflow GenAI
python -m scripts.run_evaluation \
    --dataset-name eval-chatbot-2026-06 \
    --target-uri http://localhost:12021/v1 \
    --model alia-40b-instruct
```

---

## Tests

```bash
# Suite completa (160 passed, 6 skipped — los skipped requieren MLflow real).
pytest backend/

# Solo unitarios rápidos (excluye integración con MLflow real).
pytest backend/ -m "not integration"

# Lint.
ruff check backend/ frontend/
```

Los tests marcados con `@pytest.mark.integration` (Prompt Registry,
runner de evaluación) requieren un servidor MLflow operativo
(`MLFLOW_TRACKING_URI` configurado). Se saltan automáticamente si no
está disponible.

---

## Licencia y créditos

UJAenAgent se distribuye bajo la **Apache License 2.0** — ver el fichero
[`LICENSE`](./LICENSE).

Copyright © 2026 Universidad de Jaén.

### Dependencias destacadas

| Proyecto | Licencia |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT |
| [Gradio](https://github.com/gradio-app/gradio) | Apache 2.0 |
| [MLflow](https://github.com/mlflow/mlflow) | Apache 2.0 |
| [ChromaDB](https://github.com/chroma-core/chroma) | Apache 2.0 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | MIT |

### Catálogo de trámites

`data/export.json` contiene el catálogo público de trámites municipales
de los 128 municipios de la provincia de Córdoba, accesible a través del
portal e-admin de [Eprinsa](https://www.eprinsa.es).
