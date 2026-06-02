# Docker & CI/CD — ujaenagent

El proyecto define **cuatro** ficheros de Compose. El de la raíz es el que usa el desarrollo cotidiano (incluye MLflow); los tres de este directorio (`dev`, `pre`, `pro`) cubren el flujo de promoción a través de imágenes publicadas en Docker Hub.

> Las variables de entorno están documentadas en el [`README.md`](../README.md#variables-de-entorno) raíz y en [`.env.example`](../.env.example). Aquí solo se describen los aspectos específicos del despliegue.

## Mapa de Compose files

| Fichero | Uso | Imagen | MLflow | Modelos UJA |
|---|---|---|---|---|
| `../docker-compose.yml` | **Dev local** (recomendado) | build local | ✅ servicio `mlflow:5001` | bind-mount desde host |
| `docker-compose.dev.yml` | Entorno DEV remoto | build local | ❌ no incluido | no montados |
| `docker-compose.pre.yml` | Pre-producción (`staging`) | Docker Hub `pre-latest` | ❌ no incluido | no montados |
| `docker-compose.pro.yml` | Producción (`main`) | Docker Hub `latest` | ❌ no incluido | no montados |

Limitación conocida: los perfiles `dev`/`pre`/`pro` **no levantan MLflow** y **no montan los modelos UJA** (MrBERT, cross-encoder). Solo son aptos para tests donde el LLM remoto y los embeddings se sirven externamente. Para un setup completo en local, usa el `docker-compose.yml` de la raíz.

## Tabla de entornos

| Entorno | Branch | Backend | Frontend | MLflow | Tag imagen |
|---------|--------|---------|----------|--------|------------|
| **Local (root)** | — | `:8002` | `:7862` | `:5001` | build local |
| **DEV** | `develop` | `:8003` | `:7863` | — | `dev-{sha7}`, `dev-latest` |
| **PRE** | `staging` | `:8004` | `:7864` | — | `pre-{sha7}`, `pre-latest` |
| **PRO** | `main` | `:8002` | `:7862` | — | `{sha7}`, `latest`, `{YYYY-MM-DD}` |

Cada entorno tiene su propia red Docker aislada (`ujaenagent-local`, `ujaenagent-dev`, `ujaenagent-pre`, `ujaenagent-pro`) y sus propios volúmenes nombrados para `chroma_db` y `logs` (los de PRE/PRO sobreviven a `docker compose down`; los de local son bind-mounts del workspace).

## Levantar cada entorno

### Dev local (con MLflow)

```bash
docker compose up --build       # desde la raíz del repo
```

Levanta `mlflow` (5001), `backend` (8002), `frontend` (7862). MLflow persiste a `./mlflow/`. **Acceder vía `localhost`**, no `127.0.0.1`: el server tiene `--allowed-hosts` que rechaza este último.

### DEV remoto (build local sin MLflow)

```bash
docker compose -f docker/docker-compose.dev.yml up --build
```

### Pre-producción

```bash
TAG=pre-latest docker compose -f docker/docker-compose.pre.yml up -d
```

### Producción

```bash
TAG=latest docker compose -f docker/docker-compose.pro.yml up -d
```

## Servicio MLflow (solo `docker-compose.yml` raíz)

- Imagen: `ghcr.io/mlflow/mlflow:latest`.
- Backend store: SQLite en `./mlflow/mlflow.db` (bind-mount desde el host).
- Artifact store: `mlflow-artifacts:/` (proxy interno al servidor MLflow).
- UI prefix: `--static-prefix /mlflow-ui`. Accesible en `http://localhost:5001/mlflow-ui`.
- **`--allowed-hosts`**: `localhost`, `127.0.0.1`, `mlflow`, `0.0.0.0`. Si necesitas exponer MLflow vía un dominio adicional (ngrok, túnel SSH), añádelo a la lista en `docker-compose.yml`.

## Proceso de promoción

```
develop → staging → main
  DEV       PRE      PRO
```

1. Desarrollar en `develop` — CI ejecuta lint + verificación de imports + build.
2. Merge PR a `staging` — Se publica imagen `pre-{sha}` y se despliega en PRE.
3. Validar en PRE — Tests de aceptación.
4. Merge PR a `main` — Se publica imagen `latest` y `{fecha}` y se despliega en PRO.

## Notas operativas

- **Modelos UJA**: el servicio `backend` espera MrBERT y el cross-encoder bajo `${MODELS_HOST_PATH}`. Define esa variable en tu `.env` apuntando al directorio que los contiene.
- **Perfiles `dev`/`pre`/`pro` no incluyen MLflow ni modelos** — están pensados para despliegues donde MLflow corre como servicio externo gestionado.

## Troubleshooting

### El backend no arranca
```bash
docker compose logs backend
docker inspect --format='{{json .State.Health}}' <container_id>
```

### Error de permisos en volúmenes
```bash
# Los directorios deben pertenecer al UID 1000 (definido en el Dockerfile)
sudo chown -R 1000:1000 chroma_db logs docs
```

### Limpiar todo
```bash
docker compose down -v --rmi all
docker system prune -f
```

### Verificar que la imagen funciona
```bash
docker run --rm -p 8000:8000 --env-file .env ujaenagent-backend:latest
curl http://localhost:8000/health
```
