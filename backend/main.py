"""
Backend principal de la aplicación LangGraph Experimental.

Registra los routers de cada pestaña y configura CORS, autenticación y middleware.
"""

import logging
import time
import hmac

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.tab1_chatbot.router import router as chatbot_router
from backend.tab2_react_agent.router import router as react_agent_router
from backend.tab2bis_rag.router import router as rag_router
from backend.rag_tramites.api.router import router as tramites_router
from backend.inference_logger.router import router as logs_router
from backend.comparador.router import router as comparador_router
from backend.mlflow_dashboard.router import router as mlflow_router
from backend.feedback.router import router as feedback_router

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def _enable_mlflow_langchain_autolog() -> None:
    """Activa la auto-instrumentación de LangChain en MLflow.

    Con esto, cada `ChatOpenAI.ainvoke([...messages])` queda capturada como
    span LLM con la lista completa de mensajes (SystemMessage + HumanMessage)
    en `inputs` y la respuesta en `outputs`. Esto permite auditar
    byte-a-byte el prompt final que recibe el modelo desde la UI de MLflow,
    sin tener que reconstruirlo desde el Prompt Registry + el artefacto JSON.

    Silencioso si `MLFLOW_TRACKING_URI` no está configurado.
    """
    if not settings.MLFLOW_TRACKING_URI:
        return
    try:
        import mlflow
        import mlflow.langchain

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.langchain.autolog(log_traces=True)
        logger.info("mlflow.langchain.autolog activado (log_traces=True)")
    except Exception as exc:
        logger.warning("No se pudo activar mlflow.langchain.autolog: %s", exc)


_enable_mlflow_langchain_autolog()

_ALIA_API_KEY = settings.ALIA_API_KEY

app = FastAPI(title="LangGraph Experimental Backend")

# CORS para Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7861"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas que no requieren autenticación
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def auth_and_log(request: Request, call_next):
    """Middleware de autenticación por Bearer token y logging."""
    start_time = time.time()

    # Rutas públicas no requieren autenticación
    if request.url.path not in _PUBLIC_PATHS:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "API key requerida"})

        provided_key = auth_header[7:]  # Quitar "Bearer "
        if not hmac.compare_digest(provided_key, _ALIA_API_KEY):
            return JSONResponse(status_code=403, content={"detail": "API key inválida"})

    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    return response

# Registro de routers
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(react_agent_router, prefix="/react-agent")
app.include_router(rag_router, prefix="/rag")
app.include_router(tramites_router, prefix="/tramites")
app.include_router(logs_router, prefix="/logs")
app.include_router(comparador_router, prefix="/comparador")
app.include_router(mlflow_router)
app.include_router(feedback_router, prefix="/feedback")

@app.get("/health")
async def health():
    """Endpoint de salud del servicio (no requiere autenticación)."""
    return {"status": "ok"}

@app.get("/auth/check")
async def auth_check():
    """Endpoint ligero para validar el token (requiere autenticación)."""
    return {"status": "authorized"}
