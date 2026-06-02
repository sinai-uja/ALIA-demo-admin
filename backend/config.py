"""
Configuración centralizada del backend.

Una sola clase `Settings` que consolida todas las variables de entorno que el
backend necesita. Sustituye los `os.getenv()` y `load_dotenv()` dispersos por
un singleton tipado.

Modelos y endpoints LLM son obligatorios: la app falla al arrancar con un
`pydantic.ValidationError` claro si faltan en `.env` o en el entorno.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuración del backend de UJAenAgent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mandatory: sin default. El arranque falla si falta cualquiera.
    ALIA_LLM_URL: str
    ALIA_LLM_MODEL: str
    EMBEDDING_MODEL_PATH: str
    TRAMITES_RERANKER_PATH: str

    # LLM secundario (Comparador). El router del comparador valida en runtime
    # y devuelve 503 si están a None.
    ALIA_LLM_URL_2: str | None = None
    ALIA_LLM_MODEL_2: str | None = None

    # API key compartida del backend (modo dev acepta vacío).
    ALIA_API_KEY: str = ""

    # Embeddings y vector store.
    EMBEDDING_DEVICE: str = "cpu"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # RAG Trámites.
    TRAMITES_CHROMA_COLLECTION: str = "tramites_municipales"
    TRAMITES_DATA_PATH: str = "./data/export.json"
    # Cada trámite se indexa como ~10-15 chunks por sección markdown.
    # top_k=20 amplía el pool de candidatos, rerank=8 deja margen para que
    # el cross-encoder seleccione combinaciones útiles, max_unique=5 acota
    # el contexto del LLM a un máximo de 5 trámites distintos por consulta.
    TRAMITES_TOP_K: int = 20
    TRAMITES_TOP_K_RERANK: int = 8
    TRAMITES_MAX_UNIQUE_TRAMITES: int = 5

    # MLflow. None desactiva el tracking; los módulos comprueban antes de usar.
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXTERNAL_URL: str | None = None

    # Feedback humano.
    FEEDBACK_ENABLED: bool = True

    # Prompt Registry de MLflow. Externaliza los system/user
    # prompts de los nodos de chat. Si está en False, el provider sirve
    # siempre desde el fallback empaquetado en
    # backend/prompt_registry/fallback.py (rollback sin redeploy).
    PROMPT_REGISTRY_ENABLED: bool = True
    # Alias de entorno consumido por defecto al cargar `prompts:/<name>@<alias>`.
    # Editado por el operador desde la UI de MLflow para promover versiones.
    PROMPT_REGISTRY_ALIAS: str = "production"
    # TTL en segundos de la caché in-memory del PromptProvider. Tras este
    # tiempo, la siguiente request a cada (name, alias) refresca contra MLflow.
    PROMPT_REGISTRY_CACHE_TTL_SECONDS: int = 300

    # Logging. Aplicado en backend/main.py vía logging.basicConfig.
    LOG_LEVEL: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    """Devuelve el singleton de configuración."""
    return Settings()

settings = get_settings()
