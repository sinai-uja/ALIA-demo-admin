"""Tests para backend.config — configuración centralizada."""

import pytest
from pydantic import ValidationError

from backend.config import Settings, get_settings

MANDATORY = {
    "ALIA_LLM_URL": "http://test-llm:1234/v1",
    "ALIA_LLM_MODEL": "test-model",
    "EMBEDDING_MODEL_PATH": "/tmp/test-embed",
    "TRAMITES_RERANKER_PATH": "/tmp/test-reranker",
}

ALL_SETTINGS_VARS = [
    "ALIA_LLM_URL", "ALIA_LLM_MODEL", "ALIA_LLM_URL_2", "ALIA_LLM_MODEL_2",
    "ALIA_API_KEY",
    "EMBEDDING_MODEL_PATH", "EMBEDDING_DEVICE", "CHROMA_PERSIST_DIR",
    "TRAMITES_RERANKER_PATH", "TRAMITES_CHROMA_COLLECTION",
    "TRAMITES_DATA_PATH", "TRAMITES_TOP_K", "TRAMITES_TOP_K_RERANK",
    "MLFLOW_TRACKING_URI", "MLFLOW_EXTERNAL_URL",
    "FEEDBACK_ENABLED", "LOG_LEVEL",
]

@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Aísla del .env real: cwd limpio + variables de Settings borradas."""
    monkeypatch.chdir(tmp_path)
    for key in ALL_SETTINGS_VARS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

def test_settings_loads_with_full_env(isolated_env, monkeypatch):
    for key, value in MANDATORY.items():
        monkeypatch.setenv(key, value)
    s = Settings()
    assert s.ALIA_LLM_URL == MANDATORY["ALIA_LLM_URL"]
    assert s.ALIA_LLM_MODEL == MANDATORY["ALIA_LLM_MODEL"]
    assert s.EMBEDDING_MODEL_PATH == MANDATORY["EMBEDDING_MODEL_PATH"]
    assert s.TRAMITES_RERANKER_PATH == MANDATORY["TRAMITES_RERANKER_PATH"]

def test_settings_overrides_via_env(isolated_env, monkeypatch):
    for key, value in MANDATORY.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ALIA_LLM_MODEL", "override-model")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TRAMITES_TOP_K", "9")
    monkeypatch.setenv("FEEDBACK_ENABLED", "false")
    s = Settings()
    assert s.ALIA_LLM_MODEL == "override-model"
    assert s.LOG_LEVEL == "DEBUG"
    assert s.TRAMITES_TOP_K == 9
    assert s.FEEDBACK_ENABLED is False

@pytest.mark.parametrize("missing", list(MANDATORY))
def test_settings_fails_without_each_mandatory(isolated_env, monkeypatch, missing):
    for key, value in MANDATORY.items():
        if key != missing:
            monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    assert missing in str(exc_info.value)

def test_get_settings_is_singleton(isolated_env, monkeypatch):
    for key, value in MANDATORY.items():
        monkeypatch.setenv(key, value)
    a = get_settings()
    b = get_settings()
    assert a is b

def test_get_settings_cache_clear_reinstantiates(isolated_env, monkeypatch):
    for key, value in MANDATORY.items():
        monkeypatch.setenv(key, value)
    a = get_settings()
    get_settings.cache_clear()
    b = get_settings()
    assert a is not b

def test_settings_optional_defaults(isolated_env, monkeypatch):
    for key, value in MANDATORY.items():
        monkeypatch.setenv(key, value)
    s = Settings()
    assert s.ALIA_LLM_URL_2 is None
    assert s.ALIA_LLM_MODEL_2 is None
    assert s.ALIA_API_KEY == ""
    assert s.EMBEDDING_DEVICE == "cpu"
    assert s.CHROMA_PERSIST_DIR == "./chroma_db"
    assert s.TRAMITES_CHROMA_COLLECTION == "tramites_municipales"
    assert s.TRAMITES_DATA_PATH == "./data/export.json"
    assert s.TRAMITES_TOP_K == 20
    assert s.TRAMITES_TOP_K_RERANK == 8
    assert s.TRAMITES_MAX_UNIQUE_TRAMITES == 5
    assert s.MLFLOW_TRACKING_URI is None
    assert s.MLFLOW_EXTERNAL_URL is None
    assert s.FEEDBACK_ENABLED is True
    assert s.LOG_LEVEL == "INFO"
