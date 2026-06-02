"""
Tests unitarios para los helpers de expectations.

Verifica los argumentos pasados a `mlflow.log_expectation` con MLflow mockeado.
"""

from unittest.mock import patch

import pytest

from backend.config import settings
from backend.feedback.expectations import (
    EXPECTED_FACTS,
    EXPECTED_RESPONSE,
    ExpectationSource,
    log_expected_facts,
    log_expected_response,
)

@pytest.fixture(autouse=True)
def _mlflow_uri(monkeypatch):
    """Garantiza que MLFLOW_TRACKING_URI esta seteado para los helpers."""
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "http://mlflow-mock:5000")

def test_log_expected_response_uses_standard_name_and_human_source():
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        log_expected_response(
            trace_id="tr-abc",
            response="42",
            source_id="anotador_jane",
        )

    assert log_mock.call_count == 1
    kwargs = log_mock.call_args.kwargs
    assert kwargs["trace_id"] == "tr-abc"
    assert kwargs["name"] == EXPECTED_RESPONSE
    assert kwargs["value"] == "42"
    assert kwargs["metadata"] is None
    source = kwargs["source"]
    assert source.source_type == "HUMAN"
    assert source.source_id == "anotador_jane"

def test_log_expected_facts_value_is_list_with_standard_name():
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        log_expected_facts(
            trace_id="tr-xyz",
            facts=["fact 1", "fact 2"],
            source_id="anotador_jane",
        )

    kwargs = log_mock.call_args.kwargs
    assert kwargs["name"] == EXPECTED_FACTS
    assert kwargs["value"] == ["fact 1", "fact 2"]
    assert isinstance(kwargs["value"], list)

def test_log_expected_response_supports_llm_judge_source():
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        log_expected_response(
            trace_id="tr-abc",
            response="42",
            source_id="gpt4-judge",
            source_type=ExpectationSource.LLM_JUDGE,
        )

    source = log_mock.call_args.kwargs["source"]
    assert source.source_type == "LLM_JUDGE"
    assert source.source_id == "gpt4-judge"

def test_log_expected_response_passes_metadata_through():
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        log_expected_response(
            trace_id="tr-abc",
            response="42",
            source_id="anotador",
            metadata={"campo": "valor"},
        )

    assert log_mock.call_args.kwargs["metadata"] == {"campo": "valor"}

def test_helpers_fail_if_tracking_uri_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        log_expected_response(trace_id="tr-x", response="r", source_id="s")
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        log_expected_facts(trace_id="tr-x", facts=["f"], source_id="s")

def test_facts_input_copied_into_new_list():
    """Garantiza que mutar la lista original no afecta lo registrado."""
    facts = ["a", "b"]
    with patch("mlflow.log_expectation") as log_mock, patch("mlflow.set_tracking_uri"):
        log_expected_facts(trace_id="tr-1", facts=facts, source_id="s")
    captured = log_mock.call_args.kwargs["value"]
    facts.append("c")
    assert captured == ["a", "b"]
