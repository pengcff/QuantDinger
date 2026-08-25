from __future__ import annotations

from collections.abc import Collection

import pytest
import requests

from app.services.llm import LLMProvider, LLMService
from app.utils.config_loader import clear_config_cache, load_addon_config


@pytest.fixture(autouse=True)
def reset_config_cache():
    clear_config_cache()
    yield
    clear_config_cache()


def test_openai_timeout_and_fallback_are_loaded(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT", "300")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "gpt-4.1-mini")

    config = load_addon_config()

    assert config["openai"]["timeout"] == 300
    assert config["openai"]["fallback_model"] == "gpt-4.1-mini"


def test_empty_openai_fallback_is_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "")

    config = load_addon_config()

    assert "fallback_model" in config["openai"]
    assert config["openai"]["fallback_model"] == ""


def _stub_openai_service(
    monkeypatch,
    config,
    attempted,
    *,
    failing_models: Collection[str] = (),
):
    service = LLMService(provider="openai")
    monkeypatch.setattr("app.services.llm.load_addon_config", lambda: config)
    monkeypatch.setattr(service, "get_api_key", lambda provider=None: "test-key")
    monkeypatch.setattr(
        service,
        "get_base_url",
        lambda provider=None: "https://example.test/v1",
    )
    monkeypatch.setattr(
        service,
        "get_default_model",
        lambda provider=None: "configured-default",
    )

    def fake_call(
        messages,
        model,
        temperature,
        api_key,
        base_url,
        timeout,
        use_json_mode=True,
    ):
        attempted.append((model, timeout))
        if model in failing_models:
            raise requests.exceptions.Timeout(f"{model} timed out")
        return "ok"

    monkeypatch.setattr(service, "_call_openai_compatible", fake_call)
    return service


def test_openai_timeout_is_passed_to_request(monkeypatch):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"timeout": 300, "fallback_model": "disabled"}},
        attempted,
    )

    assert service.call_llm_api(
        [],
        model="primary-model",
        try_alternative_providers=False,
    ) == "ok"
    assert attempted == [("primary-model", 300)]


@pytest.mark.parametrize("configured", ["", "none", "off", "disabled", " DISABLED "])
def test_explicitly_disabled_fallback_attempts_only_primary(monkeypatch, configured):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"fallback_model": configured}},
        attempted,
        failing_models={"primary-model"},
    )

    with pytest.raises(requests.exceptions.Timeout, match="primary-model timed out"):
        service.call_llm_api(
            [],
            model="primary-model",
            try_alternative_providers=False,
        )

    assert attempted == [("primary-model", 120)]


def test_absent_fallback_preserves_existing_candidates(monkeypatch):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {}},
        attempted,
        failing_models={"primary-model", "configured-default"},
    )

    assert service.call_llm_api(
        [],
        model="primary-model",
        try_alternative_providers=False,
    ) == "ok"
    assert [model for model, _ in attempted] == [
        "primary-model",
        "configured-default",
        "gpt-4o-mini",
    ]


def test_explicit_fallback_is_used_after_primary(monkeypatch):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"fallback_model": "supported-fallback"}},
        attempted,
        failing_models={"primary-model"},
    )

    assert service.call_llm_api(
        [],
        model="primary-model",
        try_alternative_providers=False,
    ) == "ok"
    assert [model for model, _ in attempted] == [
        "primary-model",
        "supported-fallback",
    ]


@pytest.mark.parametrize("configured", [0, -1, "invalid"])
def test_invalid_openai_timeout_uses_default(monkeypatch, configured):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"timeout": configured, "fallback_model": "disabled"}},
        attempted,
    )

    service.call_llm_api(
        [],
        model="primary-model",
        try_alternative_providers=False,
    )

    assert attempted == [("primary-model", 120)]


def test_openai_timeout_and_fallback_are_exposed_in_settings():
    from app.routes.settings import ADVANCED_KEYS, CONFIG_SCHEMA

    items = {item["key"]: item for item in CONFIG_SCHEMA["ai"]["items"]}

    assert items["OPENAI_TIMEOUT"]["type"] == "number"
    assert items["OPENAI_TIMEOUT"]["default"] == 120
    assert items["OPENAI_FALLBACK_MODEL"]["type"] == "text"
    assert items["OPENAI_FALLBACK_MODEL"]["default"] == "gpt-4o-mini"
    assert {"OPENAI_TIMEOUT", "OPENAI_FALLBACK_MODEL"} <= ADVANCED_KEYS
