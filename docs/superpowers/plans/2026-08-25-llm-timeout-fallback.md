# Configurable OpenAI Timeout and Fallback Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OpenAI-compatible analysis requests use a configurable positive timeout and allow operators to select or explicitly disable the provider fallback model.

**Architecture:** The environment-backed config loader exposes `openai.timeout` and `openai.fallback_model`, preserving an explicitly empty fallback value so it can be distinguished from an absent setting. `LLMService` centralizes timeout validation and fallback-candidate resolution, then uses those helpers for synchronous and streaming requests. Settings metadata and the example environment file expose the two knobs without changing prompts, market data, scheduling, databases, or other providers.

**Tech Stack:** Python 3.12, Flask, requests, pytest, Docker Compose, Alibaba Cloud Workbench CLI

## Global Constraints

- `OPENAI_TIMEOUT` defaults to 120 seconds and invalid or non-positive values fall back to 120.
- `OPENAI_FALLBACK_MODEL` values empty, `none`, `off`, or `disabled` disable fallback attempts.
- When `OPENAI_FALLBACK_MODEL` is absent, preserve the current configured-default then static-fallback candidate behavior.
- Do not change market-data collection, scheduling, database state, prompts, scoring, or other LLM providers.
- Deploy the ECS instance with `OPENAI_TIMEOUT=300` and `OPENAI_FALLBACK_MODEL=disabled`.
- Recreate only backend-image services; do not recreate or delete PostgreSQL or Redis volumes.

---

### Task 1: OpenAI configuration loading and LLM fallback resolution

**Files:**
- Create: `backend_api_python/tests/test_llm_timeout_fallback.py`
- Modify: `backend_api_python/app/utils/config_loader.py:78-103`
- Modify: `backend_api_python/app/services/llm.py:156-321`
- Modify: `backend_api_python/app/services/llm.py:1182-1203`
- Modify: `backend_api_python/app/services/llm.py:1329-1350`

**Interfaces:**
- Consumes: `load_addon_config() -> Dict[str, Any]`, `LLMProvider`, `PROVIDER_CONFIGS`, and `LLMService._normalize_model_for_provider(model, provider)`.
- Produces: `LLMService._get_provider_timeout(provider) -> int` and `LLMService._get_fallback_models(provider) -> List[str]`.

- [ ] **Step 1: Write the failing configuration tests**

Create `backend_api_python/tests/test_llm_timeout_fallback.py` with cache cleanup around environment mutations and assertions for both non-empty and explicitly empty fallback values:

```python
from __future__ import annotations

import pytest
import requests

from app.services.llm import LLMAPIError, LLMProvider, LLMService
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
```

- [ ] **Step 2: Write the failing LLM behavior tests**

Add a helper that stubs provider credentials and captures attempts, then cover timeout validation, disabled fallback, absent fallback compatibility, and explicit fallback selection:

```python
def _stub_openai_service(monkeypatch, config, attempted, *, fail_primary=True):
    service = LLMService(provider="openai")
    monkeypatch.setattr("app.services.llm.load_addon_config", lambda: config)
    monkeypatch.setattr(service, "get_api_key", lambda provider=None: "test-key")
    monkeypatch.setattr(service, "get_base_url", lambda provider=None: "https://example.test/v1")
    monkeypatch.setattr(service, "get_default_model", lambda provider=None: "configured-default")

    def fake_call(messages, model, temperature, api_key, base_url, timeout, use_json_mode=True):
        attempted.append((model, timeout))
        if fail_primary and model == "primary-model":
            raise requests.exceptions.Timeout("primary timed out")
        return "ok"

    monkeypatch.setattr(service, "_call_openai_compatible", fake_call)
    return service


def test_openai_timeout_is_passed_to_request(monkeypatch):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"timeout": 300, "fallback_model": "disabled"}},
        attempted,
        fail_primary=False,
    )

    assert service.call_llm_api([], model="primary-model", try_alternative_providers=False) == "ok"
    assert attempted == [("primary-model", 300)]


@pytest.mark.parametrize("configured", ["", "none", "off", "disabled", " DISABLED "])
def test_explicitly_disabled_fallback_attempts_only_primary(monkeypatch, configured):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"fallback_model": configured}},
        attempted,
    )

    with pytest.raises(requests.exceptions.Timeout, match="primary timed out"):
        service.call_llm_api([], model="primary-model", try_alternative_providers=False)

    assert attempted == [("primary-model", 120)]


def test_absent_fallback_preserves_existing_candidates(monkeypatch):
    attempted = []
    service = _stub_openai_service(monkeypatch, {"openai": {}}, attempted)

    assert service.call_llm_api([], model="primary-model", try_alternative_providers=False) == "ok"
    assert [model for model, _ in attempted] == ["primary-model", "configured-default"]


def test_explicit_fallback_is_used_after_primary(monkeypatch):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"fallback_model": "supported-fallback"}},
        attempted,
    )

    assert service.call_llm_api([], model="primary-model", try_alternative_providers=False) == "ok"
    assert [model for model, _ in attempted] == ["primary-model", "supported-fallback"]


@pytest.mark.parametrize("configured", [0, -1, "invalid"])
def test_invalid_openai_timeout_uses_default(monkeypatch, configured):
    attempted = []
    service = _stub_openai_service(
        monkeypatch,
        {"openai": {"timeout": configured, "fallback_model": "disabled"}},
        attempted,
        fail_primary=False,
    )

    service.call_llm_api([], model="primary-model", try_alternative_providers=False)

    assert attempted == [("primary-model", 120)]
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
cd backend_api_python
pytest -q tests/test_llm_timeout_fallback.py
```

Expected: failures show `OPENAI_TIMEOUT` and `OPENAI_FALLBACK_MODEL` are not loaded, explicit disable still reaches `gpt-4o-mini`, and invalid timeouts are not normalized.

- [ ] **Step 4: Implement environment mappings with explicit-empty support**

In `load_addon_config`, allow selected mappings to retain an empty value and add the two OpenAI mappings:

```python
    def env_get(name: str, *, allow_empty: bool = False) -> Optional[str]:
        val = os.getenv(name)
        if val is None:
            return None
        val = str(val).strip()
        return val if val != "" or allow_empty else None

    ...
        ('OPENAI_MODEL', 'openai.model', 'string'),
        ('OPENAI_TIMEOUT', 'openai.timeout', 'int'),
        ('OPENAI_FALLBACK_MODEL', 'openai.fallback_model', 'string'),
    ...
        raw = env_get(env_name, allow_empty=env_name == 'OPENAI_FALLBACK_MODEL')
```

- [ ] **Step 5: Implement timeout and fallback helpers**

Add constants and focused helper methods in `app/services/llm.py`:

```python
DEFAULT_LLM_TIMEOUT = 120
DISABLED_FALLBACK_VALUES = {"", "none", "off", "disabled"}

    def _get_provider_timeout(self, provider: LLMProvider) -> int:
        configured = load_addon_config().get(provider.value, {}).get(
            "timeout", DEFAULT_LLM_TIMEOUT
        )
        try:
            timeout = int(configured)
            if timeout <= 0:
                raise ValueError
            return timeout
        except (TypeError, ValueError):
            logger.warning(
                "Invalid %s timeout=%r; using %s seconds",
                provider.value,
                configured,
                DEFAULT_LLM_TIMEOUT,
            )
            return DEFAULT_LLM_TIMEOUT

    def _get_fallback_models(self, provider: LLMProvider) -> List[str]:
        provider_config = load_addon_config().get(provider.value, {})
        if "fallback_model" in provider_config:
            configured = str(provider_config.get("fallback_model") or "").strip()
            if configured.lower() in DISABLED_FALLBACK_VALUES:
                return []
            normalized = self._normalize_model_for_provider(configured, provider)
            return [normalized] if normalized else []

        configured_default = self._normalize_model_for_provider(
            self.get_default_model(provider), provider
        )
        static_fallback = self._normalize_model_for_provider(
            PROVIDER_CONFIGS[provider].get("fallback_model") or "", provider
        )
        return list(dict.fromkeys(
            candidate for candidate in (configured_default, static_fallback) if candidate
        ))
```

Use `_get_provider_timeout(p)` in both `call_llm_api` and `stream_llm_api`. Build `models_to_try` from the primary model plus `_get_fallback_models(p)` only when `use_fallback` is true, excluding duplicates.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
cd backend_api_python
pytest -q tests/test_llm_timeout_fallback.py
```

Expected: all tests pass.

- [ ] **Step 7: Run the existing LLM regression module**

Run:

```bash
cd backend_api_python
pytest -q tests/test_llm_litellm_provider.py tests/test_llm_timeout_fallback.py
```

Expected: all tests pass and the existing AtlasCloud and LiteLLM candidate behavior remains unchanged when no explicit provider fallback is configured.

- [ ] **Step 8: Commit the behavior change**

```bash
git add backend_api_python/app/utils/config_loader.py \
  backend_api_python/app/services/llm.py \
  backend_api_python/tests/test_llm_timeout_fallback.py
git commit -m "fix: configure OpenAI timeout and fallback"
```

### Task 2: Settings metadata and operator documentation

**Files:**
- Modify: `backend_api_python/app/routes/settings.py:39-49`
- Modify: `backend_api_python/app/routes/settings.py:360-389`
- Modify: `backend_api_python/env.example:177-185`
- Test: `backend_api_python/tests/test_llm_timeout_fallback.py`

**Interfaces:**
- Consumes: `CONFIG_SCHEMA["ai"]["items"]` metadata rendered by the existing settings frontend.
- Produces: settings entries for `OPENAI_TIMEOUT` and `OPENAI_FALLBACK_MODEL` and documented environment defaults.

- [ ] **Step 1: Write the failing settings-schema test**

Append:

```python
def test_openai_timeout_and_fallback_are_exposed_in_settings():
    from app.routes.settings import ADVANCED_KEYS, CONFIG_SCHEMA

    items = {item["key"]: item for item in CONFIG_SCHEMA["ai"]["items"]}

    assert items["OPENAI_TIMEOUT"]["type"] == "number"
    assert items["OPENAI_TIMEOUT"]["default"] == 120
    assert items["OPENAI_FALLBACK_MODEL"]["type"] == "text"
    assert items["OPENAI_FALLBACK_MODEL"]["default"] == "gpt-4o-mini"
    assert {"OPENAI_TIMEOUT", "OPENAI_FALLBACK_MODEL"} <= ADVANCED_KEYS
```

- [ ] **Step 2: Run the schema test and verify RED**

Run:

```bash
cd backend_api_python
pytest -q tests/test_llm_timeout_fallback.py::test_openai_timeout_and_fallback_are_exposed_in_settings
```

Expected: fail because the keys are absent.

- [ ] **Step 3: Add settings metadata and environment examples**

Add both keys to `ADVANCED_KEYS`. Add OpenAI-group schema entries with a 120-second numeric default and a text fallback default. Document that empty, `none`, `off`, or `disabled` disables fallback. Add these example lines below `OPENAI_MODEL`:

```dotenv
# Request timeout in seconds. Invalid or non-positive values use 120.
OPENAI_TIMEOUT=120
# Set empty, none, off, or disabled to make only the primary-model request.
OPENAI_FALLBACK_MODEL=gpt-4o-mini
```

- [ ] **Step 4: Run settings and LLM tests**

Run:

```bash
cd backend_api_python
pytest -q \
  tests/test_llm_timeout_fallback.py \
  tests/test_settings_env_file.py \
  tests/test_settings_secret_masking.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit operator-facing configuration**

```bash
git add backend_api_python/app/routes/settings.py \
  backend_api_python/env.example \
  backend_api_python/tests/test_llm_timeout_fallback.py
git commit -m "docs: expose OpenAI resilience settings"
```

### Task 3: Local verification and patch artifact preparation

**Files:**
- Create locally for deployment: `/tmp/quantdinger-llm-timeout-fallback/docker-compose.llm-patch.yml`
- Copy locally for deployment: the three modified Python application files

**Interfaces:**
- Consumes: the committed implementation files and the existing `/app` image layout.
- Produces: a Compose override that mounts only patched application files into backend-image services.

- [ ] **Step 1: Run focused regression verification**

Run:

```bash
cd backend_api_python
pytest -q \
  tests/test_llm_timeout_fallback.py \
  tests/test_llm_litellm_provider.py \
  tests/test_settings_env_file.py \
  tests/test_settings_secret_masking.py \
  tests/test_health.py
```

Expected: all tests pass.

- [ ] **Step 2: Run source checks**

Run:

```bash
python -m compileall -q app/utils/config_loader.py app/services/llm.py app/routes/settings.py
git diff --check HEAD~2..HEAD
git status --short
```

Expected: compilation and whitespace checks succeed; only the pre-existing `.playwright-cli/` remains untracked.

- [ ] **Step 3: Create a versioned deployment directory and override**

Create the deployment directory, then create an override whose mounts are repeated for `backend`, `trading-worker`, `scheduler-worker`, `celery-worker`, and `celery-beat`:

```yaml
services:
  backend:
    volumes:
      - ./patches/llm-timeout-fallback/app/utils/config_loader.py:/app/app/utils/config_loader.py:ro
      - ./patches/llm-timeout-fallback/app/services/llm.py:/app/app/services/llm.py:ro
      - ./patches/llm-timeout-fallback/app/routes/settings.py:/app/app/routes/settings.py:ro
  trading-worker:
    volumes:
      - ./patches/llm-timeout-fallback/app/utils/config_loader.py:/app/app/utils/config_loader.py:ro
      - ./patches/llm-timeout-fallback/app/services/llm.py:/app/app/services/llm.py:ro
      - ./patches/llm-timeout-fallback/app/routes/settings.py:/app/app/routes/settings.py:ro
  scheduler-worker:
    volumes:
      - ./patches/llm-timeout-fallback/app/utils/config_loader.py:/app/app/utils/config_loader.py:ro
      - ./patches/llm-timeout-fallback/app/services/llm.py:/app/app/services/llm.py:ro
      - ./patches/llm-timeout-fallback/app/routes/settings.py:/app/app/routes/settings.py:ro
  celery-worker:
    volumes:
      - ./patches/llm-timeout-fallback/app/utils/config_loader.py:/app/app/utils/config_loader.py:ro
      - ./patches/llm-timeout-fallback/app/services/llm.py:/app/app/services/llm.py:ro
      - ./patches/llm-timeout-fallback/app/routes/settings.py:/app/app/routes/settings.py:ro
  celery-beat:
    volumes:
      - ./patches/llm-timeout-fallback/app/utils/config_loader.py:/app/app/utils/config_loader.py:ro
      - ./patches/llm-timeout-fallback/app/services/llm.py:/app/app/services/llm.py:ro
      - ./patches/llm-timeout-fallback/app/routes/settings.py:/app/app/routes/settings.py:ro
```

- [ ] **Step 4: Validate the override locally**

Run:

```bash
docker compose -f docker-compose.yml -f /tmp/quantdinger-llm-timeout-fallback/docker-compose.llm-patch.yml config --quiet
```

Expected: exit 0 with no Compose validation error.

### Task 4: ECS deployment and end-to-end verification

**Files:**
- Create remotely: `/opt/quantdinger/patches/llm-timeout-fallback/app/utils/config_loader.py`
- Create remotely: `/opt/quantdinger/patches/llm-timeout-fallback/app/services/llm.py`
- Create remotely: `/opt/quantdinger/patches/llm-timeout-fallback/app/routes/settings.py`
- Create remotely: `/opt/quantdinger/docker-compose.llm-patch.yml`
- Modify remotely: `/opt/quantdinger/backend_api_python/.env`

**Interfaces:**
- Consumes: ECS `i-t4ngkjkk4vhsrtw5q0ta` in `ap-southeast-1`, Workbench CLI, Docker Compose.
- Produces: an active 300-second OpenAI request timeout with fallback disabled across all long-running backend-image containers.

- [ ] **Step 1: Preflight the exact ECS target and deployment paths**

Run read-only checks with Workbench:

```bash
workbench list ecs --region ap-southeast-1 --output json
workbench exec --instance-id i-t4ngkjkk4vhsrtw5q0ta --command \
  "cd /opt/quantdinger && docker compose ps && test -f backend_api_python/.env && ls -ld patches . 2>/dev/null || true"
```

Expected: the exact instance is Running; PostgreSQL and Redis are healthy; the deployment directory and environment file exist.

- [ ] **Step 2: Check upload targets before overwriting**

Run:

```bash
workbench exec --instance-id i-t4ngkjkk4vhsrtw5q0ta --command \
  "find /opt/quantdinger/patches/llm-timeout-fallback -maxdepth 5 -type f -print 2>/dev/null; test -e /opt/quantdinger/docker-compose.llm-patch.yml && stat /opt/quantdinger/docker-compose.llm-patch.yml || true"
```

Expected: new versioned patch targets do not exist. If they do, choose a new commit-derived directory instead of overwriting them.

- [ ] **Step 3: Upload patch files and Compose override**

Create remote directories first, then use `workbench upload` for each file. No secret file is uploaded or printed.

- [ ] **Step 4: Update only the two environment keys without exposing secrets**

Run a remote shell script that replaces or appends exactly these lines while never printing the environment file:

```dotenv
OPENAI_TIMEOUT=300
OPENAI_FALLBACK_MODEL=disabled
```

Then verify only key names and redacted/policy-safe values:

```bash
workbench exec --instance-id i-t4ngkjkk4vhsrtw5q0ta --command \
  "cd /opt/quantdinger && awk -F= '/^OPENAI_TIMEOUT=/{print \$1\"=\"\$2} /^OPENAI_FALLBACK_MODEL=/{print \$1\"=\"\$2}' backend_api_python/.env"
```

Expected: exactly `OPENAI_TIMEOUT=300` and `OPENAI_FALLBACK_MODEL=disabled`.

- [ ] **Step 5: Validate effective Compose configuration before restart**

Run:

```bash
workbench exec --instance-id i-t4ngkjkk4vhsrtw5q0ta --timeout 60 --command \
  "cd /opt/quantdinger && docker compose -f docker-compose.yml -f docker-compose.llm-patch.yml config --quiet"
```

Expected: exit 0. Do not continue if validation fails.

- [ ] **Step 6: Recreate only long-running backend-image services**

Run the previously approved disruptive operation:

```bash
workbench exec --instance-id i-t4ngkjkk4vhsrtw5q0ta --timeout 180 --command \
  "cd /opt/quantdinger && docker compose -f docker-compose.yml -f docker-compose.llm-patch.yml up -d --no-deps --force-recreate backend trading-worker scheduler-worker celery-worker celery-beat"
```

Expected: up to 30-60 seconds of backend interruption. `postgres`, `redis`, and `redis-jobs` are not recreated.

- [ ] **Step 7: Verify container health and effective runtime settings**

Run `docker compose ps`, the backend health endpoint, and a Python snippet inside the backend container that prints only timeout and fallback values. Confirm the mounted file paths resolve to regular files.

Expected: all five recreated services are running/healthy, public `/api/health` returns success, runtime config reports timeout `300` and fallback `disabled`, and database/cache containers retain their original creation/start times.

- [ ] **Step 8: Trigger one Portfolio Monitor analysis and inspect bounded logs**

Use the existing authenticated `POST /api/portfolio/monitors/<monitor_id>/run` flow for the user's monitor, then poll its result. Inspect only new backend/scheduler logs from the trigger timestamp.

Expected: the report contains a real AI analysis rather than the static `Analysis failed` fallback; logs contain no `gpt-4o-mini` attempt; the primary request may take longer than 120 seconds but completes before 300 seconds.

- [ ] **Step 9: Record rollback command and final state**

Rollback, if verification fails, is:

```bash
cd /opt/quantdinger
docker compose -f docker-compose.yml up -d --no-deps --force-recreate \
  backend trading-worker scheduler-worker celery-worker celery-beat
```

Do not delete patch files or data volumes during rollback. Report the implementation commits, local test counts, recreated services, health result, analysis result, and any remaining GDELT 429 warning separately from LLM success.
