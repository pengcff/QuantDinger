# Configurable OpenAI Timeout and Fallback Model

## Problem

Portfolio Monitor analysis successfully collects market data, then calls the
configured OpenAI-compatible endpoint. The current request timeout is fixed at
120 seconds for the OpenAI provider. When the configured `gpt-5.6-sol` request
times out, the service automatically tries the static `gpt-4o-mini` fallback,
which the configured gateway does not support. The user therefore receives the
default `Analysis failed` result despite valid market data.

## Scope

Add provider-specific configuration for the OpenAI request timeout and fallback
model. Configure the deployed instance to wait 300 seconds and disable its
unsupported fallback. Do not change market-data collection, scheduling,
database state, prompts, scoring, or other LLM providers.

## Design

The configuration loader will map `OPENAI_TIMEOUT` to `openai.timeout` and
`OPENAI_FALLBACK_MODEL` to `openai.fallback_model`. The LLM service will resolve
the fallback in this order:

1. An explicitly configured provider fallback model.
2. No fallback when the configured value is empty, `none`, `off`, or
   `disabled`.
3. The existing static provider fallback when the setting is absent.

This preserves existing behavior for installations that do not use the new
setting. The request timeout remains 120 seconds by default and is bounded to a
positive integer before being passed to the HTTP client.

The settings schema and example environment file will expose both options. The
current server will use:

```dotenv
OPENAI_TIMEOUT=300
OPENAI_FALLBACK_MODEL=disabled
```

## Error Handling

A primary-model timeout remains a normal analysis failure after 300 seconds.
With fallback disabled, the service will preserve the real timeout error instead
of replacing it with an unrelated unsupported-model error. Invalid timeout
values will fall back to the existing 120-second default.

## Testing

Regression tests will prove that:

- `OPENAI_TIMEOUT=300` is loaded and passed to the OpenAI-compatible request.
- an explicitly disabled fallback produces only one model attempt;
- an absent fallback setting retains the existing `gpt-4o-mini` behavior;
- an explicit supported fallback model is attempted after the primary model.

The focused LLM/config tests will be run red before implementation and green
afterward, followed by the related backend test module.

## Deployment

The patched Python files will be copied to a versioned directory under
`/opt/quantdinger/patches` and mounted into the backend, scheduler, Celery, and
other backend-image services through a Compose override. Recreating those
containers may interrupt the application for up to one minute. PostgreSQL and
Redis volumes remain attached and are not recreated or deleted.

After deployment, verification will check container health, effective timeout
and fallback values, public API health, and one manually triggered Portfolio
Monitor analysis. Success requires a non-fallback AI report without a
`gpt-4o-mini` request in the logs.

## Rollback

Rollback consists of starting the stack without the patch override and
recreating only backend-image services. The original GHCR image remains
unchanged, and no database rollback is required.
