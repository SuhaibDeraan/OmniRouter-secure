"""Regression tests for serverRouter.routes.reasoning_routes error handling.

The broad ``except Exception`` in both handlers used to convert every error --
including legitimate 400/401/429 HTTPExceptions -- into a 500 whose body
contained ``traceback.format_exc()``. These tests pin the corrected behavior.

Only firebase config is stubbed; the provider layer and the utils helpers are
monkeypatched per-test. No firebase_admin, no network, no credentials.
"""
import importlib
import json
import sys
import types

import pytest


def _ensure_importable(name, factory):
    try:
        importlib.import_module(name)
    except Exception:
        sys.modules[name] = factory()


def _config_stub():
    m = types.ModuleType("serverRouter.core.config")
    m.VALID_API_KEYS = {"test-key"}
    m.MAX_TOKENS = 100_000
    m.PROVIDERS = {}

    class _ForbiddenDB:
        def __getattr__(self, _n):
            raise AssertionError("Firestore must not be touched in these tests")

    m.db = _ForbiddenDB()
    m.firestore = types.SimpleNamespace(Increment=lambda n: n, SERVER_TIMESTAMP=object())
    return m


_ensure_importable("serverRouter.core.config", _config_stub)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from serverRouter.routes import reasoning_routes  # noqa: E402
from serverRouter.routes.utils import verify_api_key  # noqa: E402
from serverRouter.core.datamodels import ChatReasoningResponse, ReasoningTokenUsage  # noqa: E402
from serverRouter.core.exceptions import ProviderError  # noqa: E402


MODEL = "claude-3-7-sonnet-extended-thinking"


class FakeProvider:
    """A provider that supports both reasoning methods."""

    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def chat_reason_complete(self, request):
        if self._exc:
            raise self._exc
        return self._response

    async def chat_reason_complete_stream(self, request):
        if self._exc:
            raise self._exc
        return self._response


class ProviderMissingMethods:
    """A provider object exposing neither reasoning method."""


@pytest.fixture
def usage_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(reasoning_routes, "add_usage_to_user", lambda uid, n: calls.append((uid, n)))
    return calls


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(reasoning_routes.router)
    # verify_api_key returns the resolved user id
    app.dependency_overrides[verify_api_key] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client, exc):
    def _dep():
        raise exc
    client.app.dependency_overrides[verify_api_key] = _dep


def _set_route(monkeypatch, *, provider=None, raises=None):
    def fake_get_model_and_provider(model_id, models_dict):
        if raises is not None:
            raise raises
        return MODEL, provider
    monkeypatch.setattr(reasoning_routes, "get_model_and_provider", fake_get_model_and_provider)


def _body():
    return {"model": MODEL, "messages": [{"role": "user", "content": "why is the sky blue?"}]}


# --------------------------------------------------------------------------
# HTTPException status codes must pass through unchanged
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/v1/reason/completions", "/v1/reason/completions/stream"])
def test_400_stays_400(client, monkeypatch, usage_calls, path):
    _set_route(monkeypatch, raises=HTTPException(status_code=400, detail="Unknown model: x"))
    resp = client.post(path, json=_body())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Unknown model: x"


@pytest.mark.parametrize("path", ["/v1/reason/completions", "/v1/reason/completions/stream"])
def test_401_from_auth_is_returned_unchanged(client, monkeypatch, usage_calls, path):
    _set_route(monkeypatch, provider=FakeProvider(response=None))
    _override_auth(client, HTTPException(status_code=401, detail="Invalid API key"))
    resp = client.post(path, json=_body())
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ["/v1/reason/completions", "/v1/reason/completions/stream"])
def test_429_from_auth_is_returned_unchanged(client, monkeypatch, usage_calls, path):
    _set_route(monkeypatch, provider=FakeProvider(response=None))
    _override_auth(client, HTTPException(status_code=429, detail="quota"))
    resp = client.post(path, json=_body())
    assert resp.status_code == 429


@pytest.mark.parametrize("path", ["/v1/reason/completions", "/v1/reason/completions/stream"])
def test_provider_without_reasoning_method_is_400(client, monkeypatch, usage_calls, path):
    _set_route(monkeypatch, provider=ProviderMissingMethods())
    resp = client.post(path, json=_body())
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# --------------------------------------------------------------------------
# Unexpected (non-HTTPException) failures -> generic 502, no internals leaked
# --------------------------------------------------------------------------
def test_unexpected_exception_is_502_without_traceback(client, monkeypatch, usage_calls):
    boom = RuntimeError("connection to 10.1.2.3 failed at /srv/internal/secret.py")
    _set_route(monkeypatch, provider=FakeProvider(exc=boom))
    resp = client.post("/v1/reason/completions", json=_body())
    assert resp.status_code == 502
    body = resp.text
    assert "Traceback" not in body
    assert "RuntimeError" not in body
    assert "10.1.2.3" not in body
    assert "secret.py" not in body
    assert resp.json()["detail"] == "Reasoning provider request failed"


def test_unexpected_exception_is_502_on_stream(client, monkeypatch, usage_calls):
    _set_route(monkeypatch, provider=FakeProvider(exc=ValueError("internal boom")))
    resp = client.post("/v1/reason/completions/stream", json=_body())
    assert resp.status_code == 502
    assert "boom" not in resp.text
    assert resp.json()["detail"] == "Reasoning provider streaming request failed"


def test_provider_error_httpexception_passes_through_unchanged(client, monkeypatch, usage_calls):
    _set_route(monkeypatch, provider=FakeProvider(exc=ProviderError("Anthropic API error: 529 overloaded")))
    resp = client.post("/v1/reason/completions", json=_body())
    assert resp.status_code == 500  # ProviderError's own status, unchanged
    assert resp.json()["detail"] == "Anthropic API error: 529 overloaded"


# --------------------------------------------------------------------------
# Successful responses unchanged
# --------------------------------------------------------------------------
def test_successful_non_streaming_response(client, monkeypatch, usage_calls):
    resp_obj = ChatReasoningResponse(
        model=MODEL,
        content="Because of Rayleigh scattering.",
        provider="anthropic",
        usage=ReasoningTokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=30, total_tokens=60),
    )
    _set_route(monkeypatch, provider=FakeProvider(response=resp_obj))
    resp = client.post("/v1/reason/completions", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Because of Rayleigh scattering."
    assert data["usage"]["total_tokens"] == 60
    assert usage_calls == [("user-1", 60)]


def test_successful_streaming_response(client, monkeypatch, usage_calls):
    async def _events():
        yield {"event": "content", "data": json.dumps({"content": "blue"})}
        yield {"event": "usage", "data": json.dumps({"total_tokens": 42})}

    _set_route(monkeypatch, provider=FakeProvider(response=EventSourceResponse(_events())))
    resp = client.post("/v1/reason/completions/stream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: content" in resp.text
    assert "event: usage" in resp.text
    assert usage_calls == [("user-1", 42)]


def test_stream_malformed_usage_chunk_does_not_crash(client, monkeypatch, usage_calls):
    async def _events():
        yield {"event": "usage"}  # no data key -> previously json.loads({}) TypeError
        yield {"event": "usage", "data": "{not json"}
        yield {"event": "content", "data": json.dumps({"content": "still here"})}

    _set_route(monkeypatch, provider=FakeProvider(response=EventSourceResponse(_events())))
    resp = client.post("/v1/reason/completions/stream", json=_body())
    assert resp.status_code == 200
    assert "still here" in resp.text
    assert usage_calls == []


def test_stream_cumulative_usage_is_not_double_counted(client, monkeypatch, usage_calls):
    async def _events():
        yield {"event": "usage", "data": json.dumps({"total_tokens": 100})}
        yield {"event": "content", "data": json.dumps({"content": "x"})}
        yield {"event": "usage", "data": json.dumps({"total_tokens": 150})}

    _set_route(monkeypatch, provider=FakeProvider(response=EventSourceResponse(_events())))
    resp = client.post("/v1/reason/completions/stream", json=_body())
    assert resp.status_code == 200
    assert sum(n for _, n in usage_calls) == 150
