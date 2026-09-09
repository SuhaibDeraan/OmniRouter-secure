"""Regression tests for serverRouter.routes.completion_routes.

Covers error-handling correctness and streaming robustness:
  * HTTPException / ProviderError pass through unchanged
  * unexpected provider failure -> generic 502, no internals leaked
  * successful (streaming + non-streaming) responses unchanged
  * malformed / invalid-JSON / field-less stream chunks never crash the stream
  * usage accounting: recorded once, cumulative, never double-counted

Only firebase config is stubbed; provider + utils helpers are monkeypatched.
"""
import importlib
import json
import sys
import types

import pytest

from sse_starlette.sse import EventSourceResponse


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

from serverRouter.routes import completion_routes  # noqa: E402
from serverRouter.routes.utils import verify_api_key  # noqa: E402
from serverRouter.core.datamodels import ChatCompletionResponse, ImageGenerationResponse  # noqa: E402
from serverRouter.core.exceptions import ProviderError  # noqa: E402


MODEL = "gpt-4o"


class FakeProvider:
    def __init__(self, *, response=None, exc=None, stream_chunks=None, stream_exc=None):
        self._response = response
        self._exc = exc
        self._stream_chunks = stream_chunks or []
        self._stream_exc = stream_exc

    async def chat_complete(self, request):
        if self._exc:
            raise self._exc
        return self._response

    async def chat_complete_stream(self, request):
        if self._stream_exc:
            raise self._stream_exc

        chunks = self._stream_chunks

        async def gen():
            for c in chunks:
                yield c

        return EventSourceResponse(gen())


@pytest.fixture
def usage_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(completion_routes, "add_usage_to_user", lambda uid, n: calls.append((uid, n)))
    return calls


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(completion_routes.router)
    # verify_api_key returns the resolved user id
    app.dependency_overrides[verify_api_key] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


def _set_provider(monkeypatch, *, provider=None, raises=None):
    def fake_gmap(model_id, models_dict):
        if raises is not None:
            raise raises
        return MODEL, provider
    monkeypatch.setattr(completion_routes, "get_model_and_provider", fake_gmap)


def _raise(exc):
    def _dep():
        raise exc
    return _dep


def _body():
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


def _completion(content="hello", total_tokens=42):
    return ChatCompletionResponse(
        model=MODEL, content=content, provider="openai",
        usage={"prompt_tokens": 2, "completion_tokens": total_tokens - 2, "total_tokens": total_tokens},
    )


# --------------------------------------------------------------------------
# Non-streaming
# --------------------------------------------------------------------------
def test_successful_non_streaming_completion(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeProvider(response=_completion("hello world", 40)))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "model": MODEL,
        "content": "hello world",
        "provider": "openai",
        "usage": {"prompt_tokens": 2, "completion_tokens": 38, "total_tokens": 40},
    }
    assert usage_calls == [("user-1", 40)]


def test_non_streaming_missing_total_tokens_does_not_crash(client, monkeypatch, usage_calls):
    resp_obj = ChatCompletionResponse(model=MODEL, content="x", provider="openai", usage={"prompt_tokens": 1})
    _set_provider(monkeypatch, provider=FakeProvider(response=resp_obj))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    assert usage_calls == [("user-1", 0)]


def test_http_exception_passes_through(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, raises=HTTPException(status_code=400, detail="Unknown model: zzz"))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Unknown model: zzz"


def test_provider_error_maps_to_its_status(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeProvider(exc=ProviderError("OpenAI API error: 503")))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 500
    assert resp.json()["detail"] == "OpenAI API error: 503"


def test_unexpected_exception_is_502_without_internals(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeProvider(exc=RuntimeError("boom at /srv/app/secret.py line 9")))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 502
    body = resp.text
    for leak in ("Traceback", "RuntimeError", "secret.py", "boom"):
        assert leak not in body
    assert resp.json()["detail"] == "Chat completion provider request failed"
    assert usage_calls == []  # nothing recorded on failure


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
def _run_stream(client, chunks):
    provider = FakeProvider(stream_chunks=chunks)
    return provider


def test_successful_streaming(client, monkeypatch, usage_calls):
    chunks = [
        {"event": "content", "data": json.dumps({"content": "he"})},
        {"event": "content", "data": json.dumps({"content": "llo"})},
        {"event": "usage", "data": json.dumps({"total_tokens": 25})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.count("event: content") == 2
    assert "event: usage" in resp.text
    assert usage_calls == [("user-1", 25)]


def test_stream_start_failure_is_502(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeProvider(stream_exc=RuntimeError("internal /x")))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 502
    assert "internal" not in resp.text
    assert usage_calls == []


def test_stream_http_exception_passes_through(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, raises=HTTPException(status_code=401, detail="Invalid API key"))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 401


def test_malformed_non_dict_chunk_does_not_break_stream(client, monkeypatch, usage_calls):
    chunks = [
        "a bare string chunk",
        {"event": "content", "data": json.dumps({"content": "still here"})},
        {"event": "usage", "data": json.dumps({"total_tokens": 10})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert "still here" in resp.text  # valid chunk after the bad one is preserved
    assert usage_calls == [("user-1", 10)]


def test_invalid_json_usage_chunk_does_not_crash_or_count(client, monkeypatch, usage_calls):
    chunks = [
        {"event": "usage", "data": "{not valid json"},
        {"event": "content", "data": json.dumps({"content": "ok"})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert "ok" in resp.text
    assert usage_calls == []  # unparseable usage -> not recorded, no crash


def test_missing_event_and_data_fields_are_ignored(client, monkeypatch, usage_calls):
    chunks = [
        {"data": json.dumps({"content": "no event key"})},
        {"event": "usage"},  # no data key
        {"event": "usage", "data": json.dumps({"no_total": 1})},  # no total_tokens
        {"event": "content", "data": json.dumps({"content": "end"})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert "end" in resp.text
    assert usage_calls == []


def test_multiple_cumulative_usage_chunks_are_not_double_counted(client, monkeypatch, usage_calls):
    chunks = [
        {"event": "usage", "data": json.dumps({"total_tokens": 100})},
        {"event": "content", "data": json.dumps({"content": "more"})},
        {"event": "usage", "data": json.dumps({"total_tokens": 150})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert sum(n for _, n in usage_calls) == 150
    assert [uid for uid, _ in usage_calls] == ["user-1", "user-1"]


def test_repeated_identical_usage_chunk_counts_once(client, monkeypatch, usage_calls):
    chunks = [
        {"event": "usage", "data": json.dumps({"total_tokens": 100})},
        {"event": "usage", "data": json.dumps({"total_tokens": 100})},
    ]
    _set_provider(monkeypatch, provider=FakeProvider(stream_chunks=chunks))
    resp = client.post("/v1/chat/completions/stream", json=_body())
    assert resp.status_code == 200
    assert usage_calls == [("user-1", 100)]


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------
IMAGE_MODEL = "dall-e-3"


class FakeImageProvider:
    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def generate_image(self, request):
        if self._exc:
            raise self._exc
        return self._response


def _image_body():
    return {"prompt": "a red bicycle", "model": IMAGE_MODEL, "n": 1}


def _image_response(urls=("data:image/png;base64,AAA",)):
    return ImageGenerationResponse(urls=list(urls), model=IMAGE_MODEL, provider="openai")


def test_successful_image_generation(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeImageProvider(response=_image_response()))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 200
    assert resp.json() == {
        "urls": ["data:image/png;base64,AAA"],
        "model": IMAGE_MODEL,
        "provider": "openai",
    }
    # recorded once, at 0 token cost (no image price defined in the repo)
    assert usage_calls == [("user-1", 0)]


def test_image_http_exception_passes_through(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, raises=HTTPException(status_code=400, detail="Unknown model: foo"))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Unknown model: foo"
    assert usage_calls == []


def test_image_quota_gate_applies(client, monkeypatch, usage_calls):
    # /images/generate is guarded by the same verify_api_key dependency as chat
    client.app.dependency_overrides[verify_api_key] = _raise(HTTPException(status_code=429, detail="quota"))
    _set_provider(monkeypatch, provider=FakeImageProvider(response=_image_response()))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 429
    assert usage_calls == []


def test_image_provider_error_maps_to_its_status(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeImageProvider(exc=ProviderError("Stable Diffusion API error: NSFW")))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Stable Diffusion API error: NSFW"
    assert usage_calls == []  # failed generation is not billed


def test_image_unexpected_exception_is_502_without_internals(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeImageProvider(exc=RuntimeError("boom /srv/app/keys.py:3")))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 502
    body = resp.text
    for leak in ("Traceback", "RuntimeError", "keys.py", "boom"):
        assert leak not in body
    assert resp.json()["detail"] == "Image generation provider request failed"
    assert usage_calls == []


def test_image_repeated_calls_do_not_double_count(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeImageProvider(response=_image_response()))
    client.post("/v1/images/generate", json=_image_body())
    client.post("/v1/images/generate", json=_image_body())
    assert usage_calls == [("user-1", 0), ("user-1", 0)]  # one per call, not two


def test_image_response_without_usage_attr_records_zero(client, monkeypatch, usage_calls):
    _set_provider(monkeypatch, provider=FakeImageProvider(response=_image_response()))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 200
    assert usage_calls == [("user-1", 0)]


@pytest.mark.parametrize("bad_usage", [None, "oops", {"total_tokens": "nan"}, {"prompt": 1}, 123])
def test_image_malformed_or_missing_usage_metadata_records_zero(client, monkeypatch, usage_calls, bad_usage):
    obj = types.SimpleNamespace(
        urls=["data:image/png;base64,AAA"], model=IMAGE_MODEL, provider="openai", usage=bad_usage
    )
    _set_provider(monkeypatch, provider=FakeImageProvider(response=obj))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 200
    assert usage_calls == [("user-1", 0)]


def test_image_provider_reporting_token_usage_is_billed_as_reported(client, monkeypatch, usage_calls):
    # forward-compat: if a provider ever returns usage, bill exactly that value
    obj = types.SimpleNamespace(
        urls=["data:image/png;base64,AAA"], model=IMAGE_MODEL, provider="together",
        usage={"total_tokens": 50},
    )
    _set_provider(monkeypatch, provider=FakeImageProvider(response=obj))
    resp = client.post("/v1/images/generate", json=_image_body())
    assert resp.status_code == 200
    assert usage_calls == [("user-1", 50)]
