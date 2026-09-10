"""Regression tests for serverRouter.providers.gemini.provider.GeminiProvider.

Proves the blocking genai calls run off the asyncio event loop, while response
shapes / streaming semantics / error handling are unchanged.

`google.generativeai` and `dotenv` are stubbed; no network, no credentials.
"""
import asyncio
import importlib
import sys
import threading
import types

import pytest


# --------------------------------------------------------------------------
# fake google.generativeai
# --------------------------------------------------------------------------
class _Usage:
    def __init__(self, p, c):
        self.prompt_token_count = p
        self.candidates_token_count = c


class _Response:
    def __init__(self, text, prompt=3, candidates=5):
        self._text = text
        self.usage_metadata = _Usage(prompt, candidates)

    @property
    def text(self):
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _Chunk:
    def __init__(self, text, prompt=3, candidates=5, with_usage=True):
        self.text = text
        self.usage_metadata = _Usage(prompt, candidates) if with_usage else None


class _ThreadRecordingIterator:
    def __init__(self, chunks, sink):
        self._chunks = list(chunks)
        self._i = 0
        self._sink = sink

    def __iter__(self):
        return self

    def __next__(self):
        self._sink.append(("next", threading.get_ident()))
        if self._i >= len(self._chunks):
            raise StopIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


class _FakeModel:
    def __init__(self, sink, *, response=None, exc=None, stream_chunks=None):
        self._sink = sink
        self._response = response
        self._exc = exc
        self._stream_chunks = stream_chunks

    def generate_content(self, contents=None, generation_config=None, stream=False):
        self._sink.append(("generate_content", threading.get_ident()))
        if self._exc is not None:
            raise self._exc
        if stream:
            return _ThreadRecordingIterator(self._stream_chunks or [], self._sink)
        return self._response


def _make_genai(model):
    m = types.ModuleType("google.generativeai")
    m.configure = lambda **kw: None
    m.GenerativeModel = lambda model_name=None: model
    m.types = types.SimpleNamespace(GenerationConfig=lambda **kw: kw)
    return m


@pytest.fixture(autouse=True)
def _stub_modules():
    saved = {k: sys.modules.get(k) for k in
             ("google", "google.generativeai", "dotenv",
              "serverRouter.providers.gemini.provider")}
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv_stub
    google_stub = sys.modules.get("google") or types.ModuleType("google")
    sys.modules["google"] = google_stub
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _load_provider(model):
    genai_stub = _make_genai(model)
    sys.modules["google.generativeai"] = genai_stub
    sys.modules["google"].generativeai = genai_stub
    sys.modules.pop("serverRouter.providers.gemini.provider", None)
    mod = importlib.import_module("serverRouter.providers.gemini.provider")
    return mod.GeminiProvider(api_key="fake"), mod


def _req(**over):
    from serverRouter.core.datamodels import ChatCompletionRequest
    base = dict(model="gemini-2.0-pro", messages=[{"role": "user", "content": "hi"}],
                temperature=1.0, max_tokens=64)
    base.update(over)
    return ChatCompletionRequest(**base)


MAIN_THREAD = threading.get_ident()


async def _collect_stream(resp):
    out = []
    try:
        async for chunk in resp.body_iterator:
            out.append(chunk)
    except Exception as e:  # generator re-raises after emitting the error event
        out.append(("RAISED", type(e).__name__))
    return out


# --------------------------------------------------------------------------
# non-streaming
# --------------------------------------------------------------------------
def test_non_streaming_runs_generate_content_off_the_loop():
    sink = []
    provider, _ = _load_provider(_FakeModel(sink, response=_Response("hello there", 10, 20)))

    result = asyncio.run(provider.chat_complete(_req()))

    calls = [ident for kind, ident in sink if kind == "generate_content"]
    assert calls, "generate_content was never called"
    assert all(ident != MAIN_THREAD for ident in calls), "generate_content ran on the event loop thread"


def test_non_streaming_response_shape_unchanged():
    sink = []
    provider, _ = _load_provider(_FakeModel(sink, response=_Response("hello there", 10, 20)))

    result = asyncio.run(provider.chat_complete(_req()))

    assert result.model == "gemini-2.0-pro"
    assert result.content == "hello there"
    assert result.provider == "gemini"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_non_streaming_provider_error_propagates():
    provider, mod = _load_provider(_FakeModel([], exc=RuntimeError("quota exceeded")))
    with pytest.raises(mod.ProviderError) as exc:
        asyncio.run(provider.chat_complete(_req()))
    assert exc.value.status_code == 500
    assert "Gemini API error (chat)" in str(exc.value.detail)


def test_non_streaming_empty_response_is_provider_error():
    provider, mod = _load_provider(_FakeModel([], response=_Response("")))
    with pytest.raises(mod.ProviderError):
        asyncio.run(provider.chat_complete(_req()))


def test_non_streaming_malformed_response_text_raises_is_handled():
    provider, mod = _load_provider(_FakeModel([], response=_Response(ValueError("response blocked"))))
    with pytest.raises(mod.ProviderError) as exc:
        asyncio.run(provider.chat_complete(_req()))
    assert "Gemini API error (chat)" in str(exc.value.detail)


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------
def test_streaming_preserves_all_chunks_and_events():
    sink = []
    chunks = [_Chunk("Hel", 4, 1), _Chunk("lo ", 4, 2), _Chunk("world", 4, 3)]
    provider, _ = _load_provider(_FakeModel(sink, stream_chunks=chunks))

    async def run():
        resp = await provider.chat_complete_stream(_req())
        return await _collect_stream(resp)

    events = asyncio.run(run())
    kinds = [e["event"] for e in events]
    assert kinds == ["metadata", "content", "content", "content", "usage"]

    import json as _json
    contents = [_json.loads(e["data"])["content"] for e in events if e["event"] == "content"]
    assert contents == ["Hel", "lo ", "world"]
    usage = _json.loads(events[-1]["data"])
    assert usage == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}


def test_streaming_pulls_chunks_off_the_loop():
    sink = []
    chunks = [_Chunk("a", 4, 1), _Chunk("b", 4, 2)]
    provider, _ = _load_provider(_FakeModel(sink, stream_chunks=chunks))

    async def run():
        resp = await provider.chat_complete_stream(_req())
        return await _collect_stream(resp)

    asyncio.run(run())

    gen_calls = [ident for kind, ident in sink if kind == "generate_content"]
    next_calls = [ident for kind, ident in sink if kind == "next"]
    assert gen_calls and all(i != MAIN_THREAD for i in gen_calls)
    assert next_calls and all(i != MAIN_THREAD for i in next_calls)


def test_streaming_provider_error_emits_error_event_then_raises():
    provider, mod = _load_provider(_FakeModel([], exc=RuntimeError("stream boom")))

    async def run():
        resp = await provider.chat_complete_stream(_req())
        return await _collect_stream(resp)

    events = asyncio.run(run())
    assert events[0]["event"] == "metadata"
    assert events[1]["event"] == "error"
    assert events[-1] == ("RAISED", "ProviderError")


def test_streaming_malformed_chunk_is_handled_as_error_not_hang():
    sink = []
    # second chunk has no usage_metadata -> AttributeError inside the loop
    chunks = [_Chunk("ok", 4, 1), _Chunk("bad", with_usage=False)]
    provider, mod = _load_provider(_FakeModel(sink, stream_chunks=chunks))

    async def run():
        resp = await provider.chat_complete_stream(_req())
        return await _collect_stream(resp)

    events = asyncio.run(run())
    kinds = [e["event"] if isinstance(e, dict) else e for e in events]
    assert kinds[0] == "metadata"
    assert "content" in kinds  # the first good chunk was delivered
    assert "error" in kinds
    assert events[-1] == ("RAISED", "ProviderError")
