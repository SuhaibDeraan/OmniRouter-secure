"""Regression tests for the smart-router API surface.

These exercise the real request-validation, parameter-conversion,
model-ranking and response-shaping code. The only things replaced are the
two genuinely external dependencies:

  * serverRouter.core.config          -> firebase_admin / live Firestore
  * serverRouter.smartRouter.main.classify_prompt -> openai / embedding pickle

`classify_prompt` is monkeypatched on the already-imported ``main`` module,
so the real ranking pipeline runs against the real database/_task_models.json.
No provider API keys, no Firestore, no network.
"""
import importlib
import json
import sys
import types

import pytest


def _ensure_importable(name, factory):
    """Install a stand-in module only if the real one cannot be imported."""
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
        def __getattr__(self, _name):
            raise AssertionError("Firestore must not be touched in these tests")

    m.db = _ForbiddenDB()
    return m


def _classify_stub():
    m = types.ModuleType("serverRouter.smartRouter.classifyPrompt")
    m.classify_prompt = lambda query: {}
    return m


_ensure_importable("serverRouter.core.config", _config_stub)
_ensure_importable("serverRouter.smartRouter.classifyPrompt", _classify_stub)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import serverRouter.smartRouter.main as sr_main  # noqa: E402
from serverRouter.routes import smart_routes  # noqa: E402
from serverRouter.routes.smart_routes import _validate_smart_request  # noqa: E402
from serverRouter.routes.utils import verify_api_key  # noqa: E402

_CLASSIFICATION: dict = {}


def set_classification(mapping):
    _CLASSIFICATION.clear()
    _CLASSIFICATION.update(mapping)


@pytest.fixture(autouse=True)
def _patch_classification(monkeypatch):
    _CLASSIFICATION.clear()
    monkeypatch.setattr(sr_main, "classify_prompt", lambda query: dict(_CLASSIFICATION))
    yield
    _CLASSIFICATION.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(smart_routes.router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    return TestClient(app)


def _body(messages=None, **overrides):
    payload = {
        "messages": messages if messages is not None else [{"role": "user", "content": "hello"}],
        "max_latency": "balanced",
        "max_cost": "balanced",
        "model_list": [],
    }
    payload.update(overrides)
    return payload


# 1. Valid enum latency + cost
def test_valid_enum_latency_and_cost(client):
    set_classification({"coding": 1.0})
    resp = client.post("/v1/smartRouter", json=_body(max_latency="balanced", max_cost="balanced"))
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data["model"] == "deepseek-v3"  # only coding model within 1.5s / $10


# 2. Numeric latency + cost (JSON numbers)
def test_numeric_latency_and_cost(client):
    set_classification({"coding": 1.0})
    resp = client.post("/v1/smartRouter", json=_body(max_latency=2.5, max_cost=5))
    assert resp.status_code == 200
    assert resp.json()["model"] == "deepseek-v3"


# 3. Numeric strings such as "2.5" / "0.5"
@pytest.mark.parametrize("lat,cost", [("2.5", "5"), ("0.5", "0.5"), ("1.5", "10")])
def test_numeric_string_latency_and_cost(client, lat, cost):
    set_classification({"coding": 1.0})
    resp = client.post("/v1/smartRouter", json=_body(max_latency=lat, max_cost=cost))
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)
    assert "model" in resp.json()


# 4 & 5. Invalid latency / cost -> 400, never 500
@pytest.mark.parametrize("field,value", [("max_latency", "turbo"), ("max_cost", "expensive")])
def test_invalid_constraint_is_400_not_500(client, field, value):
    resp = client.post("/v1/smartRouter", json=_body(**{field: value}))
    assert resp.status_code == 400


@pytest.mark.parametrize("field,value", [("max_latency", "turbo"), ("max_cost", "expensive")])
def test_invalid_constraint_is_400_on_stream(client, field, value):
    resp = client.post("/v1/smartRouterStream", json=_body(**{field: value}))
    assert resp.status_code == 400


# 6. Empty messages -> 422
def test_empty_messages_is_422(client):
    assert client.post("/v1/smartRouter", json=_body(messages=[])).status_code == 422


def test_empty_messages_is_422_on_stream(client):
    assert client.post("/v1/smartRouterStream", json=_body(messages=[])).status_code == 422


# 7 & 11. No candidate satisfies constraints -> default model, never 500
def test_no_candidate_returns_default_model(client):
    set_classification({"coding": 1.0})
    resp = client.post("/v1/smartRouter", json=_body(max_latency="0.001", max_cost="0.001"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "gpt-4o-mini"
    assert "No models meet criteria" in data["message"]


def test_extremely_restrictive_limits_do_not_crash(client):
    set_classification({"coding": 1.0, "math": 1.0})
    resp = client.post("/v1/smartRouter", json=_body(max_latency=0.0001, max_cost=0.0001))
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4o-mini"


# 8. /v1/smartRouter returns a JSON object, not a JSON-encoded string
def test_response_is_json_object_not_string(client):
    set_classification({"coding": 1.0})
    parsed = client.post("/v1/smartRouter", json=_body()).json()
    assert isinstance(parsed, dict)
    assert not isinstance(parsed, str)
    assert {"model", "cost", "latency", "message"} <= set(parsed)


# 9. /v1/smartRouterStream produces the expected SSE event structure
def test_stream_produces_expected_sse_events(client):
    set_classification({"coding": 1.0})
    resp = client.post("/v1/smartRouterStream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: metadata" in body
    assert "event: return" in body
    return_chunk = body.split("event: return", 1)[1]
    data_line = next(ln for ln in return_chunk.splitlines() if ln.startswith("data:"))
    payload = json.loads(data_line[len("data:"):].strip())
    assert isinstance(payload, dict) and "model" in payload


# 10. Empty / unknown classification path
def test_empty_classification_returns_default(client):
    set_classification({})
    resp = client.post("/v1/smartRouter", json=_body())
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4o-mini"


def test_unknown_task_id_classification_returns_default(client):
    set_classification({"totally_unknown_task": 0.99})
    resp = client.post("/v1/smartRouter", json=_body())
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4o-mini"


# --------------------------------------------------------------------------
# Router-internal failure (e.g. embedding service down) -> 502, no leak
# --------------------------------------------------------------------------
def _boom(_query):
    raise RuntimeError("embedding backend /srv/internal/embed.py unreachable")


def test_router_internal_failure_is_502(client, monkeypatch):
    monkeypatch.setattr(sr_main, "classify_prompt", _boom)
    resp = client.post("/v1/smartRouter", json=_body())
    assert resp.status_code == 502
    body = resp.text
    for leak in ("Traceback", "RuntimeError", "/srv", "embed.py"):
        assert leak not in body
    assert resp.json()["detail"] == "Smart routing failed"


def test_router_internal_failure_on_stream_emits_error_event(client, monkeypatch):
    monkeypatch.setattr(sr_main, "classify_prompt", _boom)
    resp = client.post("/v1/smartRouterStream", json=_body())
    assert resp.status_code == 200  # stream opened before the failure
    assert "event: error" in resp.text
    for leak in ("Traceback", "RuntimeError", "/srv", "embed.py"):
        assert leak not in resp.text


# 12. model_list handling — supplied vs omitted
def test_model_list_supplied_restricts_choice(client):
    set_classification({"coding": 1.0})
    resp = client.post(
        "/v1/smartRouter",
        json=_body(max_latency="performance", max_cost="performance", model_list=["claude-3-7-sonnet"]),
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-3-7-sonnet"


def test_model_list_omitted_uses_full_ranking(client):
    set_classification({"coding": 1.0})
    payload = _body(max_latency="performance", max_cost="performance")
    payload.pop("model_list")  # omitted entirely -> defaults to []
    resp = client.post("/v1/smartRouter", json=payload)
    assert resp.status_code == 200
    assert resp.json()["model"] == "deepseek-v3"


# Direct unit coverage of the validation helper (no HTTP layer)
def _fake_request(messages, max_latency="balanced", max_cost="balanced"):
    return types.SimpleNamespace(messages=messages, max_latency=max_latency, max_cost=max_cost)


def test_validate_helper_rejects_empty_messages():
    with pytest.raises(HTTPException) as exc:
        _validate_smart_request(_fake_request([]))
    assert exc.value.status_code == 422


@pytest.mark.parametrize("field", ["max_latency", "max_cost"])
def test_validate_helper_rejects_bad_constraint(field):
    req = _fake_request([{"role": "user", "content": "x"}], **{field: "nonsense"})
    with pytest.raises(HTTPException) as exc:
        _validate_smart_request(req)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("lat,cost", [("balanced", "cheap"), (2.5, 0.5), ("2.5", "0.5")])
def test_validate_helper_accepts_valid_constraints(lat, cost):
    _validate_smart_request(_fake_request([{"role": "user", "content": "x"}], lat, cost))
