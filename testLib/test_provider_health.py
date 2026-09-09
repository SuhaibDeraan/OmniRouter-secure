"""Regression tests for provider resilience (serverRouter.core.providers),
the 503 routing behavior for unconfigured providers, and the /health endpoints.

External provider SDKs and serverRouter.core.config are stubbed; the provider
factories are replaced with in-memory fakes. No network, no credentials.
"""
import importlib
import sys
import types

import pytest


def _ensure_importable(name, factory):
    try:
        importlib.import_module(name)
    except Exception:
        sys.modules[name] = factory()


# --- stub the provider SDKs imported at module load by the real provider files
def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


_ensure_importable("dotenv", lambda: _mod("dotenv", load_dotenv=lambda *a, **k: False))
_ensure_importable("openai", lambda: _mod("openai", AsyncOpenAI=object))
_ensure_importable("anthropic", lambda: _mod("anthropic", AsyncAnthropic=object, APIError=Exception))
_ensure_importable("aiohttp", lambda: _mod("aiohttp", ClientSession=object, FormData=object))
if "google.generativeai" not in sys.modules:
    _g = _mod("google")
    _gg = _mod("google.generativeai", configure=lambda **k: None,
              types=types.SimpleNamespace(GenerationConfig=object), GenerativeModel=object)
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules.setdefault("google.generativeai", _gg)


# --- stub serverRouter.core.config -----------------------------------------
class _FakeQuery:
    def __init__(self, db):
        self._db = db

    def limit(self, _n):
        return self

    def get(self):
        if self._db.firestore_error:
            raise RuntimeError("firestore backend 10.0.0.9 down /srv/internal/x.py")
        return []


class _FakeDB:
    def __init__(self):
        self.firestore_error = False

    def collection(self, _name):
        return _FakeQuery(self)


def _install_config_stub():
    m = types.ModuleType("serverRouter.core.config")
    m.PROVIDERS = {}
    m.MAX_TOKENS = 100_000
    m.VALID_API_KEYS = set()
    m.db = _FakeDB()
    m.firestore = types.SimpleNamespace(Increment=lambda n: n, SERVER_TIMESTAMP=object())
    m.api_keys_watch = ("WATCH",)
    sys.modules["serverRouter.core.config"] = m
    return m


try:
    _cfg = importlib.import_module("serverRouter.core.config")
    if not hasattr(_cfg, "db") or _cfg.db is None:
        _cfg = _install_config_stub()
except Exception:
    _cfg = _install_config_stub()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from serverRouter.core import providers as registry  # noqa: E402
from serverRouter.core.datamodels import ModelProvider  # noqa: E402
from serverRouter.core.models import CHAT_MODELS  # noqa: E402
from serverRouter.routes import health_routes  # noqa: E402
from serverRouter.routes.utils import get_model_and_provider  # noqa: E402


class _OKProvider:
    instances = 0

    def __init__(self):
        type(self).instances += 1


def _factory_ok():
    return _OKProvider()


def _factory_missing_key():
    from serverRouter.core.exceptions import ProviderError
    raise ProviderError("OPENAI_API_KEY not set in environment.")


def _factory_boom():
    raise RuntimeError("internal failure /srv/app/secrets.py:9")


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    """Point registry / health_routes / utils at one config + one PROVIDERS dict.

    Other test files stub serverRouter.core.config differently; this keeps this
    file's expectations stable regardless of import order.
    """
    from serverRouter.routes import utils

    shared_providers = {}
    _cfg.PROVIDERS = shared_providers
    if not isinstance(getattr(_cfg, "db", None), _FakeDB):
        _cfg.db = _FakeDB()
    _cfg.db.firestore_error = False

    monkeypatch.setattr(registry, "PROVIDERS", shared_providers)
    monkeypatch.setattr(health_routes, "config", _cfg)
    monkeypatch.setattr(utils, "config", _cfg)
    registry.PROVIDER_ERRORS.clear()
    registry._initialized = False
    _OKProvider.instances = 0
    yield
    registry.PROVIDER_ERRORS.clear()
    registry._initialized = False


def _set_factories(monkeypatch, mapping):
    monkeypatch.setattr(registry, "_PROVIDER_FACTORIES", dict(mapping))


ALL_OK = None  # filled in below once ModelProvider is importable
ALL_OK = {
    ModelProvider.OPENAI: _factory_ok,
    ModelProvider.ANTHROPIC: _factory_ok,
    ModelProvider.GEMINI: _factory_ok,
    ModelProvider.TOGETHER: _factory_ok,
    ModelProvider.STABLEDIFFUSION: _factory_ok,
}


@pytest.fixture
def health_client():
    app = FastAPI()
    app.include_router(health_routes.router)
    return TestClient(app)


# --------------------------------------------------------------------------
# provider resilience
# --------------------------------------------------------------------------
def test_one_missing_key_does_not_block_other_providers(monkeypatch):
    factories = dict(ALL_OK)
    factories[ModelProvider.OPENAI] = _factory_missing_key
    _set_factories(monkeypatch, factories)

    registry.initialize_providers(force=True)

    status = registry.provider_status()
    assert status["openai"] == "unavailable"
    assert status["anthropic"] == "available"
    assert status["gemini"] == "available"
    assert status["together"] == "available"
    assert status["stablediffusion"] == "available"
    assert ModelProvider.OPENAI not in _cfg.PROVIDERS
    assert ModelProvider.ANTHROPIC in _cfg.PROVIDERS
    assert registry.PROVIDER_ERRORS[ModelProvider.OPENAI] == "credentials not configured"


def test_all_providers_missing(monkeypatch):
    _set_factories(monkeypatch, {k: _factory_missing_key for k in ALL_OK})
    registry.initialize_providers(force=True)
    assert _cfg.PROVIDERS == {}
    assert all(v == "unavailable" for v in registry.provider_status().values())


def test_provider_constructor_raising_generic_exception_is_isolated(monkeypatch):
    factories = dict(ALL_OK)
    factories[ModelProvider.GEMINI] = _factory_boom
    _set_factories(monkeypatch, factories)

    registry.initialize_providers(force=True)

    assert registry.provider_status()["gemini"] == "unavailable"
    assert registry.PROVIDER_ERRORS[ModelProvider.GEMINI] == "initialization failed"
    # the internal path from the exception must not be retained anywhere public
    assert "secrets.py" not in registry.PROVIDER_ERRORS[ModelProvider.GEMINI]
    assert registry.provider_status()["openai"] == "available"


def test_repeated_initialization_does_not_duplicate_providers(monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    assert _OKProvider.instances == 5
    registry.initialize_providers()  # idempotent, guard should skip
    registry.initialize_providers()
    assert _OKProvider.instances == 5
    registry.initialize_providers(force=True)  # explicit rebuild
    assert _OKProvider.instances == 10


def test_reload_of_registry_module_keeps_single_provider_set(monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    before = dict(_cfg.PROVIDERS)
    importlib.reload(registry)
    # reload resets _initialized; router.py calls initialize_providers() again
    registry._PROVIDER_FACTORIES = dict(ALL_OK)
    registry.initialize_providers()
    assert set(_cfg.PROVIDERS) == set(before)


# --------------------------------------------------------------------------
# get_model_and_provider routing
# --------------------------------------------------------------------------
def test_lookup_for_available_provider(monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    name, provider = get_model_and_provider("gpt-4o", CHAT_MODELS)
    assert name == "gpt-4o"
    assert isinstance(provider, _OKProvider)


def test_lookup_for_unavailable_provider_is_503(monkeypatch):
    factories = dict(ALL_OK)
    factories[ModelProvider.OPENAI] = _factory_missing_key
    _set_factories(monkeypatch, factories)
    registry.initialize_providers(force=True)

    with pytest.raises(HTTPException) as exc:
        get_model_and_provider("gpt-4o", CHAT_MODELS)  # gpt-4o is an OpenAI model
    assert exc.value.status_code == 503
    detail = str(exc.value.detail)
    assert "temporarily unavailable" in detail
    for leak in ("OPENAI_API_KEY", "/srv", "secret"):
        assert leak not in detail


def test_unknown_model_still_400(monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    with pytest.raises(HTTPException) as exc:
        get_model_and_provider("no-such-model", CHAT_MODELS)
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------
# /health and /health/ready
# --------------------------------------------------------------------------
def test_health_liveness_always_ok(health_client, monkeypatch):
    _set_factories(monkeypatch, {k: _factory_missing_key for k in ALL_OK})
    registry.initialize_providers(force=True)
    _cfg.db.firestore_error = True  # even with everything down, liveness is ok
    resp = health_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "schema_version": 1}


def test_readiness_all_healthy_is_200(health_client, monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    resp = health_client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["firestore"] == "ok"
    assert set(body["providers"].values()) == {"available"}
    assert body["schema_version"] == 1


def test_readiness_one_provider_unavailable_stays_ready(health_client, monkeypatch):
    factories = dict(ALL_OK)
    factories[ModelProvider.OPENAI] = _factory_missing_key
    _set_factories(monkeypatch, factories)
    registry.initialize_providers(force=True)
    resp = health_client.get("/health/ready")
    assert resp.status_code == 200  # other providers can still serve
    body = resp.json()
    assert body["status"] == "ready"
    assert body["providers"]["openai"] == "unavailable"
    assert body["providers"]["anthropic"] == "available"


def test_readiness_no_providers_is_503(health_client, monkeypatch):
    _set_factories(monkeypatch, {k: _factory_missing_key for k in ALL_OK})
    registry.initialize_providers(force=True)
    resp = health_client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_readiness_firestore_unavailable_is_503(health_client, monkeypatch):
    _set_factories(monkeypatch, ALL_OK)
    registry.initialize_providers(force=True)
    _cfg.db.firestore_error = True
    resp = health_client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["firestore"] == "unavailable"


def test_health_response_contains_no_secrets_or_paths(health_client, monkeypatch):
    factories = dict(ALL_OK)
    factories[ModelProvider.GEMINI] = _factory_boom
    _set_factories(monkeypatch, factories)
    registry.initialize_providers(force=True)
    _cfg.db.firestore_error = True

    for path in ("/health", "/health/ready"):
        text = health_client.get(path).text
        for leak in ("secrets.py", "/srv", "10.0.0.9", "OPENAI_API_KEY", "private_key", "Traceback"):
            assert leak not in text
