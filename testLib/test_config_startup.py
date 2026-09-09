"""Regression tests for serverRouter.core.config startup / auth-cache hardening.

firebase_admin, firebase_admin.credentials, firebase_admin.firestore and dotenv
are replaced with in-memory stubs, so the real config module is imported and
its own initialization logic is exercised. No firebase_admin package, no
network, no credentials.
"""
import importlib
import sys
import types

import pytest

# A fake service-account JSON. The private_key marker must never surface in any
# error message or module attribute.
FAKE_CREDS_JSON = '{"type": "service_account", "project_id": "p", "private_key": "SECRET-KEY-MATERIAL"}'
CONFIG_NAME = "serverRouter.core.config"


def _make_firebase_stub(
    *,
    existing_app=False,
    cert_error=False,
    client_error=False,
    initial_docs=("k1", "k2"),
    get_error=False,
    on_snapshot_error=False,
):
    fa = types.ModuleType("firebase_admin")
    _APP = ("APP",)
    state = {"apps": [_APP] if existing_app else [], "init_calls": 0}

    def get_app(name="[DEFAULT]"):
        if state["apps"]:
            return state["apps"][0]
        raise ValueError("The default Firebase app does not exist.")

    def initialize_app(cred=None, *a, **k):
        state["init_calls"] += 1
        state["apps"].append(_APP)
        return _APP

    fa.get_app = get_app
    fa.initialize_app = initialize_app
    fa._state = state

    creds_mod = types.ModuleType("firebase_admin.credentials")

    def Certificate(data):
        if cert_error:
            # emulate an SDK error that embeds the offending dict (secret!)
            raise ValueError(f"Invalid certificate: {data}")
        return ("CERT",)

    creds_mod.Certificate = Certificate
    fa.credentials = creds_mod

    class _Doc:
        def __init__(self, _id):
            self.id = _id

    class _Query:
        def get(self):
            if get_error:
                raise RuntimeError("firestore backend 10.9.8.7 unreachable /srv/internal/x.py")
            return [_Doc(d) for d in initial_docs]

        def on_snapshot(self, cb):
            if on_snapshot_error:
                raise RuntimeError("snapshot channel /srv/internal failed")
            return ("WATCH",)

    class _Client:
        def collection(self, name):
            return _Query()

    fs_mod = types.ModuleType("firebase_admin.firestore")

    def client():
        if client_error:
            raise RuntimeError("firestore.client failed at /srv/internal/fs.py")
        return _Client()

    fs_mod.client = client
    fs_mod.Increment = lambda n: ("INC", n)
    fs_mod.SERVER_TIMESTAMP = object()
    fa.firestore = fs_mod

    return fa


@pytest.fixture(autouse=True)
def _isolate_modules():
    keys = (CONFIG_NAME, "firebase_admin", "firebase_admin.credentials",
            "firebase_admin.firestore", "dotenv")
    saved = {k: sys.modules.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _fresh_config(monkeypatch, fb_stub, creds_json=FAKE_CREDS_JSON):
    if creds_json is None:
        monkeypatch.delenv("FIREBASE_CREDENTIALS_JSON", raising=False)
    else:
        monkeypatch.setenv("FIREBASE_CREDENTIALS_JSON", creds_json)
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)
    monkeypatch.setitem(sys.modules, "firebase_admin", fb_stub)
    monkeypatch.setitem(sys.modules, "firebase_admin.credentials", fb_stub.credentials)
    monkeypatch.setitem(sys.modules, "firebase_admin.firestore", fb_stub.firestore)
    sys.modules.pop(CONFIG_NAME, None)
    return importlib.import_module(CONFIG_NAME)


def _assert_no_secret(text):
    assert "SECRET-KEY-MATERIAL" not in text
    assert "private_key" not in text


# --------------------------------------------------------------------------
# first initialization
# --------------------------------------------------------------------------
def test_first_initialization_populates_everything(monkeypatch):
    fb = _make_firebase_stub(initial_docs=("a", "b", "c"))
    cfg = _fresh_config(monkeypatch, fb)
    assert cfg.app == ("APP",)
    assert cfg.db is not None
    assert cfg.VALID_API_KEYS == {"a", "b", "c"}
    assert cfg.api_keys_watch == ("WATCH",)
    assert fb._state["init_calls"] == 1
    assert cfg.MAX_TOKENS == 100000
    assert cfg.PROVIDERS == {}


def test_repeated_reload_does_not_reinitialize_app(monkeypatch):
    fb = _make_firebase_stub()
    cfg = _fresh_config(monkeypatch, fb)
    assert fb._state["init_calls"] == 1
    # simulate a re-import / hot reload against the same firebase process state
    importlib.reload(cfg)
    assert fb._state["init_calls"] == 1  # get_app() reused, no duplicate app
    assert cfg.VALID_API_KEYS == {"k1", "k2"}


def test_existing_app_is_reused(monkeypatch):
    fb = _make_firebase_stub(existing_app=True)
    cfg = _fresh_config(monkeypatch, fb)
    assert fb._state["init_calls"] == 0  # never called initialize_app
    assert cfg.db is not None


# --------------------------------------------------------------------------
# missing / invalid configuration
# --------------------------------------------------------------------------
def test_missing_env_var_fails_fast(monkeypatch):
    fb = _make_firebase_stub()
    with pytest.raises(RuntimeError) as exc:
        _fresh_config(monkeypatch, fb, creds_json=None)
    assert "FIREBASE_CREDENTIALS_JSON is not set" in str(exc.value)


def test_invalid_json_fails_fast_without_leaking_value(monkeypatch):
    fb = _make_firebase_stub()
    with pytest.raises(RuntimeError) as exc:
        _fresh_config(monkeypatch, fb, creds_json='{"type": "service_account", "private_key": "SECRET-KEY-MATERIAL"')
    assert "not valid JSON" in str(exc.value)
    _assert_no_secret(str(exc.value))
    _assert_no_secret(repr(exc.value))


def test_bad_certificate_fails_fast_without_leaking_fields(monkeypatch):
    fb = _make_firebase_stub(cert_error=True)
    with pytest.raises(RuntimeError) as exc:
        _fresh_config(monkeypatch, fb)
    _assert_no_secret(str(exc.value))
    _assert_no_secret(repr(exc.value))
    assert "service-account credential" in str(exc.value)


# --------------------------------------------------------------------------
# Firestore unavailable at startup -> fail fast, no unusable server
# --------------------------------------------------------------------------
def test_firestore_client_failure_fails_fast(monkeypatch):
    fb = _make_firebase_stub(client_error=True)
    with pytest.raises(RuntimeError) as exc:
        _fresh_config(monkeypatch, fb)
    assert "Could not initialise Firestore" in str(exc.value)
    assert "/srv/internal" not in str(exc.value)  # no internal path leak


def test_initial_key_load_failure_fails_fast(monkeypatch):
    fb = _make_firebase_stub(get_error=True)
    with pytest.raises(RuntimeError) as exc:
        _fresh_config(monkeypatch, fb)
    assert "refusing to start with an empty auth cache" in str(exc.value)
    assert "10.9.8.7" not in str(exc.value)


# --------------------------------------------------------------------------
# snapshot listener lifecycle
# --------------------------------------------------------------------------
def test_snapshot_listener_registration_failure_is_not_fatal(monkeypatch):
    fb = _make_firebase_stub(on_snapshot_error=True, initial_docs=("k1",))
    cfg = _fresh_config(monkeypatch, fb)
    # server still comes up with the keys it managed to load
    assert cfg.VALID_API_KEYS == {"k1"}
    assert cfg.api_keys_watch is None


def test_snapshot_update_replaces_keys_and_revokes(monkeypatch):
    fb = _make_firebase_stub(initial_docs=("k1", "k2", "revoked"))
    cfg = _fresh_config(monkeypatch, fb)
    assert cfg.VALID_API_KEYS == {"k1", "k2", "revoked"}

    class _Doc:
        def __init__(self, _id):
            self.id = _id

    cfg.update_api_keys([_Doc("k1"), _Doc("k2")], None, None)
    assert cfg.VALID_API_KEYS == {"k1", "k2"}  # 'revoked' is gone


def test_transient_empty_snapshot_keeps_cached_keys(monkeypatch):
    fb = _make_firebase_stub(initial_docs=("k1", "k2"))
    cfg = _fresh_config(monkeypatch, fb)
    cfg.update_api_keys([], None, None)  # transient empty push
    assert cfg.VALID_API_KEYS == {"k1", "k2"}  # not wiped


def test_callback_iteration_error_keeps_cached_keys(monkeypatch):
    fb = _make_firebase_stub(initial_docs=("k1", "k2"))
    cfg = _fresh_config(monkeypatch, fb)

    class _Boom:
        def __iter__(self):
            raise RuntimeError("listener error")

    cfg.update_api_keys(_Boom(), None, None)
    assert cfg.VALID_API_KEYS == {"k1", "k2"}


def test_empty_start_stays_empty(monkeypatch):
    fb = _make_firebase_stub(initial_docs=())
    cfg = _fresh_config(monkeypatch, fb)
    assert cfg.VALID_API_KEYS == set()  # nothing cached -> empty is honoured


# --------------------------------------------------------------------------
# no secret / raw credential left on the module
# --------------------------------------------------------------------------
def test_no_raw_credentials_on_module(monkeypatch):
    fb = _make_firebase_stub()
    cfg = _fresh_config(monkeypatch, fb)
    for attr in ("cred", "_FIREBASE_CREDENTIALS_JSON", "initial_keys"):
        assert not hasattr(cfg, attr), f"config unexpectedly exposes {attr}"
    # the firestore module handle is still exposed for utils.py
    assert hasattr(cfg.firestore, "Increment")
    assert hasattr(cfg.firestore, "SERVER_TIMESTAMP")
