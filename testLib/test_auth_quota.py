"""Regression tests for serverRouter.routes.utils auth / quota handling.

A fake Firestore stands in for the real one (which config.py would otherwise
open at import time). No firebase_admin, no network, no credentials.

Covers:
  1. stale / missing api_keys document -> 401, never 500
  2. missing usage field / user document -> treated as zero, never 500
  3. usage increments are atomic (server-side transforms, no read-modify-write)
"""
import importlib
import sys
import types

import pytest


# --------------------------------------------------------------------------
# Fake Firestore
# --------------------------------------------------------------------------
class _Increment:
    def __init__(self, amount):
        self.amount = amount


_SERVER_TIMESTAMP = object()


class _Snapshot:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class _DocRef:
    def __init__(self, db, collection, doc_id):
        self._db = db
        self._key = (collection, doc_id)

    def get(self):
        self._db.reads.append(self._key)
        return _Snapshot(self._db.store.get(self._key))

    def set(self, data, merge=False):
        self._db.writes.append(("set", self._key, data, merge))
        existing = self._db.store.get(self._key) if merge else None
        self._db.store[self._key] = _apply(existing, data)

    def update(self, data):
        self._db.writes.append(("update", self._key, data))
        existing = self._db.store.get(self._key) or {}
        existing.update(data)
        self._db.store[self._key] = existing


class _Collection:
    def __init__(self, db, name):
        self._db = db
        self._name = name

    def document(self, doc_id):
        return _DocRef(self._db, self._name, doc_id)


class FakeFirestore:
    def __init__(self, store=None):
        self.store = dict(store or {})
        self.reads = []
        self.writes = []

    def collection(self, name):
        return _Collection(self, name)


def _apply(existing, incoming):
    """Merge ``incoming`` into ``existing``, resolving sentinel transforms."""
    if isinstance(incoming, dict):
        out = dict(existing) if isinstance(existing, dict) else {}
        for k, v in incoming.items():
            if isinstance(v, _Increment):
                base = out.get(k, 0)
                out[k] = (base if isinstance(base, (int, float)) else 0) + v.amount
            elif v is _SERVER_TIMESTAMP:
                out[k] = "2026-01-01T00:00:00Z"
            elif isinstance(v, dict):
                out[k] = _apply(out.get(k), v)
            else:
                out[k] = v
        return out
    return incoming


# --------------------------------------------------------------------------
# Stub serverRouter.core.config before importing the module under test
# --------------------------------------------------------------------------
def _install_config_stub():
    mod = types.ModuleType("serverRouter.core.config")
    mod.VALID_API_KEYS = set()
    mod.MAX_TOKENS = 1000
    mod.PROVIDERS = {}
    mod.db = FakeFirestore()
    mod.firestore = types.SimpleNamespace(
        Increment=_Increment, SERVER_TIMESTAMP=_SERVER_TIMESTAMP
    )
    sys.modules["serverRouter.core.config"] = mod
    return mod


try:
    importlib.import_module("serverRouter.core.config")
except Exception:
    _install_config_stub()

from serverRouter.core import config as _config  # noqa: E402
from serverRouter.routes import utils  # noqa: E402


@pytest.fixture
def db():
    fake = FakeFirestore()
    # utils.config IS this module object; set every surface it reads so the test
    # is independent of whichever module first populated sys.modules.
    utils.config.db = fake
    utils.config.VALID_API_KEYS = set()
    utils.config.MAX_TOKENS = 1000
    utils.config.firestore = types.SimpleNamespace(
        Increment=_Increment, SERVER_TIMESTAMP=_SERVER_TIMESTAMP
    )
    return fake


class _Creds:
    def __init__(self, token):
        self.credentials = token


# --------------------------------------------------------------------------
# Defect 1 — stale / missing api_keys document
# --------------------------------------------------------------------------
def test_key_not_in_valid_set_is_401(db):
    with pytest.raises(utils.HTTPException) as exc:
        utils.verify_api_key(_Creds("nope"))
    assert exc.value.status_code == 401


def test_key_valid_but_api_keys_doc_missing_is_401_not_500(db):
    db_key = "omni-stale"
    _config.VALID_API_KEYS.add(db_key)  # present in snapshot...
    # ...but no ('api_keys', db_key) document in the store
    with pytest.raises(utils.HTTPException) as exc:
        utils.verify_api_key(_Creds(db_key))
    assert exc.value.status_code == 401


def test_api_keys_doc_without_userid_is_401(db):
    key = "omni-nouser"
    _config.VALID_API_KEYS.add(key)
    db.store[("api_keys", key)] = {"note": "no userid here"}
    with pytest.raises(utils.HTTPException) as exc:
        utils.verify_api_key(_Creds(key))
    assert exc.value.status_code == 401


def test_get_user_id_helper_raises_401_on_missing_doc(db):
    with pytest.raises(utils.HTTPException) as exc:
        utils.get_user_id_by_api_key("omni-ghost")
    assert exc.value.status_code == 401


# --------------------------------------------------------------------------
# Defect 2 — missing usage field / user document
# --------------------------------------------------------------------------
def test_missing_user_document_treated_as_zero(db):
    assert utils.get_user_usage("ghost-user") == {
        "total_tokens": 0,
        "total_messages": 0,
        "last_updated": None,
    }


def test_user_document_without_usage_field_treated_as_zero(db):
    db.store[("users", "u1")] = {"email": "x@example.com"}
    assert utils.get_user_usage("u1")["total_tokens"] == 0


def test_verify_api_key_passes_when_usage_absent(db):
    key = "omni-live"
    _config.VALID_API_KEYS.add(key)
    db.store[("api_keys", key)] = {"userid": "u1"}
    # no ('users', 'u1') doc at all
    assert utils.verify_api_key(_Creds(key)) == key


def test_verify_api_key_429_when_over_limit(db):
    key = "omni-heavy"
    _config.VALID_API_KEYS.add(key)
    db.store[("api_keys", key)] = {"userid": "u2"}
    db.store[("users", "u2")] = {"usage": {"total_tokens": 1000}}
    with pytest.raises(utils.HTTPException) as exc:
        utils.verify_api_key(_Creds(key))
    assert exc.value.status_code == 429


# --------------------------------------------------------------------------
# Defect 3 — atomic usage increments
# --------------------------------------------------------------------------
def test_add_usage_is_atomic_and_accumulates(db):
    utils.add_usage_to_user("u3", 100)
    utils.add_usage_to_user("u3", 100)
    usage = db.store[("users", "u3")]["usage"]
    assert usage["total_tokens"] == 200
    assert usage["total_messages"] == 2


def test_add_usage_creates_missing_usage_map(db):
    db.store[("users", "u4")] = {"email": "y@example.com"}
    utils.add_usage_to_user("u4", 42)
    doc = db.store[("users", "u4")]
    assert doc["email"] == "y@example.com"  # sibling field preserved
    assert doc["usage"]["total_tokens"] == 42
    assert doc["usage"]["total_messages"] == 1


def test_add_usage_does_no_read_modify_write(db):
    utils.add_usage_to_user("u5", 10)
    assert db.reads == []  # never reads the doc first
    kinds = [w[0] for w in db.writes]
    assert kinds == ["set"]
    assert db.writes[0][3] is True  # merge=True
    payload = db.writes[0][2]["usage"]
    assert isinstance(payload["total_tokens"], _Increment)
    assert payload["last_updated"] is _SERVER_TIMESTAMP
