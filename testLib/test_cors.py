"""Regression tests for CORS configuration.

The old middleware used allow_origins=["*"] with allow_credentials=True, which
Starlette resolves by *reflecting any Origin* and sending
Access-Control-Allow-Credentials: true -- i.e. every website could make
credentialed cross-origin calls. Config is now an explicit allow-list
(CORS_ALLOWED_ORIGINS), closed by default.

Pure: builds a tiny app with CORSMiddleware directly; no firebase, no network.
The CORS_ALLOWED_ORIGINS parsing itself is covered in test_config_startup.py.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


def _make_app(origins):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return TestClient(app)


def _preflight(client, origin):
    return client.options(
        "/ping",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


# --------------------------------------------------------------------------
# behavior with an explicit allow-list
# --------------------------------------------------------------------------
def test_allowed_origin_gets_cors_headers():
    client = _make_app(["https://app.example.com"])
    resp = client.get("/ping", headers={"Origin": "https://app.example.com"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_disallowed_origin_gets_no_cors_headers():
    client = _make_app(["https://app.example.com"])
    resp = client.get("/ping", headers={"Origin": "https://evil.example.net"})
    assert resp.status_code == 200  # request still served...
    # ...but the browser gets no ACAO header for this origin
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example.net"
    assert "access-control-allow-origin" not in resp.headers


def test_disallowed_origin_preflight_is_rejected():
    client = _make_app(["https://app.example.com"])
    resp = _preflight(client, "https://evil.example.net")
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example.net"


# --------------------------------------------------------------------------
# default (closed) configuration
# --------------------------------------------------------------------------
def test_default_closed_allows_no_cross_origin():
    client = _make_app([])  # CORS_ALLOWED_ORIGINS unset -> []
    resp = client.get("/ping", headers={"Origin": "https://anything.example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers
    assert "access-control-allow-credentials" not in resp.headers


# --------------------------------------------------------------------------
# documents why the old config was unsafe
# --------------------------------------------------------------------------
def test_old_wildcard_plus_credentials_reflects_any_origin_on_preflight():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,  # the old, unsafe combination
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(app)
    # A JSON API POST triggers a preflight; Starlette reflects the attacker
    # origin AND allows credentials -> any site can make credentialed calls.
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://evil.example.net",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://evil.example.net"
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_new_allowlist_rejects_attacker_origin_on_preflight():
    client = _make_app(["https://app.example.com"])
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://evil.example.net",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example.net"
