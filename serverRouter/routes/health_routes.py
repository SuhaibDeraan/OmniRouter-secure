"""Health / readiness endpoints.

  GET /health        -- liveness. 200 whenever the process can serve a request.
  GET /health/ready  -- readiness. 200 when Firestore is reachable AND at least
                        one provider is available; 503 otherwise. The body
                        always reports per-dependency state.

Response shape (stable, schema_version 1):
  /health        -> {"status": "ok", "schema_version": 1}
  /health/ready  -> {"status": "ready" | "not_ready",
                     "schema_version": 1,
                     "firestore": "ok" | "unavailable",
                     "providers": {"<name>": "available" | "unavailable", ...}}

No credentials, environment values, paths or stack traces are ever returned.
"""
import logging

from fastapi import APIRouter, Response

from serverRouter.core import config
from serverRouter.core import providers as provider_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

HEALTH_SCHEMA_VERSION = 1


def _firestore_status():
    db = getattr(config, "db", None)
    if db is None:
        return "unavailable"
    try:
        db.collection("api_keys").limit(1).get()
        return "ok"
    except Exception:
        logger.warning("health: Firestore readiness check failed", exc_info=True)
        return "unavailable"


@router.get("/health")
async def health():
    """Liveness probe: the process is up. Always 200 while reachable."""
    return {"status": "ok", "schema_version": HEALTH_SCHEMA_VERSION}


@router.get("/health/ready")
async def readiness(response: Response):
    """Readiness probe: dependencies required for normal operation are usable."""
    providers = provider_registry.provider_status()
    firestore_status = _firestore_status()
    any_provider_available = any(state == "available" for state in providers.values())

    ready = firestore_status == "ok" and any_provider_available
    if not ready:
        response.status_code = 503

    return {
        "status": "ready" if ready else "not_ready",
        "schema_version": HEALTH_SCHEMA_VERSION,
        "firestore": firestore_status,
        "providers": providers,
    }
