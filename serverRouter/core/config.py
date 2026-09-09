"""Shared configuration for the OmniRouter API: Firebase/Firestore + limits.

Public surface consumed elsewhere (unchanged):
  * ``VALID_API_KEYS``  -- set[str], the accepted API keys, kept fresh by a
                           Firestore snapshot listener.
  * ``MAX_TOKENS``      -- int lifetime token quota per user.
  * ``PROVIDERS``       -- dict, populated in-place by router.initialize_providers().
  * ``db``              -- Firestore client.
  * ``firestore``       -- the firebase_admin.firestore module (Increment, ...).
  * ``app``             -- the Firebase app.
  * ``update_api_keys`` -- the snapshot callback.
  * ``api_keys_watch``  -- the snapshot watch handle (or None).
"""
import json
import logging
import os

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

load_dotenv()

# --- static configuration ---------------------------------------------------
PROVIDERS = {}
MAX_TOKENS = 100000


def _parse_cors_origins(raw):
    return [origin.strip() for origin in (raw or "").split(",") if origin.strip()]


# Browser cross-origin access is CLOSED by default. Set CORS_ALLOWED_ORIGINS to a
# comma-separated list of exact origins (scheme + host + port) to open it for
# just those. "*" combined with credentials is unsafe and is not supported here.
CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))

# --- auth cache ------------------------------------------------------------
# Populated at startup and kept current by the api_keys snapshot listener.
VALID_API_KEYS = set()

# --- runtime handles (set by _initialize) ---------------------------------
app = None
db = None
api_keys_watch = None


def _load_credentials():
    """Build a Firebase credential from the environment.

    Raises a clean RuntimeError (never echoing the secret) on any problem.
    """
    raw = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if not raw:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_JSON is not set. Provide the Firebase "
            "service-account JSON as a single-line string via the environment "
            "or a secrets manager (see .env.example)."
        )
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        # Do not chain: the underlying error can contain the raw secret.
        raise RuntimeError("FIREBASE_CREDENTIALS_JSON is not valid JSON.") from None
    try:
        return credentials.Certificate(parsed)
    except Exception:
        logger.error(
            "FIREBASE_CREDENTIALS_JSON was rejected by credentials.Certificate "
            "(missing or malformed service-account fields)."
        )
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_JSON is not a usable service-account credential."
        ) from None


def _init_firebase_app():
    """Return the Firebase app, creating it at most once. Safe to call repeatedly."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass
    return firebase_admin.initialize_app(_load_credentials())


def update_api_keys(col_snapshot, changes=None, read_time=None):
    """Firestore ``on_snapshot`` callback: refresh ``VALID_API_KEYS``.

    Defensive by design: a transient or error snapshot must never wipe a good
    key set and lock every user out. Individual key revocations still take
    effect immediately -- they arrive as a non-empty snapshot that no longer
    contains the revoked key. Only a genuine "every key deleted" event is not
    reflected until the next non-empty snapshot or a restart, which is the
    correct trade-off against a self-inflicted total auth outage.
    """
    global VALID_API_KEYS
    try:
        new_keys = {doc.id for doc in col_snapshot}
    except Exception:
        logger.exception("api_keys snapshot callback failed; keeping cached keys")
        return
    if not new_keys and VALID_API_KEYS:
        logger.warning(
            "api_keys snapshot was empty; keeping %d cached key(s). Restart to "
            "force a full reload if every key was intentionally removed.",
            len(VALID_API_KEYS),
        )
        return
    VALID_API_KEYS = new_keys


def _initialize():
    """Bring up Firebase/Firestore and the auth cache. Runs once at import."""
    global app, db, api_keys_watch

    try:
        app = _init_firebase_app()
    except RuntimeError:
        raise
    except Exception:
        logger.exception("Firebase app initialization failed")
        raise RuntimeError("Firebase app initialization failed at startup.") from None

    try:
        db = firestore.client()
    except Exception:
        logger.exception("Could not create the Firestore client")
        raise RuntimeError("Could not initialise Firestore at startup.") from None

    try:
        update_api_keys(db.collection("api_keys").get())
    except Exception:
        logger.exception("Could not load api_keys from Firestore at startup")
        raise RuntimeError(
            "Could not load API keys from Firestore at startup; refusing to "
            "start with an empty auth cache."
        ) from None

    try:
        api_keys_watch = db.collection("api_keys").on_snapshot(update_api_keys)
    except Exception:
        api_keys_watch = None
        logger.error(
            "Could not register the api_keys snapshot listener; API keys will "
            "not refresh until the service is restarted.",
            exc_info=True,
        )


_initialize()
