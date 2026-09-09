from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core import config

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in config.VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # Also 401s if the key's backing document disappeared since the last snapshot.
    user_id = get_user_id_by_api_key(credentials.credentials)
    user_usage = get_user_usage(user_id)
    if user_usage['total_tokens'] >= config.MAX_TOKENS:
        raise HTTPException(
            status_code=429,
            detail="User has reached the maximum number of tokens"
        )
    return credentials.credentials

def get_user_id_by_api_key(api_key):
    """
    Get the user id associated with the provided API key.

    Raises HTTP 401 (never 500) when the key has no backing document — e.g. it
    was revoked after the last VALID_API_KEYS snapshot — or the document has no
    ``userid``.

    Args:
        api_key (str): The API key to look up

    Returns:
        str: The associated user id
    """
    api_key_doc = config.db.collection('api_keys').document(api_key).get()
    api_key_data = api_key_doc.to_dict() if api_key_doc is not None else None
    user_id = (api_key_data or {}).get('userid')
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user_id

def add_usage_to_user(user_id, token_count):
    """
    Atomically add token/message usage to the user's document in Firestore.

    Uses server-side field transforms so concurrent requests cannot clobber
    each other's increments, and creates the ``usage`` map if it is absent.

    Args:
        user_id (str): The ID of the user to add usage to
        token_count (int): The number of tokens to add
    """
    config.db.collection('users').document(user_id).set(
        {
            'usage': {
                'total_tokens': config.firestore.Increment(token_count),
                'total_messages': config.firestore.Increment(1),
                'last_updated': config.firestore.SERVER_TIMESTAMP,
            }
        },
        merge=True,
    )

def get_user_usage(user_id):
    """
    Get the user's usage as a dict, treating any missing document or field as
    zero so callers never hit a TypeError / 500.
    """
    user_doc = config.db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict() if user_doc is not None else None
    usage = (user_data or {}).get('usage') or {}
    return {
        'total_tokens': usage.get('total_tokens') or 0,
        'total_messages': usage.get('total_messages') or 0,
        'last_updated': usage.get('last_updated'),
    }

def get_model_and_provider(model_id: str, models_dict):
    """Get model info and provider for a given model ID."""
    model_info = models_dict.get(model_id)
    if not model_info:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}"
        )
    
    provider = config.PROVIDERS.get(model_info.provider)
    if provider is None:
        # The provider for this model is not configured / not available on this
        # server (e.g. its credentials are missing). This is a server-side
        # availability problem, not a client error.
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is temporarily unavailable: its provider is not configured on this server."
        )

    return model_info.name, provider
