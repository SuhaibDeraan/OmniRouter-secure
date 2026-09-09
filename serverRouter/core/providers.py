"""LLM provider registry.

Each provider is constructed independently so a single missing credential
cannot abort application startup. Successfully constructed providers land in
``config.PROVIDERS`` (the dict the routing layer reads); the rest are recorded
in ``PROVIDER_ERRORS`` with a non-sensitive category string.
"""
import logging

from serverRouter.core.config import PROVIDERS
from serverRouter.core.datamodels import ModelProvider
from serverRouter.core.exceptions import ProviderError
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.gemini.provider import GeminiProvider
from serverRouter.providers.openai.provider import OpenAIProvider
from serverRouter.providers.stablediffusion.provider import StableDiffusionProvider
from serverRouter.providers.together.provider import TogetherAIProvider

logger = logging.getLogger(__name__)

# DeepSeek is intentionally excluded (deprecated in favour of Together).
_PROVIDER_FACTORIES = {
    ModelProvider.OPENAI: OpenAIProvider,
    ModelProvider.ANTHROPIC: AnthropicProvider,
    ModelProvider.GEMINI: GeminiProvider,
    ModelProvider.TOGETHER: TogetherAIProvider,
    ModelProvider.STABLEDIFFUSION: StableDiffusionProvider,
}

# ModelProvider -> short, non-sensitive reason it is unavailable.
PROVIDER_ERRORS = {}

_initialized = False


def _safe_reason(exc):
    """A category string for an init failure that never contains secrets."""
    message = ""
    if isinstance(exc, ProviderError):
        message = str(getattr(exc, "detail", "") or "")
    lowered = message.lower()
    if any(s in lowered for s in ("not set", "missing", "no ", "not configured", "api key", "api_key")):
        return "credentials not configured"
    return "initialization failed"


def initialize_providers(force=False):
    """Construct every provider independently. Idempotent unless ``force``."""
    global _initialized
    if _initialized and not force:
        return

    for name, factory in _PROVIDER_FACTORIES.items():
        try:
            PROVIDERS[name] = factory()
            PROVIDER_ERRORS.pop(name, None)
            logger.info("provider %s: available", name.value)
        except Exception as exc:  # isolate: one bad provider must not kill the rest
            PROVIDERS.pop(name, None)
            PROVIDER_ERRORS[name] = _safe_reason(exc)
            logger.warning("provider %s: unavailable (%s)", name.value, PROVIDER_ERRORS[name])

    _initialized = True
    available = sorted(n.value for n in _PROVIDER_FACTORIES if n in PROVIDERS)
    logger.info("providers initialized: %d/%d available %s",
                len(available), len(_PROVIDER_FACTORIES), available)


def provider_status():
    """{provider_name: 'available' | 'unavailable'} for every known provider."""
    return {
        name.value: ("available" if name in PROVIDERS else "unavailable")
        for name in _PROVIDER_FACTORIES
    }
