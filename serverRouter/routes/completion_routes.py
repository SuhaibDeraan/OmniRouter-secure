import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from serverRouter.routes.utils import (
    verify_api_key,
    get_model_and_provider,
    add_usage_to_user,
    coerce_int as _coerce_int,
    usage_total_from_chunk as _usage_total_from_chunk,
)
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    user_id: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """Create a chat completion using the specified model."""
    try:
        model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
        request.model = model_name

        response = await provider.chat_complete(request)

        usage = getattr(response, "usage", None) or {}
        token_count = _coerce_int(usage.get("total_tokens")) or 0
        await run_in_threadpool(add_usage_to_user, user_id, token_count)
        return response
    except HTTPException:
        # 400 / 401 / 429 / ProviderError etc. are already the right response.
        raise
    except Exception:
        logger.exception("Unexpected error in chat completion")
        raise HTTPException(status_code=502, detail="Chat completion provider request failed")


@router.post("/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    user_id: str = Depends(verify_api_key)
):
    """Create a streaming chat completion using the specified model."""
    try:
        model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
        request.model = model_name
        response = await provider.chat_complete_stream(request)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error starting chat completion stream")
        raise HTTPException(status_code=502, detail="Chat completion provider request failed")

    async def usage_tracking_generator():
        # Providers emit a cumulative total_tokens; record only the growth so
        # repeated or malformed usage chunks can't double-count or crash the stream.
        reported_total = 0
        async for chunk in response.body_iterator:
            yield chunk  # forward every provider chunk unchanged

            total = _usage_total_from_chunk(chunk)
            if total is None or total <= reported_total:
                continue
            delta = total - reported_total
            reported_total = total
            try:
                await run_in_threadpool(add_usage_to_user, user_id, delta)
            except Exception:
                logger.exception("Failed to record streamed usage")

    return EventSourceResponse(usage_tracking_generator())


@router.post("/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    user_id: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """Generate images using the specified model."""
    try:
        model_name, provider = get_model_and_provider(request.model, IMAGE_MODELS)
        request.model = model_name

        response = await provider.generate_image(request)

        # This codebase defines no token price for image generation: IMAGE_MODELS
        # carry no tokenCost and ImageGenerationResponse has no usage field. We do
        # NOT invent one. A successful image request is recorded once as a
        # billable operation at 0 token cost (add_usage_to_user still bumps
        # total_messages / last_updated, consistent with chat and reasoning). If a
        # provider ever reports token usage we bill exactly that reported value.
        # Assigning images a token/quota cost is an open product decision.
        reported = getattr(response, "usage", None)
        token_count = _coerce_int(reported.get("total_tokens")) if isinstance(reported, dict) else None
        await run_in_threadpool(add_usage_to_user, user_id, token_count or 0)
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in image generation")
        raise HTTPException(status_code=502, detail="Image generation provider request failed")
