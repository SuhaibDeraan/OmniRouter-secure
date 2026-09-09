import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from serverRouter.routes.utils import (
    verify_api_key,
    get_model_and_provider,
    add_usage_to_user,
    usage_total_from_chunk,
)
from serverRouter.core.datamodels import (
    ChatReasoningRequest,
    ChatReasoningResponse,
    ReasoningTokenUsage
)
from serverRouter.core.models import REASONING_MODELS
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["reasoning"])

@router.post("/reason/completions")
async def create_reasoning_completion(
    request: ChatReasoningRequest,
    user_id: str = Depends(verify_api_key)
) -> ChatReasoningResponse:
    """
    Create a reasoning completion using a model with extended thinking capabilities.
    This endpoint generates a response with detailed reasoning process.
    """
    try:
        model_name, provider = get_model_and_provider(request.model, REASONING_MODELS)
        request.model = model_name

        # Ensure the provider supports reasoning
        if not hasattr(provider, 'chat_reason_complete'):
            raise HTTPException(
                status_code=400,
                detail=f"Provider for model {model_name} does not support reasoning capabilities"
            )

        # Get the response with reasoning
        response = await provider.chat_reason_complete(request)

        # Track usage
        total_tokens = response.usage.total_tokens
        await run_in_threadpool(add_usage_to_user, user_id, total_tokens)

        return response
    except HTTPException:
        # 400 / 401 / 429 / ProviderError etc. are already the right response.
        raise
    except Exception:
        logger.exception("Unexpected error in reasoning completion")
        raise HTTPException(status_code=502, detail="Reasoning provider request failed")

@router.post("/reason/completions/stream")
async def create_reasoning_completion_stream(
    request: ChatReasoningRequest,
    user_id: str = Depends(verify_api_key)
):
    """
    Stream a reasoning completion using a model with extended thinking capabilities.
    This endpoint streams both the reasoning process and final response.
    """
    try:
        model_name, provider = get_model_and_provider(request.model, REASONING_MODELS)
        request.model = model_name

        # Ensure the provider supports reasoning
        if not hasattr(provider, 'chat_reason_complete_stream'):
            raise HTTPException(
                status_code=400,
                detail=f"Provider for model {model_name} does not support streaming reasoning capabilities"
            )

        # Set stream to true for request
        request.stream = True

        # Get streaming response
        response = await provider.chat_reason_complete_stream(request)

        # Track usage from stream data. Providers emit a cumulative total; record
        # only its growth so repeated/malformed usage chunks can't double-count or
        # crash the stream.
        async def usage_tracking_generator():
            reported_total = 0
            async for chunk in response.body_iterator:
                yield chunk  # forward every provider chunk unchanged

                total = usage_total_from_chunk(chunk)
                if total is None or total <= reported_total:
                    continue
                delta = total - reported_total
                reported_total = total
                try:
                    await run_in_threadpool(add_usage_to_user, user_id, delta)
                except Exception:
                    logger.exception("Failed to record streamed reasoning usage")

        return EventSourceResponse(usage_tracking_generator())
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in streaming reasoning completion")
        raise HTTPException(status_code=502, detail="Reasoning provider streaming request failed")
