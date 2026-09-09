import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from serverRouter.routes.utils import verify_api_key
from serverRouter.core.datamodels import SmartRouterRequest
from serverRouter.smartRouter.main import SmartRouter
from serverRouter.smartRouter.param_types import CostType, LatencyType
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["smart"])


def _validate_smart_request(request: SmartRouterRequest) -> None:
    """Reject bad input with a 4xx before the router generator runs."""
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    try:
        LatencyType.from_value(request.max_latency)
        CostType.from_value(request.max_cost)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/smartRouterStream")
async def smartRouterStream(request: SmartRouterRequest, api_key: str = Depends(verify_api_key)):
    _validate_smart_request(request)

    def _guarded():
        try:
            yield from SmartRouter(
                request.messages, request.max_latency, request.max_cost, request.model_list
            )
        except Exception:
            logger.exception("smart router stream failed")
            yield {"event": "error", "data": json.dumps({"error": "Smart routing failed"})}

    return EventSourceResponse(_guarded())


@router.post("/smartRouter")
async def smartRouter(request: SmartRouterRequest, api_key: str = Depends(verify_api_key)):
    _validate_smart_request(request)

    try:
        for event in SmartRouter(
            request.messages, request.max_latency, request.max_cost, request.model_list
        ):
            if event["event"] == "return":
                return json.loads(event["data"])
    except HTTPException:
        raise
    except Exception:
        logger.exception("smart router failed")
        raise HTTPException(status_code=502, detail="Smart routing failed")

    raise HTTPException(status_code=500, detail="Smart router did not produce a result")
