# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Development endpoint for upstream AgentKV lifecycle events."""

from fastapi import APIRouter, FastAPI, Request

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.dev.agent_kv.protocol import (
    AgentKVEventRequest,
    AgentKVEventResponse,
)
from vllm.logger import init_logger

router = APIRouter(prefix="/v1/agent-kv", tags=["AgentKV"])
logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/events", response_model=AgentKVEventResponse)
async def apply_agent_kv_event(
    event: AgentKVEventRequest, raw_request: Request
) -> AgentKVEventResponse:
    """Apply one idempotent, ordered agent lifecycle event."""
    result = await engine_client(raw_request).apply_agent_kv_event(
        event.to_engine_event()
    )
    return AgentKVEventResponse.model_validate(result)


def attach_router(app: FastAPI) -> None:
    if not envs.VLLM_ENABLE_AGENT_KV:
        logger.info(
            "AgentKV development endpoint is disabled; set "
            "VLLM_ENABLE_AGENT_KV=1 to register it"
        )
        return
    app.include_router(router)
