# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Development endpoint for upstream AgentKV lifecycle events."""

from fastapi import APIRouter, FastAPI, Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.dev.agent_kv.protocol import (
    AgentKVEventRequest,
    AgentKVEventResponse,
)

router = APIRouter(prefix="/v1/agent-kv", tags=["AgentKV"])


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/events", response_model=AgentKVEventResponse)
async def apply_agent_kv_event(
    event: AgentKVEventRequest, raw_request: Request
) -> AgentKVEventResponse:
    """Record one idempotent, ordered agent lifecycle event.

    This development endpoint updates lifecycle metadata only. It does not
    alter cache placement or eviction behavior.
    """
    result = await engine_client(raw_request).apply_agent_kv_event(
        event.to_engine_event()
    )
    return AgentKVEventResponse.model_validate(result)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
