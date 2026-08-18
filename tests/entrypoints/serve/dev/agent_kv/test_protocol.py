# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from vllm.entrypoints.serve.dev.agent_kv.api_router import attach_router
from vllm.entrypoints.serve.dev.agent_kv.protocol import AgentKVEventRequest
from vllm.v1.agent_kv.protocol import AgentKVEventType


def valid_event_data() -> dict[str, object]:
    return {
        "protocol_version": 1,
        "event_id": "event-a",
        "event_sequence": 9,
        "event_type": "SUSPEND",
        "emitted_at_ms": 1_000,
        "namespace": "tenant-a",
        "session_id": "session-a",
        "generation": 7,
        "hints": {
            "expected_resume_in_ms": 500,
            "priority": 2,
            "retain_for_ms": 30_000,
            "prefetch_allowed": True,
        },
        "reason": "tool",
    }


def test_event_request_converts_to_wire_safe_engine_type() -> None:
    request = AgentKVEventRequest.model_validate(valid_event_data())

    event = request.to_engine_event()

    assert event.event_type == AgentKVEventType.SUSPEND
    assert event.branch_id == "main"
    assert event.hints is not None
    assert event.hints.priority == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", 0),
        ("event_sequence", 0),
        ("namespace", ""),
        ("event_type", "TOOL_STARTED"),
        ("protocol_version", 2),
    ],
)
def test_event_request_rejects_invalid_contract_values(
    field: str, value: object
) -> None:
    data = valid_event_data()
    data[field] = value

    with pytest.raises(ValidationError):
        AgentKVEventRequest.model_validate(data)


def test_event_route_requires_agent_kv_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_AGENT_KV", "0")
    app = FastAPI()

    attach_router(app)

    assert "/v1/agent-kv/events" not in app.openapi()["paths"]


def test_event_route_is_attached_when_agent_kv_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_AGENT_KV", "1")
    app = FastAPI()

    attach_router(app)

    assert "/v1/agent-kv/events" in app.openapi()["paths"]
