# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest

from vllm.v1.agent_kv.protocol import (
    AgentKVEvent,
    AgentKVEventHints,
    AgentKVEventType,
    AgentKVRequestMetadata,
    parse_agent_kv_request_metadata,
)


def test_event_round_trips_through_msgpack() -> None:
    event = AgentKVEvent(
        event_id="event-a",
        event_sequence=4,
        event_type=AgentKVEventType.RESUME_PENDING,
        emitted_at_ms=1_000,
        namespace="tenant-a",
        session_id="session-a",
        generation=7,
        hints=AgentKVEventHints(
            expected_resume_in_ms=250,
            priority=3,
            retain_for_ms=5_000,
            prefetch_allowed=True,
        ),
    )

    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(event), type=AgentKVEvent)

    assert decoded == event


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"generation": 0}, "generation"),
        ({"namespace": " "}, "namespace"),
        ({"protocol_version": 2}, "protocol version"),
    ],
)
def test_request_metadata_rejects_invalid_values(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "namespace": "tenant-a",
        "session_id": "session-a",
        "generation": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        AgentKVRequestMetadata(**values)  # type: ignore[arg-type]


def test_raw_request_metadata_values_are_parsed() -> None:
    metadata = parse_agent_kv_request_metadata(
        session_id="session-a",
        namespace="tenant-a",
        generation="7",
        branch_id=None,
        protocol_version=None,
    )

    assert metadata.generation == 7
    assert metadata.branch_id == "main"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"session_id": None}, "session_id"),
        ({"namespace": None}, "Namespace"),
        ({"generation": "seven"}, "positive integer"),
        ({"branch_id": ""}, "Branch-ID"),
    ],
)
def test_raw_request_metadata_rejects_partial_values(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "session_id": "session-a",
        "namespace": "tenant-a",
        "generation": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        parse_agent_kv_request_metadata(**values)
