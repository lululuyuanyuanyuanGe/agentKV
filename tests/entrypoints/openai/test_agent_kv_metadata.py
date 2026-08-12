# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.generate.base.agent_kv import (
    get_agent_kv_request_metadata,
)


def parse_metadata(
    *,
    session_id: str | None = None,
    headers: dict[str, str] | None = None,
    xargs: dict[str, object] | None = None,
):
    return get_agent_kv_request_metadata(
        session_id=session_id,
        headers=headers,
        xargs=xargs,
    )


def test_agent_kv_metadata_is_opt_in() -> None:
    assert parse_metadata(session_id="session-a") is None


def test_agent_kv_metadata_accepts_request_headers() -> None:
    metadata = parse_metadata(
        session_id="session-a",
        headers={
            "X-Session-ID": "session-a",
            "X-AgentKV-Namespace": "tenant-a",
            "X-AgentKV-Generation": "7",
            "X-AgentKV-Branch-ID": "research",
        },
    )

    assert metadata is not None
    assert metadata.namespace == "tenant-a"
    assert metadata.session_id == "session-a"
    assert metadata.generation == 7
    assert metadata.branch_id == "research"


def test_agent_kv_metadata_accepts_vllm_xargs() -> None:
    metadata = parse_metadata(
        session_id="session-a",
        xargs={
            "agent_kv_namespace": "tenant-a",
            "agent_kv_generation": 7,
        },
    )

    assert metadata is not None
    assert metadata.branch_id == "main"
    assert metadata.generation == 7


@pytest.mark.parametrize(
    "headers,match",
    [
        (
            {
                "X-AgentKV-Namespace": "tenant-a",
                "X-AgentKV-Generation": "7",
            },
            "session_id",
        ),
        (
            {
                "X-Session-ID": "session-a",
                "X-AgentKV-Generation": "7",
            },
            "Namespace",
        ),
        (
            {
                "X-Session-ID": "session-a",
                "X-AgentKV-Namespace": "tenant-a",
                "X-AgentKV-Generation": "seven",
            },
            "positive integer",
        ),
    ],
)
def test_agent_kv_metadata_rejects_partial_or_invalid_input(
    headers: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_metadata(
            session_id=headers.get("X-Session-ID"),
            headers=headers,
        )


def test_headers_override_vllm_xargs() -> None:
    metadata = parse_metadata(
        session_id="session-a",
        headers={
            "X-AgentKV-Namespace": "trusted-tenant",
            "X-AgentKV-Generation": "8",
        },
        xargs={
            "agent_kv_namespace": "untrusted-tenant",
            "agent_kv_generation": 7,
        },
    )

    assert metadata is not None
    assert metadata.namespace == "trusted-tenant"
    assert metadata.generation == 8
