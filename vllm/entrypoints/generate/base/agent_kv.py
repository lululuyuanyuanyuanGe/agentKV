# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AgentKV request identity parsing shared by online serving endpoints."""

from collections.abc import Mapping

import vllm.envs as envs
from vllm.v1.agent_kv.protocol import (
    AgentKVRequestMetadata,
    parse_agent_kv_request_metadata,
)

AGENT_KV_NAMESPACE_HEADER = "X-AgentKV-Namespace"
AGENT_KV_GENERATION_HEADER = "X-AgentKV-Generation"
AGENT_KV_BRANCH_ID_HEADER = "X-AgentKV-Branch-ID"
AGENT_KV_PROTOCOL_VERSION_HEADER = "X-AgentKV-Protocol-Version"


def get_agent_kv_request_metadata(
    *,
    session_id: str | None,
    headers: Mapping[str, str] | None,
    xargs: Mapping[str, object] | None,
) -> AgentKVRequestMetadata | None:
    """Return validated opt-in metadata or ``None`` for native requests.

    Headers take precedence so a trusted gateway can enforce namespace
    isolation. ``vllm_xargs`` remains available for non-HTTP callers.
    """
    xargs = xargs or {}

    def get_value(header: str, xarg: str) -> object:
        value = headers.get(header) if headers is not None else None
        return value if value is not None else xargs.get(xarg)

    namespace = get_value(AGENT_KV_NAMESPACE_HEADER, "agent_kv_namespace")
    generation = get_value(AGENT_KV_GENERATION_HEADER, "agent_kv_generation")
    branch_id = get_value(AGENT_KV_BRANCH_ID_HEADER, "agent_kv_branch_id")
    protocol_version = get_value(
        AGENT_KV_PROTOCOL_VERSION_HEADER, "agent_kv_protocol_version"
    )

    if all(
        value is None for value in (namespace, generation, branch_id, protocol_version)
    ):
        return None

    if not envs.VLLM_ENABLE_AGENT_KV:
        raise ValueError("AgentKV request metadata requires VLLM_ENABLE_AGENT_KV=1")

    return parse_agent_kv_request_metadata(
        session_id=session_id,
        namespace=namespace,
        generation=generation,
        branch_id=branch_id,
        protocol_version=protocol_version,
    )
