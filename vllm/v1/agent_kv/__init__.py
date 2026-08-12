# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Application-aware KV cache lifecycle tracking."""

from vllm.v1.agent_kv.controller import AgentKVController
from vllm.v1.agent_kv.ownership import AgentKVGenerationKey
from vllm.v1.agent_kv.policy import (
    AgentKVCacheBlockCandidate,
    AgentKVEvictionClass,
    AgentKVEvictionTarget,
    AgentKVPolicyOwner,
    plan_agent_kv_evictions,
)
from vllm.v1.agent_kv.protocol import (
    AGENT_KV_PROTOCOL_VERSION,
    AgentKVEvent,
    AgentKVEventHints,
    AgentKVEventStatus,
    AgentKVEventType,
    AgentKVLifecycleState,
    AgentKVRequestMetadata,
    parse_agent_kv_request_metadata,
)

__all__ = [
    "AGENT_KV_PROTOCOL_VERSION",
    "AgentKVController",
    "AgentKVCacheBlockCandidate",
    "AgentKVEvictionClass",
    "AgentKVEvictionTarget",
    "AgentKVEvent",
    "AgentKVEventHints",
    "AgentKVEventStatus",
    "AgentKVEventType",
    "AgentKVGenerationKey",
    "AgentKVLifecycleState",
    "AgentKVRequestMetadata",
    "AgentKVPolicyOwner",
    "plan_agent_kv_evictions",
    "parse_agent_kv_request_metadata",
]
