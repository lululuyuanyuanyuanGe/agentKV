# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.agent_kv.ownership import AgentKVGenerationKey
from vllm.v1.agent_kv.policy import (
    AgentKVCacheBlockCandidate,
    AgentKVEvictionClass,
    AgentKVPolicyOwner,
    plan_agent_kv_evictions,
)
from vllm.v1.agent_kv.protocol import AgentKVLifecycleState


def make_candidate(
    block_id: int,
    *states: AgentKVLifecycleState,
) -> AgentKVCacheBlockCandidate:
    owners = tuple(
        AgentKVPolicyOwner(
            key=AgentKVGenerationKey("tenant", f"session-{block_id}-{i}", "main", 1),
            state=state,
        )
        for i, state in enumerate(states)
    )
    return AgentKVCacheBlockCandidate(
        block_id=block_id,
        cache_keys=(f"key-{block_id}".encode(),),
        content_hashes=(f"hash-{block_id}".encode(),),
        owners=owners,
    )


def test_expired_block_is_selected_before_native_lru_resume_block() -> None:
    candidates = [
        make_candidate(1, AgentKVLifecycleState.RESUME_PENDING),
        make_candidate(2, AgentKVLifecycleState.EXPIRED),
    ]

    plan = plan_agent_kv_evictions(candidates, num_at_risk_blocks=1, now_ms=1_000)

    assert [target.block_id for target in plan] == [2]
    assert plan[0].eviction_class == AgentKVEvictionClass.EVICT_FIRST


def test_shared_block_uses_most_protective_owner() -> None:
    candidates = [
        make_candidate(
            1,
            AgentKVLifecycleState.EXPIRED,
            AgentKVLifecycleState.RESUME_PENDING,
        ),
        make_candidate(2, AgentKVLifecycleState.SUSPENDED),
    ]

    plan = plan_agent_kv_evictions(candidates, num_at_risk_blocks=1, now_ms=1_000)

    assert [target.block_id for target in plan] == [2]


def test_equivalent_candidates_preserve_native_lru_without_a_plan() -> None:
    candidates = [
        make_candidate(1, AgentKVLifecycleState.SUSPENDED),
        make_candidate(2, AgentKVLifecycleState.SUSPENDED),
    ]

    plan = plan_agent_kv_evictions(candidates, num_at_risk_blocks=1, now_ms=1_000)

    assert not plan
