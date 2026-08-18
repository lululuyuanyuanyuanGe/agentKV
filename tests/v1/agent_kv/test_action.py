# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.agent_kv.action import (
    AgentKVCacheActionType,
    AgentKVPressure,
    AgentKVPressureLevel,
    plan_agent_kv_cache_actions,
)
from vllm.v1.agent_kv.ownership import AgentKVGenerationKey
from vllm.v1.agent_kv.policy import (
    AgentKVCacheBlockCandidate,
    AgentKVPolicyOwner,
)
from vllm.v1.agent_kv.protocol import AgentKVLifecycleState


def make_owner(
    session_id: str,
    state: AgentKVLifecycleState,
    retain_until_ms: int | None = None,
) -> AgentKVPolicyOwner:
    return AgentKVPolicyOwner(
        key=AgentKVGenerationKey("tenant", session_id, "main", 1),
        state=state,
        retain_until_ms=retain_until_ms,
    )


def make_candidate(
    *owners: AgentKVPolicyOwner,
) -> AgentKVCacheBlockCandidate:
    return AgentKVCacheBlockCandidate(
        block_id=7,
        cache_keys=(b"cache-key",),
        content_hashes=(b"content-hash",),
        owners=owners,
    )


def make_pressure() -> AgentKVPressure:
    return AgentKVPressure(
        level=AgentKVPressureLevel.SOFT,
        target_blocks=4,
        available_offload_blocks=2,
    )


def plan_action(
    candidate: AgentKVCacheBlockCandidate,
    now_ms: int = 1_000,
):
    return plan_agent_kv_cache_actions(
        [candidate],
        make_pressure(),
        now_ms=now_ms,
        policy_revision=3,
    )


def test_unowned_block_preserves_native_behavior() -> None:
    plan = plan_action(make_candidate())

    assert plan.actions[0].action == AgentKVCacheActionType.DEFAULT


def test_suspended_block_is_offloaded() -> None:
    plan = plan_action(
        make_candidate(make_owner("suspended", AgentKVLifecycleState.SUSPENDED))
    )

    assert plan.actions[0].action == AgentKVCacheActionType.OFFLOAD


def test_ended_block_is_dropped_without_a_lower_tier_copy() -> None:
    plan = plan_action(
        make_candidate(
            make_owner("expired", AgentKVLifecycleState.EXPIRED),
            make_owner("terminated", AgentKVLifecycleState.TERMINATED),
        )
    )

    assert plan.actions[0].action == AgentKVCacheActionType.DROP


def test_shared_block_uses_most_protective_owner() -> None:
    plan = plan_action(
        make_candidate(
            make_owner("expired", AgentKVLifecycleState.EXPIRED),
            make_owner("active", AgentKVLifecycleState.ACTIVE),
        )
    )

    assert plan.actions[0].action == AgentKVCacheActionType.KEEP


def test_retain_window_keeps_suspended_block_until_deadline() -> None:
    candidate = make_candidate(
        make_owner(
            "retained",
            AgentKVLifecycleState.SUSPENDED,
            retain_until_ms=2_000,
        )
    )

    retained = plan_action(candidate, now_ms=1_000)
    expired = plan_action(candidate, now_ms=2_000)

    assert retained.actions[0].action == AgentKVCacheActionType.KEEP
    assert retained.reconsider_at_ms == 2_000
    assert expired.actions[0].action == AgentKVCacheActionType.OFFLOAD
    assert expired.reconsider_at_ms is None
