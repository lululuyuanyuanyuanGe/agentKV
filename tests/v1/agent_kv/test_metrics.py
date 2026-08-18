# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.agent_kv.action import (
    AgentKVCacheAction,
    AgentKVCacheActionType,
)
from vllm.v1.agent_kv.metrics import AgentKVMetrics
from vllm.v1.agent_kv.protocol import (
    AgentKVEventStatus,
    AgentKVEventType,
)


def test_metrics_drain_resets_deltas_and_retains_gauges() -> None:
    metrics = AgentKVMetrics()
    action = AgentKVCacheAction(
        policy_revision=3,
        block_id=7,
        expected_cache_keys=(b"cache-key",),
        content_hashes=(b"content-hash",),
        owner_keys=(),
        action=AgentKVCacheActionType.OFFLOAD,
        reason="suspended",
    )
    metrics.record_event(
        AgentKVEventType.SUSPEND,
        AgentKVEventStatus.ACCEPTED,
    )
    metrics.record_actions([action])
    metrics.record_offload_result("accepted", 2)
    metrics.record_fallback("planner_probe_error")
    metrics.record_cursor_reset("policy_revision")
    metrics.record_policy_revision()
    metrics.set_inflight_store_blocks(4)

    stats = metrics.drain(
        num_sessions=2,
        num_generations=3,
        num_owned_hashes=5,
    )

    assert stats.event_counts == {"SUSPEND": {"accepted": 1}}
    assert stats.action_counts == {"offload": {"suspended": 1}}
    assert stats.offload_result_counts == {"accepted": 2}
    assert stats.fallback_counts == {"planner_probe_error": 1}
    assert stats.cursor_reset_counts == {"policy_revision": 1}
    assert stats.policy_revisions == 1
    assert stats.num_sessions == 2
    assert stats.num_generations == 3
    assert stats.num_owned_hashes == 5
    assert stats.num_inflight_store_blocks == 4

    next_stats = metrics.drain(
        num_sessions=2,
        num_generations=3,
        num_owned_hashes=5,
    )

    assert next_stats.event_counts == {}
    assert next_stats.action_counts == {}
    assert next_stats.offload_result_counts == {}
    assert next_stats.fallback_counts == {}
    assert next_stats.cursor_reset_counts == {}
    assert next_stats.policy_revisions == 0
    assert next_stats.num_inflight_store_blocks == 4
