# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded scheduler-side telemetry for AgentKV."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.agent_kv.action import AgentKVCacheAction
    from vllm.v1.agent_kv.protocol import AgentKVEventStatus, AgentKVEventType


@dataclass
class AgentKVStats:
    """Serializable AgentKV deltas and current aggregate sizes."""

    event_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    action_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    offload_result_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: dict[str, int] = field(default_factory=dict)
    cursor_reset_counts: dict[str, int] = field(default_factory=dict)
    policy_revisions: int = 0
    num_sessions: int = 0
    num_generations: int = 0
    num_owned_hashes: int = 0
    num_inflight_store_blocks: int = 0


class AgentKVMetrics:
    """Accumulates only fixed-cardinality engine-owned labels."""

    def __init__(self) -> None:
        self._event_counts: Counter[tuple[str, str]] = Counter()
        self._action_counts: Counter[tuple[str, str]] = Counter()
        self._offload_result_counts: Counter[str] = Counter()
        self._fallback_counts: Counter[str] = Counter()
        self._cursor_reset_counts: Counter[str] = Counter()
        self._policy_revisions = 0
        self._num_inflight_store_blocks = 0

    def record_event(
        self,
        event_type: AgentKVEventType,
        status: AgentKVEventStatus,
    ) -> None:
        self._event_counts[event_type.value, status.value] += 1

    def record_actions(self, actions: Sequence[AgentKVCacheAction]) -> None:
        for action in actions:
            self._action_counts[action.action.value, action.reason] += 1

    def record_offload_result(self, result: str, count: int) -> None:
        if count > 0:
            self._offload_result_counts[result] += count

    def record_fallback(self, reason: str) -> None:
        self._fallback_counts[reason] += 1

    def record_cursor_reset(self, reason: str) -> None:
        self._cursor_reset_counts[reason] += 1

    def record_policy_revision(self) -> None:
        self._policy_revisions += 1

    def set_inflight_store_blocks(self, count: int) -> None:
        self._num_inflight_store_blocks = max(count, 0)

    @staticmethod
    def _nested_counts(
        counts: Counter[tuple[str, str]],
    ) -> dict[str, dict[str, int]]:
        nested: dict[str, dict[str, int]] = {}
        for (primary, secondary), count in counts.items():
            nested.setdefault(primary, {})[secondary] = count
        return nested

    def drain(
        self,
        *,
        num_sessions: int,
        num_generations: int,
        num_owned_hashes: int,
    ) -> AgentKVStats:
        """Return interval deltas and reset counters while retaining gauges."""
        stats = AgentKVStats(
            event_counts=self._nested_counts(self._event_counts),
            action_counts=self._nested_counts(self._action_counts),
            offload_result_counts=dict(self._offload_result_counts),
            fallback_counts=dict(self._fallback_counts),
            cursor_reset_counts=dict(self._cursor_reset_counts),
            policy_revisions=self._policy_revisions,
            num_sessions=num_sessions,
            num_generations=num_generations,
            num_owned_hashes=num_owned_hashes,
            num_inflight_store_blocks=self._num_inflight_store_blocks,
        )
        self._event_counts.clear()
        self._action_counts.clear()
        self._offload_result_counts.clear()
        self._fallback_counts.clear()
        self._cursor_reset_counts.clear()
        self._policy_revisions = 0
        return stats
