# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure cache-tier action planning for AgentKV-owned blocks."""

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm.v1.agent_kv.ownership import AgentKVGenerationKey
from vllm.v1.agent_kv.policy import (
    AgentKVCacheBlockCandidate,
    AgentKVPolicyOwner,
)
from vllm.v1.agent_kv.protocol import AgentKVLifecycleState


class AgentKVCacheActionType(enum.Enum):
    """Action requested for a cached block under the current pressure."""

    DEFAULT = "default"
    KEEP = "keep"
    OFFLOAD = "offload"
    DROP = "drop"


class AgentKVPressureLevel(enum.Enum):
    """Allocator pressure that caused cache action planning."""

    SOFT = "soft"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class AgentKVPressure:
    """Capacity facts available to an action planner."""

    level: AgentKVPressureLevel
    target_blocks: int
    available_offload_blocks: int = 0


@dataclass(frozen=True, slots=True)
class AgentKVCacheAction:
    """Compare-and-validate action for one immutable cache snapshot."""

    policy_revision: int
    block_id: int
    expected_cache_keys: tuple[bytes, ...]
    content_hashes: tuple[bytes, ...]
    owner_keys: tuple[AgentKVGenerationKey, ...]
    action: AgentKVCacheActionType
    reason: str


@dataclass(frozen=True, slots=True)
class AgentKVActionPlan:
    """Actions produced from one revision of controller state."""

    policy_revision: int
    pressure: AgentKVPressure
    actions: tuple[AgentKVCacheAction, ...]
    reconsider_at_ms: int | None = None


class AgentKVCacheSnapshot(Protocol):
    """Content-only snapshot accepted from a cache implementation."""

    block_id: int
    cache_keys: tuple[bytes, ...]
    content_hashes: tuple[bytes, ...]


AgentKVActionPlanner = Callable[
    [Sequence[AgentKVCacheSnapshot], AgentKVPressure],
    AgentKVActionPlan,
]
AgentKVActionValidator = Callable[[AgentKVCacheAction], bool]
AgentKVActionPolicyEnabled = Callable[[], bool]


def _owner_key(key: AgentKVGenerationKey) -> tuple[str, str, str, int]:
    return key.namespace, key.session_id, key.branch_id, key.generation


def _owner_action(
    owner: AgentKVPolicyOwner,
    now_ms: int,
) -> tuple[AgentKVCacheActionType, str, int | None]:
    if owner.state in (
        AgentKVLifecycleState.ACTIVE,
        AgentKVLifecycleState.RESUME_PENDING,
    ):
        return AgentKVCacheActionType.KEEP, owner.state.value, None
    if owner.state == AgentKVLifecycleState.SUSPENDED:
        if owner.retain_until_ms is not None and now_ms < owner.retain_until_ms:
            return (
                AgentKVCacheActionType.KEEP,
                "suspended-retain-window",
                owner.retain_until_ms,
            )
        return AgentKVCacheActionType.OFFLOAD, "suspended", None
    return AgentKVCacheActionType.DROP, owner.state.value, None


def _candidate_action(
    candidate: AgentKVCacheBlockCandidate,
    now_ms: int,
) -> tuple[AgentKVCacheActionType, str, int | None]:
    if not candidate.owners:
        return AgentKVCacheActionType.DEFAULT, "unowned", None

    owner_actions = [_owner_action(owner, now_ms) for owner in candidate.owners]
    for action_type in (
        AgentKVCacheActionType.KEEP,
        AgentKVCacheActionType.OFFLOAD,
        AgentKVCacheActionType.DROP,
    ):
        matching = [item for item in owner_actions if item[0] == action_type]
        if matching:
            reconsider_at_ms = min(
                (item[2] for item in matching if item[2] is not None),
                default=None,
            )
            return action_type, matching[0][1], reconsider_at_ms
    raise AssertionError("AgentKV owner action must be defined")


def plan_agent_kv_cache_actions(
    candidates: Sequence[AgentKVCacheBlockCandidate],
    pressure: AgentKVPressure,
    now_ms: int,
    policy_revision: int,
) -> AgentKVActionPlan:
    """Plan deterministic tier actions using the most protective owner.

    ``DEFAULT`` preserves the cache implementation's native behavior. ``DROP``
    means that a lower-tier copy should not be created; physical eviction is
    still owned by the allocator.
    """
    actions: list[AgentKVCacheAction] = []
    reconsider_at_ms: int | None = None
    for candidate in candidates:
        action_type, reason, candidate_reconsider_at_ms = _candidate_action(
            candidate, now_ms
        )
        if candidate_reconsider_at_ms is not None:
            reconsider_at_ms = (
                candidate_reconsider_at_ms
                if reconsider_at_ms is None
                else min(reconsider_at_ms, candidate_reconsider_at_ms)
            )
        actions.append(
            AgentKVCacheAction(
                policy_revision=policy_revision,
                block_id=candidate.block_id,
                expected_cache_keys=candidate.cache_keys,
                content_hashes=candidate.content_hashes,
                owner_keys=tuple(
                    sorted((owner.key for owner in candidate.owners), key=_owner_key)
                ),
                action=action_type,
                reason=reason,
            )
        )
    return AgentKVActionPlan(
        policy_revision=policy_revision,
        pressure=pressure,
        actions=tuple(actions),
        reconsider_at_ms=reconsider_at_ms,
    )
