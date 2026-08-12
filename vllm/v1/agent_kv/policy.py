# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure eviction planning for AgentKV-owned cache blocks."""

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.agent_kv.ownership import AgentKVGenerationKey
from vllm.v1.agent_kv.protocol import AgentKVLifecycleState


class AgentKVEvictionClass(enum.IntEnum):
    """Coarse eviction treatment for diagnostics and policy tests."""

    EVICT_FIRST = 0
    DEFAULT = 1
    RETAIN = 2


@dataclass(frozen=True, slots=True)
class AgentKVPolicyOwner:
    """Lifecycle facts used to score one cache owner."""

    key: AgentKVGenerationKey
    state: AgentKVLifecycleState
    priority: int = 0
    retain_until_ms: int | None = None
    resume_due_at_ms: int | None = None
    last_updated_ms: int = 0


@dataclass(frozen=True, slots=True)
class AgentKVCacheBlockCandidate:
    """Immutable cache snapshot consumed by the policy."""

    block_id: int
    cache_keys: tuple[bytes, ...]
    content_hashes: tuple[bytes, ...]
    owners: tuple[AgentKVPolicyOwner, ...]


@dataclass(frozen=True, slots=True)
class AgentKVEvictionTarget:
    """Compare-and-validate target returned to the block allocator."""

    block_id: int
    expected_cache_keys: tuple[bytes, ...]
    eviction_class: AgentKVEvictionClass


_STATE_RANK = {
    AgentKVLifecycleState.TERMINATED: 0,
    AgentKVLifecycleState.EXPIRED: 0,
    AgentKVLifecycleState.SUSPENDED: 2,
    AgentKVLifecycleState.ACTIVE: 3,
    AgentKVLifecycleState.RESUME_PENDING: 4,
}
_NO_RESUME_URGENCY = -(2**63)


def _owner_score(owner: AgentKVPolicyOwner, now_ms: int) -> tuple[int, ...]:
    retain_active = int(
        owner.retain_until_ms is not None and now_ms < owner.retain_until_ms
    )
    resume_urgency = (
        -max(owner.resume_due_at_ms - now_ms, 0)
        if owner.resume_due_at_ms is not None
        else _NO_RESUME_URGENCY
    )
    return (
        _STATE_RANK[owner.state],
        retain_active,
        owner.priority,
        resume_urgency,
        owner.last_updated_ms,
    )


def _candidate_score(
    candidate: AgentKVCacheBlockCandidate,
    now_ms: int,
) -> tuple[int, ...]:
    if not candidate.owners:
        return (1, 0, 0, _NO_RESUME_URGENCY, 0)
    return max(_owner_score(owner, now_ms) for owner in candidate.owners)


def _eviction_class(score: tuple[int, ...]) -> AgentKVEvictionClass:
    if score[0] == 0:
        return AgentKVEvictionClass.EVICT_FIRST
    if score[0] == 1:
        return AgentKVEvictionClass.DEFAULT
    return AgentKVEvictionClass.RETAIN


def plan_agent_kv_evictions(
    candidates: Sequence[AgentKVCacheBlockCandidate],
    num_at_risk_blocks: int,
    now_ms: int,
) -> tuple[AgentKVEvictionTarget, ...]:
    """Choose the immediate victims while preserving native order on ties.

    Candidate order is the allocator's existing eviction order. Sorting is
    stable, so lifecycle-equivalent blocks retain native LRU behavior.
    """
    if num_at_risk_blocks <= 0 or not candidates:
        return ()

    scored = [
        (_candidate_score(candidate, now_ms), candidate) for candidate in candidates
    ]
    selected = sorted(scored, key=lambda item: item[0])[:num_at_risk_blocks]
    native_victims = candidates[:num_at_risk_blocks]
    if [candidate.block_id for _, candidate in selected] == [
        candidate.block_id for candidate in native_victims
    ]:
        return ()

    return tuple(
        AgentKVEvictionTarget(
            block_id=candidate.block_id,
            expected_cache_keys=candidate.cache_keys,
            eviction_class=_eviction_class(score),
        )
        for score, candidate in selected
    )
