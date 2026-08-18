# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side lifecycle state and cache ownership tracking for AgentKV."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from vllm.v1.agent_kv.action import (
    AgentKVActionPlan,
    AgentKVCacheAction,
    AgentKVCacheActionType,
    AgentKVCacheSnapshot,
    AgentKVPressure,
    AgentKVPressureLevel,
    plan_agent_kv_cache_actions,
)
from vllm.v1.agent_kv.metrics import AgentKVMetrics, AgentKVStats
from vllm.v1.agent_kv.ownership import (
    AgentKVGenerationKey,
    AgentKVOwnershipIndex,
)
from vllm.v1.agent_kv.policy import (
    AgentKVCacheBlockCandidate,
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
)

_MAX_GENERATIONS_PER_BRANCH = 16
_MAX_SEEN_EVENT_IDS_PER_SESSION = 256
_DEFAULT_MAX_SESSIONS = 100_000


@dataclass(slots=True)
class _GenerationRecord:
    state: AgentKVLifecycleState = AgentKVLifecycleState.ACTIVE
    active_request_ids: set[str] = field(default_factory=set)
    finished_request_count: int = 0
    block_hashes: set[bytes] = field(default_factory=set)
    latest_hints: AgentKVEventHints | None = None
    last_event_type: AgentKVEventType | None = None
    last_event_emitted_at_ms: int | None = None
    reason: str | None = None
    last_updated_ms: int = 0


@dataclass(slots=True)
class _BranchRecord:
    current_generation: int = 0
    generations: OrderedDict[int, _GenerationRecord] = field(
        default_factory=OrderedDict
    )


@dataclass(slots=True)
class _SessionRecord:
    branches: dict[str, _BranchRecord] = field(default_factory=dict)
    last_event_sequence: int = 0
    seen_event_ids: set[str] = field(default_factory=set)
    seen_event_order: deque[str] = field(default_factory=deque)
    terminated: bool = False
    last_updated_ms: int = 0


@dataclass(frozen=True, slots=True)
class _ActionSnapshot:
    block_id: int
    cache_keys: tuple[bytes, ...]
    content_hashes: tuple[bytes, ...]

    @classmethod
    def from_action(cls, action: AgentKVCacheAction) -> _ActionSnapshot:
        return cls(
            block_id=action.block_id,
            cache_keys=action.expected_cache_keys,
            content_hashes=action.content_hashes,
        )


class AgentKVController:
    """Tracks upstream facts and plans content-addressed cache ordering.

    The controller deliberately stores content hashes instead of physical block
    identifiers. Block identifiers are allocator-owned and may be reused after
    a request finishes.
    """

    def __init__(
        self,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        *,
        enabled: bool = True,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be greater than or equal to 1")
        self.max_sessions = max_sessions
        self.enabled = enabled
        self._sessions: OrderedDict[tuple[str, str], _SessionRecord] = OrderedDict()
        self._request_index: dict[str, tuple[tuple[str, str], str, int]] = {}
        self._ownership = AgentKVOwnershipIndex()
        self._policy_revision = 0
        self.metrics = AgentKVMetrics()

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _bump_policy_revision(self) -> None:
        self._policy_revision += 1
        self.metrics.record_policy_revision()

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ValueError(
                "AgentKV is disabled; set VLLM_ENABLE_AGENT_KV=1 before startup"
            )

    def _get_or_create_session(
        self, namespace: str, session_id: str
    ) -> tuple[tuple[str, str], _SessionRecord]:
        key = (namespace, session_id)
        session = self._sessions.get(key)
        if session is None:
            while len(self._sessions) >= self.max_sessions:
                evicted_key, _ = self._sessions.popitem(last=False)
                self._remove_request_index_for_session(evicted_key)
                self._ownership.remove_session(*evicted_key)
            session = _SessionRecord()
            self._sessions[key] = session
        else:
            self._sessions.move_to_end(key)
        return key, session

    def _remove_request_index_for_session(self, key: tuple[str, str]) -> None:
        stale_request_ids = [
            request_id
            for request_id, (session_key, _, _) in self._request_index.items()
            if session_key == key
        ]
        for request_id in stale_request_ids:
            del self._request_index[request_id]

    @staticmethod
    def _get_or_create_generation(
        session: _SessionRecord,
        branch_id: str,
        generation: int,
    ) -> tuple[_BranchRecord, _GenerationRecord]:
        branch = session.branches.setdefault(branch_id, _BranchRecord())
        record = branch.generations.get(generation)
        if record is None:
            record = _GenerationRecord()
            branch.generations[generation] = record
        else:
            branch.generations.move_to_end(generation)
        return branch, record

    def _prune_branch_generations(
        self,
        session_key: tuple[str, str],
        branch_id: str,
        branch: _BranchRecord,
    ) -> None:
        while len(branch.generations) > _MAX_GENERATIONS_PER_BRANCH:
            generation, record = next(iter(branch.generations.items()))
            if record.active_request_ids:
                break
            del branch.generations[generation]
            self._ownership.remove_owner(
                AgentKVGenerationKey(
                    namespace=session_key[0],
                    session_id=session_key[1],
                    branch_id=branch_id,
                    generation=generation,
                )
            )

    @staticmethod
    def _remember_event(session: _SessionRecord, event_id: str) -> None:
        session.seen_event_ids.add(event_id)
        session.seen_event_order.append(event_id)
        while len(session.seen_event_order) > _MAX_SEEN_EVENT_IDS_PER_SESSION:
            expired_id = session.seen_event_order.popleft()
            session.seen_event_ids.discard(expired_id)

    def register_request(
        self, metadata: AgentKVRequestMetadata, request_id: str
    ) -> None:
        """Register one inference request as the active session generation."""
        self._require_enabled()
        key, session = self._get_or_create_session(
            metadata.namespace, metadata.session_id
        )
        branch, generation = self._get_or_create_generation(
            session, metadata.branch_id, metadata.generation
        )
        now_ms = self._now_ms()
        generation.active_request_ids.add(request_id)
        generation.last_updated_ms = now_ms
        if session.terminated:
            generation.state = AgentKVLifecycleState.TERMINATED
        elif metadata.generation >= branch.current_generation:
            branch.current_generation = metadata.generation
            generation.state = AgentKVLifecycleState.ACTIVE
        session.last_updated_ms = now_ms
        self._request_index[request_id] = (
            key,
            metadata.branch_id,
            metadata.generation,
        )
        self._prune_branch_generations(key, metadata.branch_id, branch)
        self._bump_policy_revision()

    def finish_request(self, request_id: str, block_hashes: Iterable[bytes]) -> None:
        """Persist the content identity before scheduler request state is freed."""
        if not self.enabled:
            return
        request_ref = self._request_index.pop(request_id, None)
        if request_ref is None:
            return
        key, branch_id, generation_id = request_ref
        session = self._sessions.get(key)
        if session is None:
            return
        branch = session.branches.get(branch_id)
        if branch is None:
            return
        generation = branch.generations.get(generation_id)
        if generation is None:
            return
        generation.active_request_ids.discard(request_id)
        generation.finished_request_count += 1
        hashes = {bytes(block_hash) for block_hash in block_hashes}
        generation.block_hashes.update(hashes)
        self._ownership.add(
            AgentKVGenerationKey(
                namespace=key[0],
                session_id=key[1],
                branch_id=branch_id,
                generation=generation_id,
            ),
            hashes,
        )
        now_ms = self._now_ms()
        generation.last_updated_ms = now_ms
        session.last_updated_ms = now_ms
        self._sessions.move_to_end(key)
        self._prune_branch_generations(key, branch_id, branch)
        self._bump_policy_revision()

    def apply_event(self, event: AgentKVEvent) -> dict[str, object]:
        """Apply one idempotent, ordered lifecycle event."""
        self._require_enabled()
        key, session = self._get_or_create_session(event.namespace, event.session_id)

        if event.event_id in session.seen_event_ids:
            return self._result(AgentKVEventStatus.DUPLICATE, session, event)

        self._remember_event(session, event.event_id)
        if event.event_sequence <= session.last_event_sequence:
            return self._result(AgentKVEventStatus.STALE_EVENT, session, event)

        session.last_event_sequence = event.event_sequence
        session.last_updated_ms = self._now_ms()

        if session.terminated:
            return self._result(AgentKVEventStatus.SESSION_TERMINATED, session, event)

        if event.event_type == AgentKVEventType.SESSION_TERMINATED:
            session.terminated = True
            for branch in session.branches.values():
                for generation in branch.generations.values():
                    generation.state = AgentKVLifecycleState.TERMINATED
            self._bump_policy_revision()
            return self._result(AgentKVEventStatus.ACCEPTED, session, event)

        branch, generation = self._get_or_create_generation(
            session, event.branch_id, event.generation
        )

        if event.event_type == AgentKVEventType.GENERATION_EXPIRED:
            self._record_event_details(generation, event)
            generation.state = AgentKVLifecycleState.EXPIRED
            generation.last_updated_ms = session.last_updated_ms
            if event.generation > branch.current_generation:
                branch.current_generation = event.generation
            self._prune_branch_generations(key, event.branch_id, branch)
            self._bump_policy_revision()
            return self._result(AgentKVEventStatus.ACCEPTED, session, event)

        if event.generation < branch.current_generation:
            self._prune_branch_generations(key, event.branch_id, branch)
            return self._result(AgentKVEventStatus.STALE_GENERATION, session, event)

        if event.generation > branch.current_generation:
            branch.current_generation = event.generation

        self._record_event_details(generation, event)
        generation.state = (
            AgentKVLifecycleState.SUSPENDED
            if event.event_type == AgentKVEventType.SUSPEND
            else AgentKVLifecycleState.RESUME_PENDING
        )
        generation.last_updated_ms = session.last_updated_ms
        self._prune_branch_generations(key, event.branch_id, branch)
        self._bump_policy_revision()
        return self._result(AgentKVEventStatus.ACCEPTED, session, event)

    def has_cache_owners(self) -> bool:
        """Return whether policy evaluation could affect cache ordering."""
        return self.enabled and bool(self._ownership)

    def drain_metrics(self) -> AgentKVStats | None:
        """Return interval telemetry without exposing lifecycle identities."""
        if not self.enabled:
            return None
        num_generations = sum(
            len(branch.generations)
            for session in self._sessions.values()
            for branch in session.branches.values()
        )
        return self.metrics.drain(
            num_sessions=len(self._sessions),
            num_generations=num_generations,
            num_owned_hashes=self._ownership.num_hashes,
        )

    def get_block_owners(self, block_hash: bytes) -> frozenset[AgentKVGenerationKey]:
        """Return generation identities associated with a content hash."""
        return self._ownership.get_owners(block_hash)

    def plan_evictions(
        self,
        candidates: Sequence[AgentKVCacheSnapshot],
        num_at_risk_blocks: int,
    ) -> tuple[AgentKVEvictionTarget, ...]:
        """Attach current lifecycle facts and invoke the pure policy."""
        now_ms = self._now_ms()
        policy_candidates = self._build_policy_candidates(candidates)
        return plan_agent_kv_evictions(
            policy_candidates,
            num_at_risk_blocks,
            now_ms,
        )

    def plan_cache_actions(
        self,
        candidates: Sequence[AgentKVCacheSnapshot],
        pressure: AgentKVPressure,
    ) -> AgentKVActionPlan:
        """Plan lower-tier actions from a consistent lifecycle revision."""
        return plan_agent_kv_cache_actions(
            self._build_policy_candidates(candidates),
            pressure,
            self._now_ms(),
            self._policy_revision,
        )

    def validate_cache_action(self, action: AgentKVCacheAction) -> bool:
        """Revalidate an asynchronous store before publishing its result."""
        if action.action not in (
            AgentKVCacheActionType.DEFAULT,
            AgentKVCacheActionType.OFFLOAD,
        ):
            return False
        pressure = AgentKVPressure(
            level=AgentKVPressureLevel.SOFT,
            target_blocks=1,
            available_offload_blocks=1,
        )
        current = self.plan_cache_actions(
            [_ActionSnapshot.from_action(action)], pressure
        ).actions[0]
        return (
            current.action == action.action
            and current.owner_keys == action.owner_keys
            and current.expected_cache_keys == action.expected_cache_keys
            and current.content_hashes == action.content_hashes
        )

    def _build_policy_candidates(
        self,
        candidates: Sequence[AgentKVCacheSnapshot],
    ) -> list[AgentKVCacheBlockCandidate]:
        policy_candidates: list[AgentKVCacheBlockCandidate] = []
        for candidate in candidates:
            cache_keys = tuple(candidate.cache_keys)
            content_hashes = tuple(candidate.content_hashes)
            owner_keys: set[AgentKVGenerationKey] = set()
            for content_hash in content_hashes:
                owner_keys.update(self._ownership.get_owners(content_hash))
            owners = tuple(
                owner
                for owner_key in sorted(
                    owner_keys,
                    key=lambda key: (
                        key.namespace,
                        key.session_id,
                        key.branch_id,
                        key.generation,
                    ),
                )
                if (owner := self._get_policy_owner(owner_key)) is not None
            )
            policy_candidates.append(
                AgentKVCacheBlockCandidate(
                    block_id=candidate.block_id,
                    cache_keys=cache_keys,
                    content_hashes=content_hashes,
                    owners=owners,
                )
            )
        return policy_candidates

    def _get_policy_owner(self, key: AgentKVGenerationKey) -> AgentKVPolicyOwner | None:
        session = self._sessions.get((key.namespace, key.session_id))
        branch = session.branches.get(key.branch_id) if session is not None else None
        generation = (
            branch.generations.get(key.generation) if branch is not None else None
        )
        if generation is None:
            return None
        hints = generation.latest_hints
        retain_until_ms = (
            generation.last_updated_ms + hints.retain_for_ms
            if hints is not None and hints.retain_for_ms is not None
            else None
        )
        resume_due_at_ms = (
            generation.last_updated_ms + hints.expected_resume_in_ms
            if hints is not None and hints.expected_resume_in_ms is not None
            else None
        )
        return AgentKVPolicyOwner(
            key=key,
            state=generation.state,
            priority=hints.priority if hints and hints.priority is not None else 0,
            retain_until_ms=retain_until_ms,
            resume_due_at_ms=resume_due_at_ms,
            last_updated_ms=generation.last_updated_ms,
        )

    @staticmethod
    def _record_event_details(
        generation: _GenerationRecord, event: AgentKVEvent
    ) -> None:
        generation.latest_hints = event.hints
        generation.last_event_type = event.event_type
        generation.last_event_emitted_at_ms = event.emitted_at_ms
        generation.reason = event.reason

    def get_session_snapshot(
        self, namespace: str, session_id: str, branch_id: str = "main"
    ) -> dict[str, object] | None:
        """Return a content-free state snapshot for tests and diagnostics."""
        session = self._sessions.get((namespace, session_id))
        if session is None:
            return None
        branch = session.branches.get(branch_id)
        current_generation = branch.current_generation if branch else 0
        current = (
            branch.generations.get(current_generation)
            if branch and current_generation
            else None
        )
        return {
            "namespace": namespace,
            "session_id": session_id,
            "branch_id": branch_id,
            "current_generation": current_generation,
            "state": current.state.value if current else None,
            "last_event_sequence": session.last_event_sequence,
            "terminated": session.terminated,
            "active_request_count": len(current.active_request_ids) if current else 0,
            "finished_request_count": (
                current.finished_request_count if current else 0
            ),
            "block_hash_count": len(current.block_hashes) if current else 0,
            "priority": (
                current.latest_hints.priority
                if current and current.latest_hints is not None
                else None
            ),
            "expected_resume_in_ms": (
                current.latest_hints.expected_resume_in_ms
                if current and current.latest_hints is not None
                else None
            ),
        }

    def _result(
        self,
        status: AgentKVEventStatus,
        session: _SessionRecord,
        event: AgentKVEvent,
    ) -> dict[str, object]:
        self.metrics.record_event(event.event_type, status)
        snapshot = self.get_session_snapshot(
            event.namespace, event.session_id, event.branch_id
        )
        current_generation = snapshot["current_generation"] if snapshot else 0
        state = snapshot["state"] if snapshot else None
        return {
            "protocol_version": AGENT_KV_PROTOCOL_VERSION,
            "event_id": event.event_id,
            "status": status.value,
            "accepted": status == AgentKVEventStatus.ACCEPTED,
            "current_generation": current_generation,
            "current_state": state,
            "last_event_sequence": session.last_event_sequence,
        }
