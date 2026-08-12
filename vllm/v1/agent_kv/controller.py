# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side lifecycle state tracking for AgentKV."""

import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field

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
    block_hashes: tuple[bytes, ...] = ()
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


class AgentKVController:
    """Tracks upstream lifecycle facts without changing cache policy.

    The controller deliberately stores content hashes instead of physical block
    identifiers. Block identifiers are allocator-owned and may be reused after
    a request finishes. Cache actions will be added behind this boundary in a
    later phase.
    """

    def __init__(self, max_sessions: int = _DEFAULT_MAX_SESSIONS) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be greater than or equal to 1")
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], _SessionRecord] = OrderedDict()
        self._request_index: dict[str, tuple[tuple[str, str], str, int]] = {}

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _get_or_create_session(
        self, namespace: str, session_id: str
    ) -> tuple[tuple[str, str], _SessionRecord]:
        key = (namespace, session_id)
        session = self._sessions.get(key)
        if session is None:
            while len(self._sessions) >= self.max_sessions:
                evicted_key, _ = self._sessions.popitem(last=False)
                self._remove_request_index_for_session(evicted_key)
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

    @staticmethod
    def _prune_generations(branch: _BranchRecord) -> None:
        while len(branch.generations) > _MAX_GENERATIONS_PER_BRANCH:
            generation, record = next(iter(branch.generations.items()))
            if record.active_request_ids:
                break
            del branch.generations[generation]

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
        key, session = self._get_or_create_session(
            metadata.namespace, metadata.session_id
        )
        branch, generation = self._get_or_create_generation(
            session, metadata.branch_id, metadata.generation
        )
        now_ms = self._now_ms()
        generation.active_request_ids.add(request_id)
        generation.last_updated_ms = now_ms
        if not session.terminated and metadata.generation >= branch.current_generation:
            branch.current_generation = metadata.generation
            generation.state = AgentKVLifecycleState.ACTIVE
        session.last_updated_ms = now_ms
        self._request_index[request_id] = (
            key,
            metadata.branch_id,
            metadata.generation,
        )
        self._prune_generations(branch)

    def finish_request(self, request_id: str, block_hashes: Iterable[bytes]) -> None:
        """Persist the content identity before scheduler request state is freed."""
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
        hashes = tuple(bytes(block_hash) for block_hash in block_hashes)
        if len(hashes) >= len(generation.block_hashes):
            generation.block_hashes = hashes
        now_ms = self._now_ms()
        generation.last_updated_ms = now_ms
        session.last_updated_ms = now_ms
        self._sessions.move_to_end(key)
        self._prune_generations(branch)

    def apply_event(self, event: AgentKVEvent) -> dict[str, object]:
        """Apply one idempotent, ordered lifecycle event."""
        _, session = self._get_or_create_session(event.namespace, event.session_id)

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
            self._prune_generations(branch)
            return self._result(AgentKVEventStatus.ACCEPTED, session, event)

        if event.generation < branch.current_generation:
            self._prune_generations(branch)
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
        self._prune_generations(branch)
        return self._result(AgentKVEventStatus.ACCEPTED, session, event)

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
