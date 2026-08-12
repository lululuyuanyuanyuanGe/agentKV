# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.agent_kv.controller import AgentKVController
from vllm.v1.agent_kv.policy import AgentKVEvictionClass
from vllm.v1.agent_kv.protocol import (
    AgentKVEvent,
    AgentKVEventHints,
    AgentKVEventStatus,
    AgentKVEventType,
    AgentKVLifecycleState,
    AgentKVRequestMetadata,
)


@dataclass(frozen=True)
class _Candidate:
    block_id: int
    cache_keys: tuple[bytes, ...]
    content_hashes: tuple[bytes, ...]


def make_metadata(
    generation: int,
    session_id: str = "session-a",
) -> AgentKVRequestMetadata:
    return AgentKVRequestMetadata(
        namespace="tenant-a",
        session_id=session_id,
        branch_id="main",
        generation=generation,
    )


def make_event_for_session(
    event_id: str,
    sequence: int,
    event_type: AgentKVEventType,
    generation: int,
    session_id: str = "session-a",
) -> AgentKVEvent:
    return AgentKVEvent(
        event_id=event_id,
        event_sequence=sequence,
        event_type=event_type,
        emitted_at_ms=1_000 + sequence,
        namespace="tenant-a",
        session_id=session_id,
        branch_id="main",
        generation=generation,
    )


def make_event(
    event_id: str,
    sequence: int,
    event_type: AgentKVEventType,
    generation: int,
) -> AgentKVEvent:
    return make_event_for_session(event_id, sequence, event_type, generation)


def test_request_registration_and_finish_capture_content_hashes() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")

    active = controller.get_session_snapshot("tenant-a", "session-a")
    assert active is not None
    assert active["current_generation"] == 7
    assert active["state"] == AgentKVLifecycleState.ACTIVE.value
    assert active["active_request_count"] == 1

    controller.finish_request("request-7", [b"hash-1", b"hash-2"])

    finished = controller.get_session_snapshot("tenant-a", "session-a")
    assert finished is not None
    assert finished["active_request_count"] == 0
    assert finished["finished_request_count"] == 1
    assert finished["block_hash_count"] == 2


def test_events_are_idempotent_and_ordered() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")
    event = make_event("event-1", 10, AgentKVEventType.SUSPEND, 7)

    accepted = controller.apply_event(event)
    duplicate = controller.apply_event(event)
    stale = controller.apply_event(
        make_event("event-stale", 9, AgentKVEventType.RESUME_PENDING, 7)
    )

    assert accepted["status"] == AgentKVEventStatus.ACCEPTED.value
    assert accepted["current_state"] == AgentKVLifecycleState.SUSPENDED.value
    assert duplicate["status"] == AgentKVEventStatus.DUPLICATE.value
    assert stale["status"] == AgentKVEventStatus.STALE_EVENT.value
    assert stale["last_event_sequence"] == 10


def test_delayed_event_cannot_change_a_newer_generation() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")
    controller.apply_event(make_event("suspend-7", 10, AgentKVEventType.SUSPEND, 7))
    controller.register_request(make_metadata(8), "request-8")

    delayed = controller.apply_event(
        make_event("delayed-suspend-7", 11, AgentKVEventType.SUSPEND, 7)
    )

    assert delayed["status"] == AgentKVEventStatus.STALE_GENERATION.value
    assert delayed["current_generation"] == 8
    assert delayed["current_state"] == AgentKVLifecycleState.ACTIVE.value


def test_expiring_an_old_generation_does_not_expire_current_generation() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")
    controller.register_request(make_metadata(8), "request-8")

    result = controller.apply_event(
        make_event("expire-7", 1, AgentKVEventType.GENERATION_EXPIRED, 7)
    )

    assert result["status"] == AgentKVEventStatus.ACCEPTED.value
    assert result["current_generation"] == 8
    assert result["current_state"] == AgentKVLifecycleState.ACTIVE.value


def test_session_termination_is_session_scoped() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")

    terminated = controller.apply_event(
        make_event("terminate", 1, AgentKVEventType.SESSION_TERMINATED, 7)
    )
    after_termination = controller.apply_event(
        make_event("resume", 2, AgentKVEventType.RESUME_PENDING, 7)
    )

    assert terminated["status"] == AgentKVEventStatus.ACCEPTED.value
    assert terminated["current_state"] == AgentKVLifecycleState.TERMINATED.value
    assert after_termination["status"] == AgentKVEventStatus.SESSION_TERMINATED.value


def test_business_hints_are_retained_for_policy_evaluation() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(7), "request-7")
    event = AgentKVEvent(
        event_id="suspend-with-hints",
        event_sequence=1,
        event_type=AgentKVEventType.SUSPEND,
        emitted_at_ms=1_000,
        namespace="tenant-a",
        session_id="session-a",
        generation=7,
        hints=AgentKVEventHints(
            expected_resume_in_ms=10_000,
            priority=3,
            retain_for_ms=60_000,
            prefetch_allowed=True,
        ),
    )

    controller.apply_event(event)
    snapshot = controller.get_session_snapshot("tenant-a", "session-a")

    assert snapshot is not None
    assert snapshot["priority"] == 3
    assert snapshot["expected_resume_in_ms"] == 10_000


def test_content_hash_ownership_accumulates_across_finished_requests() -> None:
    controller = AgentKVController()
    metadata = make_metadata(7)
    controller.register_request(metadata, "request-1")
    controller.finish_request("request-1", [b"prefix"])
    controller.register_request(metadata, "request-2")
    controller.finish_request("request-2", [b"prefix", b"continuation"])

    owners = controller.get_block_owners(b"prefix")
    continuation_owners = controller.get_block_owners(b"continuation")

    assert len(owners) == 1
    assert owners == continuation_owners


def test_shared_hash_tracks_each_session_owner() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(1, "session-a"), "request-a")
    controller.register_request(make_metadata(1, "session-b"), "request-b")
    controller.finish_request("request-a", [b"shared"])
    controller.finish_request("request-b", [b"shared"])

    owners = controller.get_block_owners(b"shared")

    assert {owner.session_id for owner in owners} == {"session-a", "session-b"}


def test_shared_hash_remains_protected_by_live_owner_after_other_terminates() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(1, "terminated"), "request-terminated")
    controller.register_request(make_metadata(1, "live"), "request-live")
    controller.finish_request("request-terminated", [b"shared"])
    controller.finish_request("request-live", [b"shared"])
    controller.apply_event(
        make_event_for_session(
            "terminate",
            1,
            AgentKVEventType.SESSION_TERMINATED,
            1,
            "terminated",
        )
    )
    controller.apply_event(
        make_event_for_session(
            "resume",
            1,
            AgentKVEventType.RESUME_PENDING,
            1,
            "live",
        )
    )
    candidates = [
        _Candidate(1, (b"shared-key",), (b"shared",)),
        _Candidate(2, (b"unowned-key",), (b"unowned",)),
    ]

    plan = controller.plan_evictions(candidates, num_at_risk_blocks=1)

    assert [target.block_id for target in plan] == [2]


def test_lifecycle_state_changes_content_addressed_eviction_plan() -> None:
    controller = AgentKVController()
    controller.register_request(make_metadata(1, "resume"), "request-resume")
    controller.register_request(make_metadata(1, "expired"), "request-expired")
    controller.finish_request("request-resume", [b"resume-hash"])
    controller.finish_request("request-expired", [b"expired-hash"])
    controller.apply_event(
        AgentKVEvent(
            event_id="resume",
            event_sequence=1,
            event_type=AgentKVEventType.RESUME_PENDING,
            emitted_at_ms=1_000,
            namespace="tenant-a",
            session_id="resume",
            generation=1,
        )
    )
    controller.apply_event(
        AgentKVEvent(
            event_id="expire",
            event_sequence=1,
            event_type=AgentKVEventType.GENERATION_EXPIRED,
            emitted_at_ms=1_000,
            namespace="tenant-a",
            session_id="expired",
            generation=1,
        )
    )
    candidates = [
        _Candidate(1, (b"resume-key",), (b"resume-hash",)),
        _Candidate(2, (b"expired-key",), (b"expired-hash",)),
    ]

    plan = controller.plan_evictions(candidates, num_at_risk_blocks=1)

    assert [target.block_id for target in plan] == [2]
    assert plan[0].eviction_class == AgentKVEvictionClass.EVICT_FIRST
