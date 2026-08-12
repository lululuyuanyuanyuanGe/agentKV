# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wire-safe types for the AgentKV lifecycle protocol."""

import enum

import msgspec

AGENT_KV_PROTOCOL_VERSION = 1
DEFAULT_AGENT_KV_BRANCH_ID = "main"
MAX_AGENT_KV_IDENTIFIER_LENGTH = 256


class AgentKVEventType(str, enum.Enum):
    """Lifecycle facts emitted by an upstream agent runtime."""

    SUSPEND = "SUSPEND"
    RESUME_PENDING = "RESUME_PENDING"
    GENERATION_EXPIRED = "GENERATION_EXPIRED"
    SESSION_TERMINATED = "SESSION_TERMINATED"


class AgentKVLifecycleState(str, enum.Enum):
    """Observed lifecycle state for one session generation."""

    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RESUME_PENDING = "RESUME_PENDING"
    EXPIRED = "EXPIRED"
    TERMINATED = "TERMINATED"


class AgentKVEventStatus(str, enum.Enum):
    """Result of applying a lifecycle event."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE_EVENT = "stale_event"
    STALE_GENERATION = "stale_generation"
    SESSION_TERMINATED = "session_terminated"


def _validate_identifier(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > MAX_AGENT_KV_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{name} must not exceed {MAX_AGENT_KV_IDENTIFIER_LENGTH} characters"
        )


class AgentKVRequestMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Identity attached to each inference request from an agent runtime."""

    namespace: str
    session_id: str
    generation: int
    branch_id: str = DEFAULT_AGENT_KV_BRANCH_ID
    protocol_version: int = AGENT_KV_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_KV_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported AgentKV protocol version: "
                f"{self.protocol_version}; expected {AGENT_KV_PROTOCOL_VERSION}"
            )
        _validate_identifier("namespace", self.namespace)
        _validate_identifier("session_id", self.session_id)
        _validate_identifier("branch_id", self.branch_id)
        if self.generation < 1:
            raise ValueError("generation must be greater than or equal to 1")


def parse_agent_kv_request_metadata(
    *,
    session_id: object,
    namespace: object,
    generation: object,
    branch_id: object = None,
    protocol_version: object = None,
) -> AgentKVRequestMetadata:
    """Validate raw request values and build the engine metadata type."""
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("AgentKV metadata requires session_id or X-Session-ID")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("AgentKV metadata requires X-AgentKV-Namespace")

    def parse_positive_int(name: str, value: object, default: int | None = None) -> int:
        if value is None and default is not None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError(f"{name} must be a positive integer")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed < 1 or (isinstance(value, str) and str(parsed) != value.strip()):
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    parsed_generation = parse_positive_int("X-AgentKV-Generation", generation)
    parsed_version = parse_positive_int(
        "X-AgentKV-Protocol-Version",
        protocol_version,
        AGENT_KV_PROTOCOL_VERSION,
    )
    if branch_id is None:
        branch_id = DEFAULT_AGENT_KV_BRANCH_ID
    if not isinstance(branch_id, str) or not branch_id:
        raise ValueError("X-AgentKV-Branch-ID must be a non-empty string")

    return AgentKVRequestMetadata(
        protocol_version=parsed_version,
        namespace=namespace,
        session_id=session_id,
        generation=parsed_generation,
        branch_id=branch_id,
    )


class AgentKVEventHints(msgspec.Struct, frozen=True, kw_only=True):
    """Optional business hints; resource actions remain engine decisions."""

    expected_resume_in_ms: int | None = None
    priority: int | None = None
    retain_for_ms: int | None = None
    prefetch_allowed: bool | None = None

    def __post_init__(self) -> None:
        if self.expected_resume_in_ms is not None and self.expected_resume_in_ms < 0:
            raise ValueError("expected_resume_in_ms must be non-negative")
        if self.priority is not None and not 0 <= self.priority <= 3:
            raise ValueError("priority must be between 0 and 3")
        if self.retain_for_ms is not None and self.retain_for_ms < 0:
            raise ValueError("retain_for_ms must be non-negative")


class AgentKVEvent(msgspec.Struct, frozen=True, kw_only=True):
    """Versioned lifecycle event accepted by the engine core."""

    event_id: str
    event_sequence: int
    event_type: AgentKVEventType
    emitted_at_ms: int
    namespace: str
    session_id: str
    generation: int
    branch_id: str = DEFAULT_AGENT_KV_BRANCH_ID
    hints: AgentKVEventHints | None = None
    reason: str | None = None
    protocol_version: int = AGENT_KV_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_KV_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported AgentKV protocol version: "
                f"{self.protocol_version}; expected {AGENT_KV_PROTOCOL_VERSION}"
            )
        _validate_identifier("event_id", self.event_id)
        _validate_identifier("namespace", self.namespace)
        _validate_identifier("session_id", self.session_id)
        _validate_identifier("branch_id", self.branch_id)
        if self.event_sequence < 1:
            raise ValueError("event_sequence must be greater than or equal to 1")
        if self.emitted_at_ms < 0:
            raise ValueError("emitted_at_ms must be non-negative")
        if self.generation < 1:
            raise ValueError("generation must be greater than or equal to 1")
        if self.reason is not None and len(self.reason) > 64:
            raise ValueError("reason must not exceed 64 characters")
