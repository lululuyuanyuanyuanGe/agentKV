# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP models for the AgentKV lifecycle protocol."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllm.v1.agent_kv.protocol import (
    AGENT_KV_PROTOCOL_VERSION,
    DEFAULT_AGENT_KV_BRANCH_ID,
    MAX_AGENT_KV_IDENTIFIER_LENGTH,
    AgentKVEvent,
    AgentKVEventHints,
    AgentKVEventType,
)


class AgentKVEventHintsRequest(BaseModel):
    """Optional business hints supplied by the upstream agent runtime."""

    model_config = ConfigDict(extra="forbid")

    expected_resume_in_ms: int | None = Field(default=None, ge=0)
    priority: int | None = Field(default=None, ge=0, le=3)
    retain_for_ms: int | None = Field(default=None, ge=0)
    prefetch_allowed: bool | None = None

    def to_engine_hints(self) -> AgentKVEventHints:
        return AgentKVEventHints(
            expected_resume_in_ms=self.expected_resume_in_ms,
            priority=self.priority,
            retain_for_ms=self.retain_for_ms,
            prefetch_allowed=self.prefetch_allowed,
        )


class AgentKVEventRequest(BaseModel):
    """Version 1 lifecycle event accepted from an upstream agent runtime."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = AGENT_KV_PROTOCOL_VERSION
    event_id: str = Field(min_length=1, max_length=MAX_AGENT_KV_IDENTIFIER_LENGTH)
    event_sequence: int = Field(ge=1)
    event_type: AgentKVEventType
    emitted_at_ms: int = Field(ge=0)
    namespace: str = Field(min_length=1, max_length=MAX_AGENT_KV_IDENTIFIER_LENGTH)
    session_id: str = Field(min_length=1, max_length=MAX_AGENT_KV_IDENTIFIER_LENGTH)
    generation: int = Field(ge=1)
    branch_id: str = Field(
        default=DEFAULT_AGENT_KV_BRANCH_ID,
        min_length=1,
        max_length=MAX_AGENT_KV_IDENTIFIER_LENGTH,
    )
    hints: AgentKVEventHintsRequest | None = None
    reason: str | None = Field(default=None, max_length=64)

    @field_validator("event_id", "namespace", "session_id", "branch_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    def to_engine_event(self) -> AgentKVEvent:
        return AgentKVEvent(
            protocol_version=self.protocol_version,
            event_id=self.event_id,
            event_sequence=self.event_sequence,
            event_type=self.event_type,
            emitted_at_ms=self.emitted_at_ms,
            namespace=self.namespace,
            session_id=self.session_id,
            generation=self.generation,
            branch_id=self.branch_id,
            hints=self.hints.to_engine_hints() if self.hints else None,
            reason=self.reason,
        )


class AgentKVEventResponse(BaseModel):
    """Idempotent acknowledgement returned by the lifecycle endpoint."""

    protocol_version: int
    event_id: str
    status: str
    accepted: bool
    current_generation: int
    current_state: str | None
    last_event_sequence: int
