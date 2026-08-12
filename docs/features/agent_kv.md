# AgentKV lifecycle protocol

AgentKV is an experimental, application-aware lifecycle signal for the vLLM
prefix cache. Version 1 records upstream intent and cache content identities but
does not change cache placement or eviction behavior.

The protocol separates business facts from resource decisions:

- The upstream agent runtime reports stable identity and lifecycle events.
- vLLM observes cache hashes, allocation pressure, capacity, and transfers.
- A future policy layer will combine both sources to choose retention,
  offload, eviction, or prefetch actions.

Upstream clients must never send prompt text, model output, tool arguments,
tool results, physical cache block identifiers, or cache hashes through this
protocol.

## Inference request identity

An inference request opts in to AgentKV by providing all required headers:

| Header | Required | Description |
| --- | --- | --- |
| `X-Session-ID` | Yes | Stable opaque agent session identity. |
| `X-AgentKV-Namespace` | Yes | Trusted tenant or isolation namespace. |
| `X-AgentKV-Generation` | Yes | Positive, monotonically increasing inference generation within the branch. |
| `X-AgentKV-Branch-ID` | No | Opaque branch identity; defaults to `main`. |
| `X-AgentKV-Protocol-Version` | No | Protocol version; defaults to `1`. |

For non-HTTP callers, the equivalent `vllm_xargs` keys are
`agent_kv_namespace`, `agent_kv_generation`, `agent_kv_branch_id`, and
`agent_kv_protocol_version`.

Headers take precedence over `vllm_xargs`. A trusted gateway should set or
validate `X-AgentKV-Namespace`; an untrusted client must not be allowed to
select another tenant's namespace.

Example:

```http
POST /v1/chat/completions
X-Session-ID: session-opaque-123
X-AgentKV-Namespace: tenant-opaque-17
X-AgentKV-Generation: 7
X-AgentKV-Branch-ID: main
```

Requests that provide no AgentKV-specific fields retain native vLLM behavior.
Partially specified AgentKV identities are rejected instead of being silently
mis-associated.

## Lifecycle events

The development endpoint is available when `VLLM_SERVER_DEV_MODE=1`:

```http
POST /v1/agent-kv/events
Content-Type: application/json
```

Version 1 supports four lifecycle facts:

| Event | Scope | Meaning |
| --- | --- | --- |
| `SUSPEND` | Generation | The agent does not currently need another inference. |
| `RESUME_PENDING` | Generation | Another inference is expected soon. |
| `GENERATION_EXPIRED` | Generation | This generation no longer has an owner. |
| `SESSION_TERMINATED` | Session | The entire session is permanently terminated. |

Example:

```json
{
  "protocol_version": 1,
  "event_id": "event-opaque-456",
  "event_sequence": 18,
  "event_type": "SUSPEND",
  "emitted_at_ms": 1786339200000,
  "namespace": "tenant-opaque-17",
  "session_id": "session-opaque-123",
  "branch_id": "main",
  "generation": 7,
  "hints": {
    "expected_resume_in_ms": 5000,
    "priority": 2,
    "retain_for_ms": 120000,
    "prefetch_allowed": true
  },
  "reason": "tool"
}
```

`event_sequence` is monotonically increasing across a session. The endpoint
accepts at-least-once delivery: a repeated `event_id` returns `duplicate`, an
event behind the last sequence returns `stale_event`, and a state transition
for an older generation returns `stale_generation`.

Example acknowledgement:

```json
{
  "protocol_version": 1,
  "event_id": "event-opaque-456",
  "status": "accepted",
  "accepted": true,
  "current_generation": 7,
  "current_state": "SUSPENDED",
  "last_event_sequence": 18
}
```

The optional hints express business intent only. They do not force a cache
action. Missing, late, or invalid lifecycle signals must not affect inference;
when AgentKV has no valid state, cache management falls back to native vLLM
behavior.

## Generation semantics

`generation` identifies the cache snapshot produced by one logical inference
within `(namespace, session_id, branch_id)`. It is independent for each branch.
An older generation's delayed `SUSPEND` event cannot change a newer active
generation. `SESSION_TERMINATED` is session-scoped and terminates every branch.

## Current limitations

- Version 1 is metadata-only and does not retain, offload, evict, or prefetch.
- Lifecycle endpoints are development endpoints and require an authenticated,
  trusted gateway before production use.
- Data-parallel deployments do not yet provide session-affine request routing.
- Prometheus labels must not contain session, event, branch, or request
  identities because those values have unbounded cardinality.
