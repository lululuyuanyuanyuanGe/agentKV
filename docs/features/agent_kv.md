# AgentKV lifecycle protocol

AgentKV is an experimental, application-aware lifecycle signal for the vLLM
prefix cache. Version 1 records upstream intent and cache content identities and
uses them to refine eviction order and, when lazy native offload is configured,
to choose whether an idle block should stay local, be copied to a lower tier,
or be dropped without creating another copy.

The protocol separates business facts from resource decisions:

- The upstream agent runtime reports stable identity and lifecycle events.
- vLLM observes cache hashes, allocation pressure, capacity, and transfers.
- The policy layer combines both sources to refine immediate eviction choices.

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

## Eviction behavior

AgentKV only evaluates blocks that are already free and therefore eligible for
native prefix-cache eviction from graphics processing unit (GPU) memory. It
never removes or reorders a block referenced by a running request. The policy is
invoked only when the current allocation would overwrite cached blocks.

The default retention order is `RESUME_PENDING`, `ACTIVE`, `SUSPENDED`, native
unowned cache, then `EXPIRED` or `TERMINATED`. Higher priority, an active
`retain_for_ms` window, and a nearer expected resume refine ties. A block shared
by several generations uses its most protective live owner. Blocks with equal
AgentKV policy scores retain native least recently used (LRU) order.

Plans use content hashes for ownership and validate the complete cache key set
before changing the free queue. A stale plan is ignored if its physical block
identifier has been reused or its cache aliases have changed.

## Lazy tier actions

When `SimpleCPUOffloadConnector` is configured with `lazy_offload=true`, its
soft-pressure scan asks AgentKV for one action per eligible cached block:

| Most protective owner state | Action | Effect |
| --- | --- | --- |
| `ACTIVE` or `RESUME_PENDING` | `KEEP` | Skip the lower-tier copy and leave eviction to the protected GPU ordering. |
| `SUSPENDED` with an active `retain_for_ms` window | `KEEP` | Reconsider after the retain deadline. |
| `SUSPENDED` | `OFFLOAD` | Copy to the configured central processing unit (CPU) or disk tier when capacity is available. |
| All owners are `EXPIRED` or `TERMINATED` | `DROP` | Do not create a lower-tier copy; normal allocation performs physical eviction. |
| No AgentKV owner | `DEFAULT` | Preserve the connector's native lazy-offload behavior. |

Example connector configuration:

```json
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 8589934592,
    "lazy_offload": true
  }
}
```

Every action carries the complete cache-key snapshot, content hashes, owner
generations, and controller policy revision. The connector compares the live
cache snapshot before scheduling a copy. Because a store is asynchronous, it
also asks the controller to validate `OFFLOAD` again before publishing the
completed lower-tier entry. A resume, termination, owner-set change, block
reuse, or alias change makes the stale result non-cacheable and only releases
its transfer references.

AgentKV state changes reset the lazy scan cursor so a previously skipped block
can be reconsidered. Time-limited retention records the earliest deadline and
does the same when that deadline expires. Capacity exhaustion is best effort:
the scheduler never blocks inference waiting for lower-tier space.

## Current limitations

- Tier actions currently integrate only with `SimpleCPUOffloadConnector` in
  lazy mode. Eager offload and other connectors keep their native behavior.
- Version 1 does not prefetch cache data, reserve fixed capacity, or guarantee
  retention. `KEEP` is an ordering preference, and `DROP` does not proactively
  remove a block.
- Eviction within the lower tier still uses the connector's native policy;
  AgentKV currently chooses admission to that tier, not its internal victims.
- Lifecycle endpoints are development endpoints and require an authenticated,
  trusted gateway before production use.
- Data-parallel deployments do not yet provide session-affine request routing.
- Prometheus labels must not contain session, event, branch, or request
  identities because those values have unbounded cardinality.
