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

## Enablement and safe fallback

AgentKV is disabled by default. Enable it explicitly before starting vLLM:

```bash
export VLLM_ENABLE_AGENT_KV=1
```

When disabled, requests without AgentKV metadata retain native vLLM behavior.
A request carrying AgentKV metadata is rejected, and the lifecycle route is not
registered. This prevents a caller from assuming that lifecycle policy is
active when the server is not configured for it.

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
incorrectly associated.

## Lifecycle events

The development endpoint is available when both
`VLLM_ENABLE_AGENT_KV=1` and `VLLM_SERVER_DEV_MODE=1`:

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
    "lazy_offload": true,
    "kv_offload_quantization": "int8",
    "quantization_buffer_blocks": 64
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

## Fused 8-bit integer (INT8) transport

`kv_offload_quantization=int8` enables an optional lossy transport path for
16-bit floating-point (FP16) and Brain Floating Point 16 (BF16) cache blocks on
NVIDIA Compute Unified Device Architecture (CUDA). Each packed row contains an
aligned 32-bit floating-point (FP32) scale header followed by contiguous INT8
payloads for every unique KV tensor segment. Quantization uses one symmetric
scale per block and segment.

The store pipeline waits for model computation, quantizes into one of two GPU
staging slots, and copies the packed row to persistent pinned host memory. The
quantization and Peripheral Component Interconnect Express (PCIe) streams are
independent, so quantizing the next chunk can overlap transfer of the previous
chunk. The load pipeline performs the inverse:
one stream copies packed rows into a staging slot while another waits on a CUDA
event and runs fused unpack-and-dequantize into the destination KV blocks.

Transfers larger than `quantization_buffer_blocks` are chunked and alternate
between the two slots. Slot completion events prevent reuse while an earlier
transfer or kernel is still in flight. The final event is recorded on the
ordered transfer or dequantization stream and represents completion of the
entire logical operation.

## Observability

Scheduler statistics export the following Prometheus metrics:

- `vllm:agent_kv_events` by event type and status;
- `vllm:agent_kv_cache_actions` by action and bounded reason;
- `vllm:agent_kv_offload_results` by completion result;
- `vllm:agent_kv_fallbacks` by bounded reason;
- `vllm:agent_kv_cursor_resets` by bounded reason;
- `vllm:agent_kv_policy_revisions`;
- aggregate gauges for sessions, generations, owned hashes, and in-flight
  lower-tier store blocks.

Session, event, branch, request, namespace, and other upstream identities are
not metric labels. The implementation uses only bounded engine-owned label
values to avoid unbounded cardinality and identity leakage.

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
- INT8 transport currently supports only NVIDIA CUDA, CPU offload, and FP16 or
  BF16 cache storage. It is lossy and requires model-specific evaluation.
- Offload capacity accounting remains based on the uncompressed block size.
- A production dashboard, alert thresholds, and disabled-versus-enabled
  scheduler overhead baseline have not yet been established.
