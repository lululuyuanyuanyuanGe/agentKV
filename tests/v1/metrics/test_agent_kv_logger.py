# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.agent_kv.metrics import AgentKVStats
from vllm.v1.metrics.loggers import PrometheusStatLogger


class _BoundMetric:
    def __init__(self, metric: _RecordingMetric, labels: tuple[str, ...]) -> None:
        self.metric = metric
        self.labels = labels

    def inc(self, value: int) -> None:
        self.metric.increments.append((self.labels, value))


class _RecordingMetric:
    def __init__(self) -> None:
        self.increments: list[tuple[tuple[str, ...], int]] = []
        self.values: list[int] = []

    def labels(self, *labels: str) -> _BoundMetric:
        return _BoundMetric(self, labels)

    def inc(self, value: int) -> None:
        self.increments.append(((), value))

    def set(self, value: int) -> None:
        self.values.append(value)


def test_prometheus_logger_maps_agent_kv_stats_to_bounded_labels() -> None:
    logger = object.__new__(PrometheusStatLogger)
    logger.per_engine_labelvalues = {0: ["model-a"]}

    metric_names = (
        "counter_agent_kv_events",
        "counter_agent_kv_actions",
        "counter_agent_kv_offload_results",
        "counter_agent_kv_fallbacks",
        "counter_agent_kv_cursor_resets",
    )
    metrics: dict[str, _RecordingMetric] = {}
    for name in metric_names:
        metric = _RecordingMetric()
        metrics[name] = metric
        setattr(logger, name, metric)

    policy_revisions = _RecordingMetric()
    logger.counter_agent_kv_policy_revisions = {0: policy_revisions}
    gauges: dict[str, _RecordingMetric] = {}
    for name in (
        "gauge_agent_kv_sessions",
        "gauge_agent_kv_generations",
        "gauge_agent_kv_owned_hashes",
        "gauge_agent_kv_inflight_store_blocks",
    ):
        metric = _RecordingMetric()
        gauges[name] = metric
        setattr(logger, name, {0: metric})

    stats = AgentKVStats(
        event_counts={"SUSPEND": {"accepted": 2}},
        action_counts={"offload": {"suspended": 3}},
        offload_result_counts={"invalidated": 4},
        fallback_counts={"validator_error": 5},
        cursor_reset_counts={"policy_revision": 6},
        policy_revisions=7,
        num_sessions=8,
        num_generations=9,
        num_owned_hashes=10,
        num_inflight_store_blocks=11,
    )

    PrometheusStatLogger._record_agent_kv_stats(logger, stats, 0)

    assert metrics["counter_agent_kv_events"].increments == [
        (("model-a", "SUSPEND", "accepted"), 2)
    ]
    assert metrics["counter_agent_kv_actions"].increments == [
        (("model-a", "offload", "suspended"), 3)
    ]
    assert metrics["counter_agent_kv_offload_results"].increments == [
        (("model-a", "invalidated"), 4)
    ]
    assert metrics["counter_agent_kv_fallbacks"].increments == [
        (("model-a", "validator_error"), 5)
    ]
    assert metrics["counter_agent_kv_cursor_resets"].increments == [
        (("model-a", "policy_revision"), 6)
    ]
    assert policy_revisions.increments == [((), 7)]
    assert gauges["gauge_agent_kv_sessions"].values == [8]
    assert gauges["gauge_agent_kv_generations"].values == [9]
    assert gauges["gauge_agent_kv_owned_hashes"].values == [10]
    assert gauges["gauge_agent_kv_inflight_store_blocks"].values == [11]
