# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.simple_kv_offload.int8_backend import (
    Int8DoubleBufferBackend,
    build_int8_packed_layout,
)


def test_int8_layout_places_scales_before_aligned_payloads() -> None:
    caches = {
        "key": torch.empty((4, 17), dtype=torch.float16),
        "value": torch.empty((4, 9), dtype=torch.bfloat16),
    }

    layout = build_int8_packed_layout(caches)

    assert layout.num_blocks == 4
    assert layout.original_row_bytes == 52
    assert layout.row_bytes == 48
    assert [segment.scale_offset for segment in layout.segments] == [0, 4]
    assert [segment.payload_offset for segment in layout.segments] == [16, 33]


@pytest.mark.parametrize(
    "caches,match",
    [
        ({}, "at least one"),
        ({"key": torch.empty((2, 4), dtype=torch.float32)}, "FP16 or BF16"),
        ({"key": torch.empty(8, dtype=torch.float16)}, "two-dimensional"),
        (
            {
                "key": torch.empty((2, 8), dtype=torch.float16),
                "value": torch.empty((3, 8), dtype=torch.float16),
            },
            "share a block count",
        ),
    ],
)
def test_int8_layout_rejects_unsafe_cache_shapes(
    caches: dict[str, torch.Tensor], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_int8_packed_layout(caches)


def test_int8_backend_requires_positive_staging_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        Int8DoubleBufferBackend(buffer_blocks=0)
