# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.int8_backend import (
    build_int8_packed_layout,
    int8_kernels_available,
    quantize_pack,
    unpack_dequantize,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not torch.cuda.is_available()
    or not int8_kernels_available(),
    reason="AgentKV INT8 kernels require an NVIDIA CUDA build",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_agent_kv_int8_kernel_roundtrip_and_block_mapping(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    source = torch.randn((6, 513), dtype=dtype, device="cuda")
    layout = build_int8_packed_layout({"kv": source})
    packed = torch.zeros((3, layout.row_bytes), dtype=torch.uint8, device="cuda")
    source_ids = torch.tensor([4, 1, 5], dtype=torch.int64, device="cuda")
    segment = layout.segments[0]

    quantize_pack(
        source,
        source_ids,
        packed,
        segment.payload_offset,
        segment.scale_offset,
    )

    output = torch.zeros_like(source)
    destination_ids = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    unpack_dequantize(
        packed,
        destination_ids,
        output,
        segment.payload_offset,
        segment.scale_offset,
    )
    torch.cuda.synchronize()

    expected = source[source_ids]
    actual = output[destination_ids]
    scale_bytes = packed[:, segment.scale_offset : segment.scale_offset + 4]
    scales = scale_bytes.contiguous().view(torch.float32).flatten()
    error = (actual.float() - expected.float()).abs()
    assert torch.all(error.amax(dim=1) <= scales * 0.6 + 0.02)
    assert torch.count_nonzero(output[[1, 4, 5]]) == 0


def test_agent_kv_int8_kernel_preserves_zero_block() -> None:
    source = torch.zeros((2, 256), dtype=torch.float16, device="cuda")
    layout = build_int8_packed_layout({"kv": source})
    packed = torch.empty((1, layout.row_bytes), dtype=torch.uint8, device="cuda")
    block_ids = torch.tensor([1], dtype=torch.int64, device="cuda")
    segment = layout.segments[0]

    quantize_pack(
        source,
        block_ids,
        packed,
        segment.payload_offset,
        segment.scale_offset,
    )
    unpack_dequantize(
        packed,
        block_ids,
        source,
        segment.payload_offset,
        segment.scale_offset,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(source) == 0


def test_agent_kv_int8_kernel_packs_multiple_segments() -> None:
    caches = {
        "key": torch.randn((4, 257), dtype=torch.float16, device="cuda"),
        "value": torch.randn((4, 129), dtype=torch.float16, device="cuda"),
    }
    layout = build_int8_packed_layout(caches)
    packed = torch.zeros((2, layout.row_bytes), dtype=torch.uint8, device="cuda")
    source_ids = torch.tensor([3, 1], dtype=torch.int64, device="cuda")

    for segment in layout.segments:
        quantize_pack(
            segment.tensor,
            source_ids,
            packed,
            segment.payload_offset,
            segment.scale_offset,
        )

    outputs = {name: torch.zeros_like(tensor) for name, tensor in caches.items()}
    destination_ids = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    for segment in layout.segments:
        unpack_dequantize(
            packed,
            destination_ids,
            outputs[segment.name],
            segment.payload_offset,
            segment.scale_offset,
        )
    torch.cuda.synchronize()

    for segment in layout.segments:
        expected = segment.tensor[source_ids].float()
        actual = outputs[segment.name][destination_ids].float()
        scale_bytes = packed[:, segment.scale_offset : segment.scale_offset + 4]
        scales = scale_bytes.contiguous().view(torch.float32).flatten()
        error = (actual - expected).abs().amax(dim=1)
        assert torch.all(error <= scales * 0.6 + 0.02)
