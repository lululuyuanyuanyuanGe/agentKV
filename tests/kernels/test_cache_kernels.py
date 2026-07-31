# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

from vllm.platforms import current_platform

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    batch_size = 1
    block_size = 64
    entry_size = 128

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calling the C++ function gather_and_maybe_dequant_cache
    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.accelerator.synchronize()
    assert True


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Need CUDA device")
def test_batched_partial_block_copy_preserves_private_tail():
    block_size = 16
    num_heads = 4
    head_size = 16
    num_copies = 5
    cache = torch.full(
        (num_copies * 2, 2, block_size, num_heads, head_size),
        -1,
        dtype=torch.float16,
        device="cuda",
    )
    for block_id in range(num_copies):
        cache[block_id].copy_(torch.randn_like(cache[block_id]))

    valid_tokens = [1, 4, 7, 11, 15]
    mapping = torch.tensor(
        [(src, src + num_copies, valid) for src, valid in enumerate(valid_tokens)],
        dtype=torch.int64,
        device="cuda",
    )
    element_size = cache.element_size()
    segment_addresses = torch.tensor(
        [
            cache.data_ptr() + kv_index * cache.stride(1) * element_size
            for kv_index in range(2)
        ],
        dtype=torch.int64,
        device="cuda",
    )

    ops.batched_partial_block_copy(
        cache,
        segment_addresses,
        mapping,
        cache.stride(0) * element_size,
        cache.stride(2) * element_size,
        block_size,
    )
    torch.accelerator.synchronize()

    for src, valid in enumerate(valid_tokens):
        dst = src + num_copies
        torch.testing.assert_close(cache[dst, :, :valid], cache[src, :, :valid])
        assert torch.all(cache[dst, :, valid:] == -1)


if __name__ == "__main__":
    pytest.main([__file__])
