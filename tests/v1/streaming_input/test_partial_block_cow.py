# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, BatchedPartialBlockCopyManager

pytestmark = pytest.mark.cpu_test


class _FakeBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 0


def _build_manager(cache: torch.Tensor) -> BatchedPartialBlockCopyManager:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
    )
    group = AttentionGroup(
        backend=_FakeBackend,  # type: ignore[arg-type]
        layer_names=["layer"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    return BatchedPartialBlockCopyManager(
        device=torch.device("cpu"),
        pin_memory=False,
        attn_groups_iter=[group],
        kernel_block_sizes=[16],
        cache_dtype="auto",
        static_forward_context={"layer": SimpleNamespace(kv_cache=cache)},
    )


def test_nhd_layout_collapses_heads_into_two_kv_segments() -> None:
    cache = torch.empty((8, 2, 16, 4, 8), dtype=torch.float32)
    manager = _build_manager(cache)

    [layout] = manager._layouts[0]
    assert layout.segment_addresses.numel() == 2
    assert layout.page_stride_bytes == cache.stride(0) * cache.element_size()
    assert layout.token_stride_bytes == cache.stride(2) * cache.element_size()


def test_hnd_layout_uses_one_segment_per_kv_head() -> None:
    physical = torch.empty((8, 2, 4, 16, 8), dtype=torch.float32)
    cache = physical.permute(0, 1, 3, 2, 4)
    manager = _build_manager(cache)

    [layout] = manager._layouts[0]
    assert layout.segment_addresses.numel() == 8
    assert layout.page_stride_bytes == cache.stride(0) * cache.element_size()
    assert layout.token_stride_bytes == cache.stride(2) * cache.element_size()
