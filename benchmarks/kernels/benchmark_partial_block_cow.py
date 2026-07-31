# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark batched partial-block COW against per-branch tensor copies."""

import argparse
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def _measure(operation, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not current_platform.is_cuda():
        raise RuntimeError("batched partial-block COW benchmark requires CUDA")

    num_blocks = args.batch_size * 2
    caches = [
        torch.randn(
            (
                num_blocks,
                2,
                args.block_size,
                args.num_kv_heads,
                args.head_size,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(args.num_layers)
    ]
    valid_tokens = torch.randint(1, args.block_size, (args.batch_size,))
    src_ids = torch.arange(args.batch_size)
    dst_ids = src_ids + args.batch_size
    mapping_host = (
        torch.stack((src_ids, dst_ids, valid_tokens), dim=1)
        .to(torch.int64)
        .pin_memory()
    )
    mapping = torch.empty_like(mapping_host, device="cuda")

    first = caches[0]
    element_size = first.element_size()
    segment_addresses = torch.tensor(
        [
            cache.data_ptr() + kv * cache.stride(1) * element_size
            for cache in caches
            for kv in range(2)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    page_stride_bytes = first.stride(0) * element_size
    token_stride_bytes = first.stride(2) * element_size
    valid_tokens_cpu = valid_tokens.tolist()

    def batched() -> None:
        mapping.copy_(mapping_host, non_blocking=True)
        ops.batched_partial_block_copy(
            first,
            segment_addresses,
            mapping,
            page_stride_bytes,
            token_stride_bytes,
            args.block_size,
        )

    def per_branch() -> None:
        for cache in caches:
            for src, valid in enumerate(valid_tokens_cpu):
                cache[src + args.batch_size, :, :valid].copy_(cache[src, :, :valid])

    for _ in range(10):
        batched()
        per_branch()
    torch.accelerator.synchronize()

    batched_us = _measure(batched, args.repeats)
    baseline_us = _measure(per_branch, args.repeats)
    print(
        {
            "batch_size": args.batch_size,
            "layers": args.num_layers,
            "batched_median_us": statistics.median(batched_us),
            "per_branch_median_us": statistics.median(baseline_us),
            "speedup": statistics.median(baseline_us) / statistics.median(batched_us),
        }
    )


if __name__ == "__main__":
    main()
