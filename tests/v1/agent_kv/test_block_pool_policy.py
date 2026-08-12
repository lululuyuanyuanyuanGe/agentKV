# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.core.block_pool import BlockPool, FreeCachedBlockSnapshot
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


@dataclass(frozen=True)
class Target:
    block_id: int
    expected_cache_keys: tuple[bytes, ...]


def make_cached_free_pool() -> tuple[BlockPool, list[bytes]]:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    hashes = [f"hash-{i}".encode() for i in range(3)]
    for block, block_hash in zip(pool.blocks[1:], hashes):
        cache_key = make_block_hash_with_group_id(BlockHash(block_hash), 0)
        pool._insert_block_hash(cache_key, block, num_tokens=16)
    return pool, hashes


def test_allocator_uses_planner_only_when_cached_blocks_are_at_risk() -> None:
    pool, hashes = make_cached_free_pool()
    calls: list[tuple[tuple[FreeCachedBlockSnapshot, ...], int]] = []

    def planner(
        candidates: tuple[FreeCachedBlockSnapshot, ...],
        num_at_risk_blocks: int,
    ) -> tuple[Target, ...]:
        calls.append((candidates, num_at_risk_blocks))
        selected = candidates[-1]
        return (Target(selected.block_id, selected.cache_keys),)

    pool.set_free_cached_block_eviction_planner(planner, lambda: True)

    allocated = pool.get_new_blocks(1)

    assert calls and calls[0][1] == 1
    assert allocated[0].block_id == 3
    assert pool.get_cached_block(BlockHash(hashes[0]), [0]) is not None


def test_disabled_planner_preserves_native_eviction_order() -> None:
    pool, _ = make_cached_free_pool()
    pool.set_free_cached_block_eviction_planner(
        lambda candidates, count: (),
        lambda: False,
    )

    allocated = pool.get_new_blocks(1)

    assert allocated[0].block_id == 1


def test_planner_is_not_called_when_all_allocated_blocks_are_uncached() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    called = False

    def planner(
        candidates: tuple[FreeCachedBlockSnapshot, ...],
        num_at_risk_blocks: int,
    ) -> tuple[Target, ...]:
        nonlocal called
        called = True
        return ()

    pool.set_free_cached_block_eviction_planner(planner, lambda: True)

    allocated = pool.get_new_blocks(1)

    assert allocated[0].block_id == 1
    assert not called


def test_planner_failure_falls_back_to_native_eviction_order() -> None:
    pool, _ = make_cached_free_pool()

    def planner(
        candidates: tuple[FreeCachedBlockSnapshot, ...],
        num_at_risk_blocks: int,
    ) -> tuple[Target, ...]:
        raise RuntimeError("policy unavailable")

    pool.set_free_cached_block_eviction_planner(planner, lambda: True)

    allocated = pool.get_new_blocks(1)

    assert allocated[0].block_id == 1


def test_stale_target_is_ignored_after_cache_identity_changes() -> None:
    pool, _ = make_cached_free_pool()
    snapshot = pool.get_free_cached_block_snapshots()[0]
    block = pool.blocks[snapshot.block_id]
    pool._maybe_evict_cached_block(block)

    moved = pool.prioritize_free_cached_blocks_for_eviction(
        [Target(snapshot.block_id, snapshot.cache_keys)]
    )

    assert moved == 0
    assert pool.free_block_queue.get_all_free_blocks()[0].block_id == 1
