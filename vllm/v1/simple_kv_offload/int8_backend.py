# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Double-buffered INT8 transport backend for AgentKV offload."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

_BUFFER_ALIGNMENT = 16
_SCALE_BYTES = 4


def _align_up(value: int, alignment: int = _BUFFER_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Int8PackedSegment:
    """One cache tensor's location in a packed transfer row."""

    name: str
    tensor: torch.Tensor
    payload_offset: int
    scale_offset: int


@dataclass(frozen=True)
class Int8PackedLayout:
    """Fixed packed-row layout shared by staging and host cache buffers."""

    segments: tuple[Int8PackedSegment, ...]
    row_bytes: int
    original_row_bytes: int
    num_blocks: int

    @property
    def compression_ratio(self) -> float:
        return self.row_bytes / self.original_row_bytes


def build_int8_packed_layout(
    gpu_caches: dict[str, torch.Tensor],
) -> Int8PackedLayout:
    """Build scale-header and payload offsets for two-dimensional caches."""
    if not gpu_caches:
        raise ValueError("INT8 offload requires at least one KV cache tensor")

    num_blocks: int | None = None
    original_row_bytes = 0
    payload_offset = _align_up(len(gpu_caches) * _SCALE_BYTES)
    segments: list[Int8PackedSegment] = []
    for scale_index, (name, tensor) in enumerate(gpu_caches.items()):
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "INT8 offload supports only FP16 or BF16 KV cache tensors; "
                f"{name!r} uses {tensor.dtype}"
            )
        if tensor.ndim != 2 or not tensor.is_contiguous():
            raise ValueError(
                f"INT8 offload cache {name!r} must be contiguous and two-dimensional"
            )
        if num_blocks is None:
            num_blocks = tensor.shape[0]
        elif tensor.shape[0] != num_blocks:
            raise ValueError("INT8 offload cache tensors must share a block count")

        elements_per_block = tensor.shape[1]
        segments.append(
            Int8PackedSegment(
                name=name,
                tensor=tensor,
                payload_offset=payload_offset,
                scale_offset=scale_index * _SCALE_BYTES,
            )
        )
        payload_offset += elements_per_block
        original_row_bytes += elements_per_block * tensor.element_size()

    assert num_blocks is not None
    return Int8PackedLayout(
        segments=tuple(segments),
        row_bytes=_align_up(payload_offset),
        original_row_bytes=original_row_bytes,
        num_blocks=num_blocks,
    )


def int8_kernels_available() -> bool:
    """Return whether the NVIDIA-only custom operators were loaded."""
    return hasattr(torch.ops._C, "agent_kv_int8_quantize_pack") and hasattr(
        torch.ops._C, "agent_kv_int8_unpack_dequantize"
    )


def quantize_pack(
    input_tensor: torch.Tensor,
    block_ids: torch.Tensor,
    packed_buffer: torch.Tensor,
    payload_offset: int,
    scale_offset: int,
) -> None:
    """Quantize selected cache blocks into packed transfer rows."""
    torch.ops._C.agent_kv_int8_quantize_pack(
        input_tensor,
        block_ids,
        packed_buffer,
        payload_offset,
        scale_offset,
    )


def unpack_dequantize(
    packed_buffer: torch.Tensor,
    block_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    payload_offset: int,
    scale_offset: int,
) -> None:
    """Unpack transfer rows into selected cache blocks."""
    torch.ops._C.agent_kv_int8_unpack_dequantize(
        packed_buffer,
        block_ids,
        output_tensor,
        payload_offset,
        scale_offset,
    )


@dataclass
class _PipelineSlot:
    packed: torch.Tensor
    host_block_ids: torch.Tensor
    device_block_ids: torch.Tensor
    copy_params: BatchMemcpyParams
    busy_event: torch.Event | None = None


class Int8DoubleBufferBackend:
    """Overlap INT8 conversion and host transfers with two slots per direction."""

    def __init__(self, buffer_blocks: int = 64) -> None:
        if buffer_blocks < 1:
            raise ValueError("INT8 pipeline buffer_blocks must be positive")
        self.buffer_blocks = buffer_blocks
        self.layout: Int8PackedLayout | None = None
        self.cpu_cache: torch.Tensor | None = None
        self._store_slots: list[_PipelineSlot] = []
        self._load_slots: list[_PipelineSlot] = []
        self._store_slot_index = 0
        self._load_slot_index = 0
        self._quantize_stream: torch.cuda.Stream | None = None
        self._dequantize_stream: torch.cuda.Stream | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        num_cpu_blocks: int,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        if not current_platform.is_cuda() or not int8_kernels_available():
            raise RuntimeError(
                "INT8 KV offload requires an NVIDIA CUDA build with AgentKV kernels"
            )
        if num_cpu_blocks < 1:
            raise ValueError("INT8 offload requires at least one CPU cache block")

        self.layout = build_int8_packed_layout(gpu_caches)
        if self.layout.row_bytes >= self.layout.original_row_bytes:
            raise ValueError("INT8 packing does not reduce this KV block layout")

        self._load_stream = load_stream
        self._store_stream = store_stream
        low_priority, _ = torch.cuda.Stream.priority_range()
        self._quantize_stream = torch.cuda.Stream(priority=low_priority)
        self._dequantize_stream = torch.cuda.Stream(priority=low_priority)

        self.cpu_cache = torch.empty(
            (num_cpu_blocks, self.layout.row_bytes),
            dtype=torch.uint8,
            device="cpu",
        )
        pin_tensor(self.cpu_cache)
        self._store_slots = self._make_slots(
            device,
            self.cpu_cache,
            store_stream,
            is_store=True,
        )
        self._load_slots = self._make_slots(
            device,
            self.cpu_cache,
            load_stream,
            is_store=False,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device),
            daemon=True,
        )
        self._thread.start()

    def _make_slots(
        self,
        device: torch.device,
        cpu_cache: torch.Tensor,
        stream: torch.cuda.Stream,
        *,
        is_store: bool,
    ) -> list[_PipelineSlot]:
        assert self.layout is not None
        slots = []
        for _ in range(2):
            packed = torch.empty(
                (self.buffer_blocks, self.layout.row_bytes),
                dtype=torch.uint8,
                device=device,
            )
            host_block_ids = torch.empty(
                self.buffer_blocks,
                dtype=torch.int64,
                device="cpu",
            )
            pin_tensor(host_block_ids)
            device_block_ids = torch.empty(
                self.buffer_blocks,
                dtype=torch.int64,
                device=device,
            )
            source = {"packed": packed if is_store else cpu_cache}
            destination = {"packed": cpu_cache if is_store else packed}
            params = build_params(
                source,
                destination,
                stream,
                src_access_order=(
                    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
                    if is_store
                    else CU_MEMCPY_SRC_ACCESS_ORDER_ANY
                ),
            )
            slots.append(
                _PipelineSlot(
                    packed=packed,
                    host_block_ids=host_block_ids,
                    device_block_ids=device_block_ids,
                    copy_params=params,
                )
            )
        return slots

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        if len(src_blocks) != len(dst_blocks):
            raise ValueError("INT8 offload source and destination lengths differ")
        if not src_blocks:
            return
        if self._queue is None:
            raise RuntimeError("INT8 offload backend is not initialized")
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _copy_loop(self, work_queue: queue.SimpleQueue, device: torch.device) -> None:
        current_platform.set_device(device)
        while True:
            item = work_queue.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            if is_store:
                event = self._store(src_blocks, dst_blocks, wait_event)
            else:
                event = self._load(src_blocks, dst_blocks)
            events_list.append((event_idx, event))

    def _next_slot(self, *, is_store: bool) -> _PipelineSlot:
        if is_store:
            slot = self._store_slots[self._store_slot_index % 2]
            self._store_slot_index += 1
        else:
            slot = self._load_slots[self._load_slot_index % 2]
            self._load_slot_index += 1
        if slot.busy_event is not None:
            slot.busy_event.synchronize()
        return slot

    @staticmethod
    def _stage_block_ids(slot: _PipelineSlot, block_ids: list[int]) -> torch.Tensor:
        count = len(block_ids)
        slot.host_block_ids[:count].copy_(torch.tensor(block_ids, dtype=torch.int64))
        slot.device_block_ids[:count].copy_(
            slot.host_block_ids[:count], non_blocking=True
        )
        return slot.device_block_ids[:count]

    def _store(
        self,
        gpu_blocks: list[int],
        cpu_blocks: list[int],
        wait_event: torch.Event | None,
    ) -> torch.Event:
        assert self.layout is not None
        assert self._quantize_stream is not None
        assert self._store_stream is not None
        final_event: torch.Event | None = None
        for start in range(0, len(gpu_blocks), self.buffer_blocks):
            gpu_chunk = gpu_blocks[start : start + self.buffer_blocks]
            cpu_chunk = cpu_blocks[start : start + self.buffer_blocks]
            slot = self._next_slot(is_store=True)
            with torch.cuda.stream(self._quantize_stream):
                if wait_event is not None:
                    self._quantize_stream.wait_event(wait_event)
                block_ids = self._stage_block_ids(slot, gpu_chunk)
                for segment in self.layout.segments:
                    quantize_pack(
                        segment.tensor,
                        block_ids,
                        slot.packed,
                        segment.payload_offset,
                        segment.scale_offset,
                    )
                quantized = torch.Event()
                quantized.record(self._quantize_stream)

            with torch.cuda.stream(self._store_stream):
                self._store_stream.wait_event(quantized)
                copy_blocks(list(range(len(gpu_chunk))), cpu_chunk, slot.copy_params)
                final_event = torch.Event()
                final_event.record(self._store_stream)
            slot.busy_event = final_event

        assert final_event is not None
        return final_event

    def _load(self, cpu_blocks: list[int], gpu_blocks: list[int]) -> torch.Event:
        assert self.layout is not None
        assert self._load_stream is not None
        assert self._dequantize_stream is not None
        final_event: torch.Event | None = None
        for start in range(0, len(cpu_blocks), self.buffer_blocks):
            cpu_chunk = cpu_blocks[start : start + self.buffer_blocks]
            gpu_chunk = gpu_blocks[start : start + self.buffer_blocks]
            slot = self._next_slot(is_store=False)
            with torch.cuda.stream(self._load_stream):
                copy_blocks(cpu_chunk, list(range(len(cpu_chunk))), slot.copy_params)
                copied = torch.Event()
                copied.record(self._load_stream)

            with torch.cuda.stream(self._dequantize_stream):
                self._dequantize_stream.wait_event(copied)
                block_ids = self._stage_block_ids(slot, gpu_chunk)
                for segment in self.layout.segments:
                    unpack_dequantize(
                        slot.packed,
                        block_ids,
                        segment.tensor,
                        segment.payload_offset,
                        segment.scale_offset,
                    )
                final_event = torch.Event()
                final_event.record(self._dequantize_stream)
            slot.busy_event = final_event

        assert final_event is not None
        return final_event
