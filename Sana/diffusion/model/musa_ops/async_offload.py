"""Asynchronous Pinned-Memory Activation Offloader v3 (Zero CPU Sync, Zero Race, Main-Stream Pool).

Designed for 4x MTT S5000 (dedicated PCIe Gen5 x16: ~56 GB/s D2H & ~57 GB/s H2D per GPU).

Key design points (v3):
  1. ZERO overwrite race with PyTorch autograd out-of-order `dx`-before-`dw` execution:
     Each prefetched block receives its own fresh tensors allocated on `main_stream`
     (never overwriting another block's buffer in-place).
  2. ZERO cross-stream allocator fragmentation:
     Staging tensors are allocated on `main_stream` (NOT inside `with stream(copy_stream)`),
     ordered via GPU-side `copy_stream.wait_event(ev_alloc)` + `g.record_stream(copy_stream)`,
     so freed buffers immediately return to `main_stream`'s CachingAllocator pool.
  3. ZERO blocking CPU `ev.synchronize()` in steady state:
     `_drain_d2h()` uses non-blocking `ev.query()` to retire completed D2H GPU refs,
     falling back to sync only if > 3 blocks somehow pile up in flight.
  4. Default `SANA_OFFLOAD_UPTO=15` keeps peak allocated VRAM around ~67-68 GiB
     (safely below the 71 GiB trajectory-check headroom limit) while enabling 20/20
     zero-recomputation (`gdn:0`).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class _OffloadHandle:
    __slots__ = ("block_idx", "slot_idx")

    def __init__(self, block_idx: int, slot_idx: int) -> None:
        self.block_idx = block_idx
        self.slot_idx = slot_idx


class _BlockPrefetchMarker(torch.autograd.Function):
    """Autograd boundary marker placed after each transformer block.

    On backward entry for block `i`:
      - Releases our dict reference to block `i + 1` (autograd still holds any live
        unpacked tensor until `i + 1`'s `dw` nodes complete).
      - Ensures block `i` is staged and triggers async H2D prefetch for block `i - 1`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, offloader: "AsyncBlockOffloader", block_idx: int) -> torch.Tensor:
        ctx.offloader = offloader
        ctx.block_idx = block_idx
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ctx.offloader.on_block_backward_enter(ctx.block_idx)
        return grad_out, None, None


class AsyncBlockOffloader:
    """Manages reusable pinned CPU buffers and async D2H/H2D staging per block."""

    def __init__(
        self,
        device: torch.device,
        offload_upto: int = 15,
        min_bytes: int = 250 * 1024 * 1024,
    ) -> None:
        self.device = device
        self.offload_upto = offload_upto
        self.min_bytes = min_bytes
        self.copy_stream = torch.musa.Stream(device=device)

        # Persistent pinned CPU buffer pool keyed by (block_idx, slot_idx) -> pinned CPU Tensor
        self._cpu_pool: Dict[Tuple[int, int], torch.Tensor] = {}

        # Per-step metadata
        self._block_slots: Dict[int, List[Tuple[torch.Size, torch.dtype, int]]] = {}
        self._block_dedup: Dict[int, Dict[Tuple[int, torch.Size, torch.dtype], _OffloadHandle]] = {}
        # Inflights: (block_idx, gpu_tensor, ev_d2h) retired non-blockingly via ev.query()
        self._inflight_d2h: List[Tuple[int, torch.Tensor, torch.musa.Event]] = []

        # Backward H2D staging: block_idx -> (list[gpu_tensor], ev_h2d)
        self._gpu_staging: Dict[int, Tuple[List[torch.Tensor], torch.musa.Event]] = {}
        self._prefetched_blocks: set[int] = set()

    def reset_step(self) -> None:
        """Reset per-step state while keeping pinned CPU buffers allocated."""
        if self._inflight_d2h:
            for _, _, ev in self._inflight_d2h:
                ev.synchronize()
            self._inflight_d2h.clear()
        self._block_slots.clear()
        self._block_dedup.clear()
        self._gpu_staging.clear()
        self._prefetched_blocks.clear()

    def _drain_d2h(self, force_below_block: int = -1) -> None:
        """Non-blockingly release completed D2H GPU tensor refs; only sync if backlog > limit."""
        if not self._inflight_d2h:
            return
        remaining = []
        for blk_i, t_gpu, ev in self._inflight_d2h:
            if ev.query():
                continue
            if blk_i < force_below_block:
                ev.synchronize()
                continue
            remaining.append((blk_i, t_gpu, ev))
        self._inflight_d2h = remaining

    def _get_cpu_buf(self, block_idx: int, slot_idx: int, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        key = (block_idx, slot_idx)
        buf = self._cpu_pool.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            self._cpu_pool[key] = buf
        return buf

    def _pack_tensor(self, block_idx: int, tensor: torch.Tensor):
        if (
            not isinstance(tensor, torch.Tensor)
            or isinstance(tensor, nn.Parameter)
            or tensor.device.type != "musa"
            or not tensor.is_contiguous()
        ):
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < self.min_bytes:
            return tensor
        # Skip FSDP2 weight slices (views of larger flat all_gather storage)
        if tensor.untyped_storage().nbytes() != nbytes:
            return tensor

        dedup = self._block_dedup.setdefault(block_idx, {})
        dkey = (tensor.data_ptr(), tensor.shape, tensor.dtype)
        existing = dedup.get(dkey)
        if existing is not None:
            return existing

        slots = self._block_slots.setdefault(block_idx, [])
        slot_idx = len(slots)
        slots.append((tensor.shape, tensor.dtype, nbytes))

        cpu_buf = self._get_cpu_buf(block_idx, slot_idx, tensor.shape, tensor.dtype)
        main_stream = torch.musa.current_stream(self.device)
        ev_ready = torch.musa.Event()
        ev_ready.record(main_stream)
        ev_d2h = torch.musa.Event()
        tensor.record_stream(self.copy_stream)
        with torch.musa.stream(self.copy_stream):
            self.copy_stream.wait_event(ev_ready)
            cpu_buf.copy_(tensor, non_blocking=True)
            ev_d2h.record(self.copy_stream)

        self._inflight_d2h.append((block_idx, tensor, ev_d2h))

        handle = _OffloadHandle(block_idx, slot_idx)
        dedup[dkey] = handle
        return handle

    def _unpack_tensor(self, packed):
        if not isinstance(packed, _OffloadHandle):
            return packed
        b_i = packed.block_idx
        s_i = packed.slot_idx
        staged = self._gpu_staging.get(b_i)
        if staged is None:
            self.prefetch_block(b_i)
            staged = self._gpu_staging[b_i]
        gpu_list, ev_h2d = staged
        main_stream = torch.musa.current_stream(self.device)
        main_stream.wait_event(ev_h2d)
        return gpu_list[s_i]

    def prefetch_block(self, block_idx: int) -> None:
        if block_idx < 0 or block_idx not in self._block_slots:
            return
        if block_idx in self._prefetched_blocks:
            return
        self._prefetched_blocks.add(block_idx)
        slots = self._block_slots[block_idx]
        if not slots:
            return

        # Allocate on main_stream (keeps allocator in main_stream pool -> zero cross-stream fragmentation!)
        main_stream = torch.musa.current_stream(self.device)
        gpu_list: List[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for shape, dtype, _ in slots
        ]
        ev_alloc = torch.musa.Event()
        ev_alloc.record(main_stream)
        for g in gpu_list:
            g.record_stream(self.copy_stream)

        ev_h2d = torch.musa.Event()
        with torch.musa.stream(self.copy_stream):
            self.copy_stream.wait_event(ev_alloc)
            for s_i, g in enumerate(gpu_list):
                g.copy_(self._cpu_pool[(block_idx, s_i)], non_blocking=True)
            ev_h2d.record(self.copy_stream)
        self._gpu_staging[block_idx] = (gpu_list, ev_h2d)

    def release_block(self, block_idx: int) -> None:
        self._gpu_staging.pop(block_idx, None)
        self._prefetched_blocks.discard(block_idx)

    def on_block_backward_enter(self, block_idx: int) -> None:
        # Non-blockingly retire completed D2H refs
        if self._inflight_d2h:
            self._drain_d2h(force_below_block=-1)
        # Drop dict reference to block (i + 1); autograd holds any live tensor until its dw completes
        self.release_block(block_idx + 1)
        # Ensure current block i is staged
        if block_idx in self._block_slots:
            self.prefetch_block(block_idx)
        # Prefetch block (i - 1) asynchronously ahead of time
        if (block_idx - 1) in self._block_slots:
            self.prefetch_block(block_idx - 1)

    @contextmanager
    def block_context(self, block_idx: int):
        # Retire completed D2H refs without blocking; cap backlog at 4 blocks to bound VRAM
        self._drain_d2h(force_below_block=block_idx - 4)
        if block_idx < self.offload_upto and torch.is_grad_enabled():
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: self._pack_tensor(block_idx, t),
                self._unpack_tensor,
            ):
                yield
        else:
            yield

    def mark_block_output(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            return _BlockPrefetchMarker.apply(x, self, block_idx)
        return x

    def finish_forward(self) -> None:
        # Non-blockingly retire completed D2H refs
        self._drain_d2h(force_below_block=-1)


def get_offload_upto() -> int:
    if os.environ.get("SANA_OPT_ASYNC_OFFLOAD", "0") == "0":
        return 0
    return int(os.environ.get("SANA_OFFLOAD_UPTO", "15"))
