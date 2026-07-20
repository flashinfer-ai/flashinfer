# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/distributed/pcie_twoshot.py @ 7bfc9455 (2026-07-04) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""PCIe two-shot sequence-parallel collectives with fp8 transport.

Pull-based reduce_scatter / all_gather over IPC-mapped peer slabs for
TP sequence parallelism: values are quantized exactly once at the
source (per-token e4m3 scales), moved as fp8, and dequantized fused
with the fp32 reduction (RS) or the bf16 store (AG). Staging goes
through alternating eager slots so both ops are CUDA-graph-capturable;
a runtime instance must not be shared concurrently across CUDA streams.
"""

from __future__ import annotations

import os
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    _align_up,
    _broadcast_gather_object,
    _normalize_device,
    _OwnedSharedBuffer,
)

SUPPORTED_WORLD_SIZES = (2, 4, 8)
FP8_MAX = 448.0


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_twoshot.cu")
    verbose = os.getenv("FLASHINFER_EXP_SM12X_PCIE_TWOSHOT_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sm12x_pcie_twoshot_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row e4m3 quantization (tests / non-fused callers)."""
    assert x.dim() == 2
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / FP8_MAX
    payload = (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return payload, scale.squeeze(-1).contiguous()


class PCIeTwoShotSP:
    """Two-shot fp8-transport reduce_scatter / all_gather runtime."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        ext_module,
        fptr: int,
        owned_buffers: Sequence[_OwnedSharedBuffer],
        ipc: CudaRTLibrary,
        max_rows: int,
        row_elems: int,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._ext = ext_module
        self._fptr = fptr
        self._owned_buffers = list(owned_buffers)
        self._ipc = ipc
        self.max_rows = max_rows
        self.row_elems = row_elems
        self._closed = False

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        row_elems: int,
        ext_module=None,
    ) -> "PCIeTwoShotSP":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if row_elems % 16 != 0:
            raise ValueError("row_elems must be a multiple of 16")
        if max_rows % world_size != 0:
            raise ValueError("max_rows must be divisible by world size")

        device_obj = _normalize_device(device)
        if device_obj.type != "cuda":
            raise ValueError("PCIe twoshot requires a CUDA device")

        ipc = CudaRTLibrary()
        ipc.cudaSetDevice(device_obj.index or 0)
        ext = ext_module or _load_extension()

        # Per-slot staging: [world][pack_stride] Fp8Packs then
        # [world][scale_stride] fp32 scales, regions 256B-aligned.
        max_rows_per_rank = max_rows // world_size
        packs_per_row = row_elems // 16
        pack_stride = _align_up(max_rows_per_rank * packs_per_row, 16)
        payload_bytes = world_size * pack_stride * 16
        scale_offset = _align_up(payload_bytes, IPC_SLAB_ALIGNMENT)
        scale_stride = _align_up(max_rows_per_rank, 64)
        slot_bytes = _align_up(
            scale_offset + world_size * scale_stride * 4, IPC_SLAB_ALIGNMENT
        )
        signal_bytes = _align_up(int(ext.meta_size()), IPC_SLAB_ALIGNMENT)
        slab_bytes = signal_bytes + 2 * slot_bytes

        local_ptr = ipc.cudaMalloc(slab_bytes)
        owned: list[_OwnedSharedBuffer] = []
        try:
            # Signals must start zeroed.
            ipc.cudaMemset(local_ptr, 0, signal_bytes)
            local_handle = ipc.cudaIpcGetMemHandleBytes(local_ptr)
            handles = _broadcast_gather_object(local_handle, exchange_group)

            peer_ptrs: list[int] = []
            remote_ptrs: list[int] = []
            for idx, handle in enumerate(handles):
                if idx == rank:
                    peer_ptrs.append(local_ptr)
                else:
                    remote_ptr = ipc.cudaIpcOpenMemHandleBytes(handle)
                    peer_ptrs.append(remote_ptr)
                    remote_ptrs.append(remote_ptr)
            owned.append(
                _OwnedSharedBuffer(
                    local_ptr=local_ptr,
                    peer_ptrs=tuple(peer_ptrs),
                    remote_ptrs=tuple(remote_ptrs),
                )
            )

            signal_ptrs = [p for p in peer_ptrs]
            staging0 = [p + signal_bytes for p in peer_ptrs]
            staging1 = [p + signal_bytes + slot_bytes for p in peer_ptrs]

            fptr = ext.init_twoshot(
                signal_ptrs,
                staging0,
                staging1,
                pack_stride,
                scale_offset,
                scale_stride,
                rank,
            )
            return cls(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                ext_module=ext,
                fptr=fptr,
                owned_buffers=owned,
                ipc=ipc,
                max_rows=max_rows,
                row_elems=row_elems,
            )
        except Exception:
            for shared in owned:
                for ptr in shared.remote_ptrs:
                    with suppress(Exception):
                        ipc.cudaIpcCloseMemHandle(ptr)
            with suppress(Exception):
                ipc.cudaFree(local_ptr)
            raise

    def _check(self, payload: torch.Tensor, scale: torch.Tensor, rows: int) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotSP is closed")
        if payload.shape != (rows, self.row_elems):
            raise ValueError(
                f"payload shape {tuple(payload.shape)} != ({rows}, {self.row_elems})"
            )
        if scale.numel() != rows:
            raise ValueError(f"scale numel {scale.numel()} != {rows}")

    def reduce_scatter_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Sum per-token-quantized partials; return the local row shard."""
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if rows % self.world_size != 0:
            raise ValueError("rows must be divisible by world size")
        if out is None:
            out = torch.empty(
                rows // self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        self._ext.reduce_scatter_fp8(
            self._fptr, payload, scale, out, threads, block_limit
        )
        return out

    def all_gather_fp8(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """Gather per-token-quantized shards; return bf16 full width."""
        rows = payload.shape[0]
        self._check(payload, scale, rows)
        if out is None:
            out = torch.empty(
                rows * self.world_size,
                self.row_elems,
                dtype=torch.bfloat16,
                device=self.device,
            )
        self._ext.all_gather_fp8(self._fptr, payload, scale, out, threads, block_limit)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self._ext.dispose(self._fptr)
        for shared in self._owned_buffers:
            for ptr in shared.remote_ptrs:
                with suppress(Exception):
                    self._ipc.cudaIpcCloseMemHandle(ptr)
            with suppress(Exception):
                self._ipc.cudaFree(shared.local_ptr)
        self._owned_buffers.clear()
