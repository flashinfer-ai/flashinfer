"""Ulysses sequence parallelism helpers for the Wan FlashInfer example.

Provides the head-scatter / sequence-gather all-to-all used by Ulysses
sequence parallelism over the head_dim==2 layout ``[B, S, H, D]`` that
``FlashInferWanAttention`` uses, with two interchangeable implementations:

- ``impl="flashinfer"``: the fused-transpose NVLink-P2P kernel
  (``flashinfer.comm.ulysses_a2a``, adapted from ThunderKittens'
  https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu).
  The layout permutation is folded into the cross-GPU writes, so no
  permute-contiguous glue is materialized.
- ``impl="nccl"``: the conventional ``dist.all_to_all_single`` path with
  explicit permute/contiguous glue before and after. Serves as the baseline
  and produces bit-identical results.

Usage (per rank, inside a torch.distributed process group):

    ctx = UlyssesContext(group, impl="flashinfer")
    set_ulysses_context(ctx)          # picked up by FlashInferWanAttention
    ...                               # run the sequence-sharded model
    ctx.shutdown()
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist

import flashinfer.comm as comm

_CTX: Optional["UlyssesContext"] = None


def set_ulysses_context(ctx: Optional["UlyssesContext"]) -> None:
    global _CTX
    _CTX = ctx


def get_ulysses_context() -> Optional["UlyssesContext"]:
    return _CTX


class UlyssesContext:
    """Owns the process group and (for the flashinfer impl) the IPC buffers.

    ``max_elems`` bounds the largest all-to-all operand (``B*S_global*H*D``
    elements of the widest dtype used); the staging buffer is allocated once.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        impl: str = "flashinfer",
        max_elems: int = 0,
        elem_bytes: int = 2,
        full_nvlink: bool = True,
    ):
        assert impl in ("flashinfer", "nccl"), impl
        self.group = group
        self.impl = impl
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._fa: Optional[int] = None
        self._out_ptrs: Optional[List[int]] = None
        self._sig_ptrs: Optional[List[int]] = None
        if impl == "flashinfer" and self.world_size > 1:
            assert max_elems > 0, "max_elems required for the flashinfer impl"
            self._out_ptrs = comm.create_shared_buffer(
                max_elems * elem_bytes, group=group
            )
            self._sig_ptrs = comm.create_shared_buffer(
                comm.vllm_meta_size(), group=group
            )
            self._fa = comm.init_ulysses_a2a(
                self._out_ptrs, self._sig_ptrs, self.rank, self.world_size, full_nvlink
            )
            # init zeroed this rank's signal; make it globally visible before use
            dist.barrier(group=group)

    def shutdown(self) -> None:
        if self._fa is not None:
            comm.dispose_ulysses_a2a(self._fa)
            self._fa = None
        if self._out_ptrs is not None:
            comm.free_shared_buffer(self._out_ptrs, self.group)
            self._out_ptrs = None
        if self._sig_ptrs is not None:
            comm.free_shared_buffer(self._sig_ptrs, self.group)
            self._sig_ptrs = None

    # ---- collective ops ------------------------------------------------

    def input_all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S_local, H, D] -> [B, S_global, H_local, D] (scatter heads, gather seq)."""
        if self.world_size == 1:
            return x
        b, s_local, h, d = x.shape
        w = self.world_size
        assert h % w == 0, f"heads {h} not divisible by world size {w}"
        h_local = h // w
        if self.impl == "flashinfer":
            out = torch.empty(b, s_local * w, h_local, d, dtype=x.dtype, device=x.device)
            comm.ulysses_a2a(self._fa, x.contiguous(), out, b, s_local, h, d, 0)
            return out
        # NCCL baseline: permute -> all_to_all_single -> permute (the glue the
        # fused kernel eliminates).
        xt = (
            x.reshape(b, s_local, w, h_local, d)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )  # [w, b, s_local, h_local, d]
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk j == rank j's contribution to my sequence block j
        return (
            recv.permute(1, 0, 2, 3, 4)
            .reshape(b, w * s_local, h_local, d)
            .contiguous()
        )

    def output_all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S_global, H_local, D] -> [B, S_local, H, D] (gather heads, scatter seq)."""
        if self.world_size == 1:
            return x
        b, s_global, h_local, d = x.shape
        w = self.world_size
        assert s_global % w == 0, f"seq {s_global} not divisible by world size {w}"
        s_local = s_global // w
        h = h_local * w
        if self.impl == "flashinfer":
            out = torch.empty(b, s_local, h, d, dtype=x.dtype, device=x.device)
            comm.ulysses_a2a(self._fa, x.contiguous(), out, b, s_local, h, d, 1)
            return out
        xt = (
            x.reshape(b, w, s_local, h_local, d)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )  # [w, b, s_local, h_local, d]
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk p == my sequence block's head-slice p
        return (
            recv.permute(1, 2, 0, 3, 4)
            .reshape(b, s_local, h, d)
            .contiguous()
        )
