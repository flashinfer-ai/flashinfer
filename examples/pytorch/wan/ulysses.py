"""Ulysses sequence parallelism helpers for the Wan FlashInfer example.

Provides the head-scatter / sequence-gather all-to-all used by Ulysses
sequence parallelism over the head_dim==2 layout ``[B, S, H, D]`` that
``FlashInferWanAttention`` uses, with two interchangeable implementations:

- ``impl="flashinfer"``: the public :class:`flashinfer.comm.UlyssesCommunicator`
  (fused-transpose NVLink-P2P kernel when the topology supports it, with
  automatic NCCL fallback otherwise; backend/fallback reason exposed on the
  context).
- ``impl="nccl"``: the conventional ``dist.all_to_all_single`` path with
  explicit permute/contiguous glue before and after. Serves as the baseline
  and produces bit-identical results.

Usage (per rank, inside a torch.distributed process group):

    ctx = UlyssesContext(group, impl="flashinfer", max_elems=B*S_local*H*D)
    set_ulysses_context(ctx)          # picked up by FlashInferWanAttention
    ...                               # run the sequence-sharded model
    ctx.shutdown()

``max_elems`` is the element count of the largest single all-to-all operand,
i.e. ``B * S_local * H * D`` (input and output have equal numel).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator

_CTX: Optional["UlyssesContext"] = None


def set_ulysses_context(ctx: Optional["UlyssesContext"]) -> None:
    global _CTX
    _CTX = ctx


def get_ulysses_context() -> Optional["UlyssesContext"]:
    return _CTX


class UlyssesContext:
    """Thin adapter between the Wan example attention and the Ulysses
    collectives; owns a :class:`UlyssesCommunicator` for the flashinfer impl.

    ``max_elems`` bounds the largest all-to-all operand: ``B*S_local*H*D``
    elements of ``dtype``.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        impl: str = "flashinfer",
        max_elems: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "auto",
    ):
        assert impl in ("flashinfer", "nccl"), impl
        self.group = group
        self.impl = impl
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._comm: Optional[UlyssesCommunicator] = None
        if impl == "flashinfer" and self.world_size > 1:
            assert max_elems > 0, "max_elems required for the flashinfer impl"
            self._comm = UlyssesCommunicator(
                group, max_elems=max_elems, dtype=dtype, backend=backend
            )

    @property
    def backend(self) -> str:
        return self._comm.backend if self._comm is not None else "nccl"

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._comm.fallback_reason if self._comm is not None else None

    def shutdown(self) -> None:
        if self._comm is not None:
            self._comm.close()
            self._comm = None

    # ---- collective ops ------------------------------------------------

    def input_all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S_local, H, D] -> [B, S_global, H_local, D] (scatter heads, gather seq)."""
        if self.world_size == 1:
            return x
        if self._comm is not None:
            return self._comm.scatter_heads(x.contiguous())
        # NCCL baseline: permute -> all_to_all_single -> permute (the glue the
        # fused kernel eliminates).
        b, s_local, h, d = x.shape
        w = self.world_size
        assert h % w == 0, f"heads {h} not divisible by world size {w}"
        h_local = h // w
        xt = x.reshape(b, s_local, w, h_local, d).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk j == rank j's contribution to my sequence block j
        return (
            recv.permute(1, 0, 2, 3, 4).reshape(b, w * s_local, h_local, d).contiguous()
        )

    def output_all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S_global, H_local, D] -> [B, S_local, H, D] (gather heads, scatter seq)."""
        if self.world_size == 1:
            return x
        if self._comm is not None:
            return self._comm.gather_heads(x.contiguous())
        b, s_global, h_local, d = x.shape
        w = self.world_size
        assert s_global % w == 0, f"seq {s_global} not divisible by world size {w}"
        s_local = s_global // w
        h = h_local * w
        xt = x.reshape(b, w, s_local, h_local, d).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        # chunk p == my sequence block's head-slice p
        return recv.permute(1, 2, 0, 3, 4).reshape(b, s_local, h, d).contiguous()
