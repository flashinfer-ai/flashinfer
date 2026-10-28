"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fused QKV projection + per-head RMSNorm + RoPE + Ulysses head exchange.

.. code-block:: python

    from flashinfer.comm import QkvNormRopeExchangeOverlap

    session = QkvNormRopeExchangeOverlap(
        group, hidden=8192, num_heads=64, head_dim=128, layout="bshd",
        max_tokens=65536, backend="auto",
    )
    q, k, v = session(x, w_qkv, q_norm_weight, k_norm_weight, cos_sin)
    # x [B, S_local, hidden] bf16; w_qkv [3 * hidden, hidden] bf16 (torch.nn.Linear
    # layout, rows ordered Q, K, V by head); norm weights [head_dim] bf16;
    # cos_sin [B * S_local, head_dim] fp32 (cos of pairs 0..63 then sin, interleaved RoPE);
    # q, k, v [B, S_global, H_local, head_dim] views of the current receive slot.
    session.close()
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed as dist

from .cake_qkv_exchange_plan import LAYOUTS, Geometry, heads_divisibility_error
from .cake_qkv_norm_rope_exchange_overlap import (
    CakeQkvExchangeSession,
    device_arch,
    supports_symmetric_memory_backend,
)
from .ulysses_topology import (
    ULYSSES_BACKENDS,
    UlyssesBackendError,
    resolve_ulysses_backend,
)

SUPPORTED_WORLD_SIZES = (2, 4, 8)
HEAD_DIM = 128
EPS = 1e-6


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"the fused QKV exchange needs a CUDA device, got {parsed}")
    if parsed.index is None:
        parsed = torch.device("cuda", torch.cuda.current_device())
    return parsed


def validate_request(
    *,
    world_size: int,
    hidden: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    layout: str,
    max_tokens: int,
    num_slots: int,
    backend: str,
    fold_barrier: Optional[bool] = None,
) -> None:
    """Static checks of a session request (no collective); raises ``ValueError`` on every rank alike."""

    if backend not in ULYSSES_BACKENDS:
        raise ValueError(f"backend must be one of {ULYSSES_BACKENDS}, got {backend!r}")
    if fold_barrier is not None and not isinstance(fold_barrier, bool):
        raise ValueError(f"fold_barrier must be a bool or None, got {fold_barrier!r}")
    if fold_barrier is not None and backend == "nccl":
        raise ValueError(
            "fold_barrier selects the launch path of the Cake (nvlink) backend; it has no meaning with "
            "backend='nccl'"
        )
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if dtype != torch.bfloat16:
        raise ValueError(f"only torch.bfloat16 is supported, got {dtype}")
    if int(head_dim) != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")
    if int(world_size) not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
    if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots < 2:
        raise ValueError(f"num_slots must be an int of at least 2, got {num_slots!r}")
    if int(num_heads) * int(head_dim) != int(hidden):
        raise ValueError(
            f"hidden={hidden} must equal num_heads * head_dim = {num_heads} * {head_dim}"
        )
    if int(num_heads) % int(world_size):
        raise ValueError(heads_divisibility_error(int(num_heads), int(world_size)))
    # Every other constraint (hidden % 64, positivity) is the plan geometry's.
    Geometry(
        world_size=int(world_size),
        batch=1,
        s_local=int(max_tokens),
        hidden=int(hidden),
        head_dim=int(head_dim),
    )


class _NcclQkvExchange:
    """The per-operator path: cuBLAS projection, ``rmsnorm`` per head, RoPE, then one head-chunk collective."""

    graph_capturable = False

    def __init__(
        self,
        group,
        *,
        device,
        hidden,
        num_heads,
        head_dim,
        layout,
        max_tokens,
        num_slots,
    ) -> None:
        from .ulysses import UlyssesCommunicator as _Communicator

        self.device = device
        self.num_slots = int(num_slots)
        self.hidden, self.num_heads, self.head_dim = (
            int(hidden),
            int(num_heads),
            int(head_dim),
        )
        self.layout = layout
        self.max_tokens = int(max_tokens)
        self.world_size = int(dist.get_world_size(group=group))
        self.rank = int(dist.get_rank(group=group))
        self.h_local = self.num_heads // self.world_size
        self._comm = _Communicator(
            group,
            max_bytes=self.max_tokens * 3 * self.hidden * torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
            backend="nccl",
            device=device,
        )
        norm_shape = (self.max_tokens, self.num_heads, self.head_dim)
        self._q_norm = torch.empty(norm_shape, dtype=torch.bfloat16, device=device)
        self._k_norm = torch.empty(norm_shape, dtype=torch.bfloat16, device=device)
        # ``num_slots`` receive regions cycled by consecutive calls, so a call's outputs
        # stay valid for ``num_slots - 1`` further calls (the same contract as the Cake path).
        self._slot_elems = (
            self.world_size * self.max_tokens * self.h_local * 3 * self.head_dim
        )
        self._payload = torch.empty(
            self.num_slots * self._slot_elems, dtype=torch.bfloat16, device=device
        )
        self._next_slot = 0
        self._positions: dict[int, torch.Tensor] = {}
        self._last = None

    def __call__(self, x, w_qkv, q_scale, k_scale, cos_sin):
        import torch.nn.functional as functional

        from ..norm import rmsnorm
        from ..rope import apply_rope_with_cos_sin_cache_inplace

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the NCCL path of the fused QKV exchange is not CUDA-graph capturable"
            )
        batch, s_local = int(x.shape[0]), int(x.shape[1])
        rows = batch * s_local
        if rows > self.max_tokens:
            raise ValueError(
                f"batch * s_local = {rows} exceeds the session's max_tokens={self.max_tokens}"
            )
        heads, head_dim, hidden = self.num_heads, self.head_dim, self.hidden
        qkv = functional.linear(x.reshape(rows, hidden), w_qkv)
        q_view = qkv[:, :hidden].view(rows, heads, head_dim)
        k_view = qkv[:, hidden : 2 * hidden].view(rows, heads, head_dim)
        v_view = qkv[:, 2 * hidden :].view(rows, heads, head_dim)
        q_norm = rmsnorm(q_view, q_scale, eps=EPS, out=self._q_norm[:rows])
        k_norm = rmsnorm(k_view, k_scale, eps=EPS, out=self._k_norm[:rows])
        positions = self._positions.get(rows)
        if positions is None:
            positions = torch.arange(rows, device=self.device, dtype=torch.int32)
            self._positions[rows] = positions
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            q_norm.view(rows, heads * head_dim),
            k_norm.view(rows, heads * head_dim),
            head_dim,
            cos_sin,
            is_neox=False,
        )
        local = (batch, s_local, heads, head_dim)
        s_global = s_local * self.world_size
        numel = batch * s_global * self.h_local * 3 * head_dim
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.num_slots
        base = slot * self._slot_elems
        payload = self._payload[base : base + numel].view(
            batch, s_global, self.h_local, 3 * head_dim
        )
        self._comm.scatter_qkv_head_chunk(
            q_norm.view(local),
            k_norm.view(local),
            v_view.view(local),
            head_offset=0,
            head_count=self.h_local,
            out=payload,
        )
        self._last = payload
        return payload

    def last_outputs(self):
        if self._last is None:
            raise RuntimeError("no call has run on this session yet")
        payload = self._last
        if self.layout == "bshd":
            q, k, v = payload.split(self.head_dim, dim=-1)
            return (q, k, v)
        return payload

    def poison_receive_slots(self) -> None:
        self._payload.view(torch.int16).fill_(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "nccl",
            "world_size": self.world_size,
            "rank": self.rank,
            "layout": self.layout,
            "hidden": self.hidden,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "max_tokens": self.max_tokens,
            "communicator": "UlyssesCommunicator(backend='nccl').scatter_qkv_head_chunk",
        }

    def close(self) -> None:
        if self._comm is not None:
            self._comm.close()
            self._comm = None
        self._last = None


class QkvNormRopeExchangeOverlap:
    r"""Prepared session of the fused QKV projection + norm + RoPE + head exchange on one process group.

    Every rank constructs the session collectively (same arguments, same order) and calls it with its
    own ``[B, S_local, hidden]`` activation shard; each rank receives head band
    ``[rank * H_local, (rank + 1) * H_local)`` of the whole sequence.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup, optional
        NCCL process group of 2, 4 or 8 ranks on one node (``None`` uses ``WORLD``).
    hidden : int
        Model width ``num_heads * head_dim`` (a multiple of 64).
    num_heads : int
        Attention heads; must be divisible by the world size.
    head_dim : int, default 128
        Head dimension (the fused kernels fix 128).
    dtype : torch.dtype, default torch.bfloat16
        Operand dtype (BF16 only).
    layout : {"bshd", "fused"}
        ``"bshd"``: ``(q, k, v)`` each ``[B, S_global, H_local, head_dim]``;
        ``"fused"``: one ``[B, S_global, H_local, 3 * head_dim]`` payload.
    max_tokens : int
        Largest ``B * S_local`` of any call; sizes the symmetric receive slots once (collective).
    num_slots : int, default 2
        Receive slots cycled by consecutive calls; a call's outputs stay valid for ``num_slots - 1`` further calls.
    backend : {"auto", "nvlink", "nccl"}
        ``"nvlink"`` selects the Cake fused kernels (single-node NVLink mesh, SM90 / SM100 / SM103 / SM107, groups of
        2, 4 or 8 ranks (2 or 4 on SM107), torch's default CUDA symmetric-memory backend) and raises when they
        cannot be used; ``"nccl"`` selects the
        per-operator path (cuBLAS, ``rmsnorm``, RoPE, ``UlyssesCommunicator.scatter_qkv_head_chunk``);
        ``"auto"`` resolves through :func:`resolve_ulysses_backend` and the device architecture.
    device : torch.device or str or int, optional
        CUDA device of this rank (``None``: the current device).
    fold_barrier : bool, optional
        Launch path of the Cake backend.  ``True``: the producer program with the folded cross-rank
        rendezvous -- two launches per call (producer, wait kernel); ``False``: the separate barrier kernel
        before the producer -- three launches.  ``None`` (default) reads the environment variable
        ``FLASHINFER_QKV_EXCHANGE_FOLD_BARRIER`` (``0`` / ``1`` / ``auto``) and otherwise takes the
        architecture default: ``True`` on SM90, SM100, SM103 and SM107 (the folded producers are delivered
        on every architecture).  Both paths store identical bits.  Must be uniform across the ranks of the group; not accepted with ``backend="nccl"``.

    Notes
    -----
    The Cake backend selects its producer build per architecture and world size
    (``flashinfer.jit.cake_qkv_norm_rope_exchange_overlap.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``: SM100 groups of
    8 ranks run the ``bulk_ow256`` build -- wide operand loads and bulk row stores --, SM100 groups of 4 ranks and
    SM103 groups of 8 or 4 ranks the ``ow256`` build -- wide operand loads --, SM107 groups of 4 ranks the ``bulk``
    build -- bulk row stores --, groups of 2 ranks and SM90 the baseline build).  Every build stores identical bits;
    the wide-load builds require 32-byte-aligned ``cos_sin`` and norm-weight tensors (a misaligned view raises and
    names the opt-out).  The environment variable ``FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT`` (``auto`` /
    ``shipped`` / a delivered build name) overrides the selection for a session, rank-uniformly; the baseline build is
    delivered beside every variant, and a build that is not delivered for the session's tile width raises and names
    the delivered ones.  The host plan's group scaling follows the same keying
    (``cake_qkv_exchange_plan.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD``: ``2`` on SM100 groups of 4 and 2 ranks and on
    SM103, ``1`` otherwise -- SM100 groups of 8 ranks, SM90 and SM107; it applies to grouped rows with
    ``hidden >= 6144`` and changes the tile order only); ``FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE`` (``auto`` / ``1`` /
    ``2`` / ``4``) overrides it for a session, rank-uniformly.

    The Cake backend also applies per-architecture row-count gates to each call's host plan
    (``cake_qkv_exchange_plan.SHAPE_GATE_DEFAULT_BY_ARCH``; SM100 / SM103: a call with ``batch * s_local <= 4096``
    rows publishes one readiness word per destination, a call with ``>= 16384`` rows launches the producer once per
    destination run; SM90: a call with ``<= 4096`` rows publishes one readiness word per destination and a call
    with ``>= 16384`` rows runs the persistent producer grid over every SM -- ``reserved_sms_large_m = 16384``, no
    reserved SMs --, the per-destination launch gate off; SM107: every gate off).  One (architecture, world size)
    cell differs from these tables (``cake_qkv_exchange_plan.SHAPE_GATE_DEFAULT_BY_ARCH_WORLD``): a group of 8 ranks
    on SM90 publishes one readiness word per destination for every row count (``words_capacity_small_m = all``).  A
    fourth gate, ``pair_off_large_m``, is off on every architecture: switched on, a 256-column call with at least
    ``N`` rows per rank runs the single-CTA producer instead of the 2-CTA pair tile (tile order unchanged; that
    producer must be delivered for the architecture -- no delivered Blackwell route ships it today, so the session
    refuses at construction when a 256-column call can reach the gate, ``max_tokens >= N``, naming the gate; SM90
    builds no pair tile and refuses the gate).  The environment variable
    ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` (``auto`` / ``off`` /
    ``words_capacity_small_m=N|all,reserved_sms_large_m=N,per_dest_large_m=N,pair_off_large_m=N``, ``0`` = never)
    overrides them for a session, rank-uniformly (``words_capacity_small_m=4096`` on SM90 with 8 ranks restores the
    previous plan).
    ``describe()`` reports the resolved build (``producer_variant``), scale
    (``group_m_scale``) and gates (``shape_gates``), each with its value, source, environment variable and default,
    and every prepared row's verdict (``prepared_row_plans``, including ``reserved_sms`` and ``part_grids``).  The
    first three gates change the host plan only -- neither the tile table nor the stored bits; the pair-tile gate
    changes the producer program and the tile table (no lockstep pads) of the rows it selects; no gate changes the
    stored bits.  The largest 2-rank shape of the evaluation grid
    (``S_global = 131072``, ``hidden = 8192``) is not supported on an 80 GB H100: its operands and receive slots
    exceed the device memory.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        *,
        hidden: int,
        num_heads: int,
        head_dim: int = HEAD_DIM,
        dtype: torch.dtype = torch.bfloat16,
        layout: str = "bshd",
        max_tokens: int,
        num_slots: int = 2,
        backend: str = "auto",
        device: Optional[Union[torch.device, str, int]] = None,
        fold_barrier: Optional[bool] = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("an initialized NCCL process group is required")
        if group is None:
            group = dist.group.WORLD
        self.group = group
        self.world_size = int(dist.get_world_size(group=group))
        self.rank = int(dist.get_rank(group=group))
        validate_request(
            world_size=self.world_size,
            hidden=hidden,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            layout=layout,
            max_tokens=max_tokens,
            num_slots=num_slots,
            backend=backend,
            fold_barrier=fold_barrier,
        )
        self.device = _resolve_device(device)
        self.hidden, self.num_heads, self.head_dim = (
            int(hidden),
            int(num_heads),
            int(head_dim),
        )
        self.dtype = dtype
        self.layout = layout
        self.max_tokens = int(max_tokens)
        self.num_slots = int(num_slots)
        self.fold_barrier_requested = fold_barrier
        decision = resolve_ulysses_backend(
            backend, group, self.device
        )  # collective, rank-uniform
        chosen, reason = decision.backend, decision.reason
        if chosen == "nvlink":
            arch = device_arch(self.device)
            local_reason = None
            if arch is None:
                local_reason = f"device capability {tuple(torch.cuda.get_device_capability(self.device))} has no Cake programs"
            elif not supports_symmetric_memory_backend(self.device):
                local_reason = (
                    "torch symmetric-memory backend is not the default CUDA backend"
                )
            verdicts: list[Any] = [None] * self.world_size
            dist.all_gather_object(verdicts, local_reason, group=group)
            blockers = [item for item in verdicts if item is not None]
            if blockers:
                if backend == "nvlink":
                    raise UlyssesBackendError(
                        f"backend='nvlink' cannot be used: {blockers[0]}"
                    )
                chosen, reason = "nccl", f"fallback: {blockers[0]}"
        self.backend = chosen
        self.backend_reason = reason
        if chosen == "nvlink":
            self._impl: Any = CakeQkvExchangeSession(
                group,
                device=self.device,
                hidden=self.hidden,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                layout=layout,
                max_tokens=self.max_tokens,
                num_slots=self.num_slots,
                fold_barrier=fold_barrier,
            )
        else:
            self._impl = _NcclQkvExchange(
                group,
                device=self.device,
                hidden=self.hidden,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                layout=layout,
                max_tokens=self.max_tokens,
                num_slots=self.num_slots,
            )
        self.graph_capturable = bool(getattr(self._impl, "graph_capturable", True))

    def __call__(
        self,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        *,
        out: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ):
        """Project, normalise, rotate and exchange one activation shard; returns the receive views (or ``out``)."""

        self._impl(x, w_qkv, q_norm_weight, k_norm_weight, cos_sin)
        views = self._impl.last_outputs()
        if out is None:
            return views
        if self.layout == "bshd":
            if not isinstance(out, Sequence) or len(out) != 3:
                raise ValueError(
                    "out must hold three tensors (q, k, v) for layout='bshd'"
                )
            for target, view in zip(out, views, strict=True):
                target.copy_(view)
            return tuple(out)
        if isinstance(out, Sequence):
            raise ValueError("out must be one tensor for layout='fused'")
        out.copy_(views)
        return out

    def last_outputs(self):
        """Views of the most recent call's outputs."""

        return self._impl.last_outputs()

    def bound_programs(self, batch: int, s_local: int) -> dict[str, str]:
        """Cake backend: program names of every delivered stage of a ``(batch, s_local)`` call (prepares the row
        eagerly); ``describe()["launch_sequence"]`` says which of them the launch path launches."""

        if self.backend != "nvlink":
            raise RuntimeError(
                "bound_programs is only defined for the Cake (nvlink) backend"
            )
        return self._impl.bound_programs(batch, s_local)

    def poison_receive_slots(self) -> None:
        """Fill this rank's receive region with a NaN pattern (eager only), so the next call must rewrite it."""

        self._impl.poison_receive_slots()

    def describe(self) -> dict[str, Any]:
        return {
            "backend_reason": self.backend_reason,
            "fold_barrier_requested": (
                "auto"
                if self.fold_barrier_requested is None
                else self.fold_barrier_requested
            ),
            **self._impl.describe(),
        }

    def close(self) -> None:
        """Release the session's resources (collective for the Cake backend)."""

        self._impl.close()

    def __enter__(self) -> "QkvNormRopeExchangeOverlap":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = [
    "HEAD_DIM",
    "SUPPORTED_WORLD_SIZES",
    "QkvNormRopeExchangeOverlap",
    "validate_request",
]
