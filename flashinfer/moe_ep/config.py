"""Config dataclasses + I/O envelopes for moe_ep.

Frozen dataclasses for `BootstrapConfig` (the inputs each backend needs at
construction), `FleetParams` (durable transport sizing), `HandleParams`
(per-iteration topk_ids), and the four envelope types
:class:`DispatchInputParams` / :class:`DispatchOutput` /
:class:`CombineInputParams` / :class:`CombineOutput` that the Handle
interface passes around.

Validation: ctors enforce non-negative ints. Backend-specific constraints
(`max_tokens_per_rank ≤ 1024` for nixl_ep, `num_experts % world_size == 0`,
etc.) live in :mod:`flashinfer.moe_ep._validators` and run inside each
backend's Fleet __init__.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    import torch
    import torch.distributed


class EpAlgorithm(enum.Enum):
    LOW_LATENCY = 0
    HIGH_THROUGHPUT = 1


class EpLayout(enum.Enum):
    """LL receive-buffer layout, mirroring ``nccl.ep.Layout`` for the LL paths.

    * ``EXPERT_MAJOR`` — recv buffer ``[num_local_experts, max_tokens_per_rank *
      world, hidden]``; each padded row is pre-assigned to one local expert and
      combine reweights per-token on receive.
    * ``RANK_MAJOR`` — recv buffer ``[world, max_tokens_per_rank, hidden]``;
      tokens grouped by source rank (received once each, carrying their
      ``topk_idx`` / ``topk_weights``). The caller pre-reduces across local
      experts before combine; combine then just sums across ranks.

    HT always uses the library's ``FLAT`` layout regardless of this field; it is
    only consulted on the LL path.
    """

    EXPERT_MAJOR = 1
    RANK_MAJOR = 2


class QuantType(enum.Enum):
    """Quantization variants surfaced via FleetAlgoKnobQuantization."""

    FP8E4M3 = "fp8e4m3"
    FP8E5M2 = "fp8e5m2"
    NVFP8 = "nvfp8"
    UE8M0 = "ue8m0"


@dataclass(frozen=True)
class BootstrapConfig:
    """Inputs each backend needs to construct a Fleet.

    Backends consume only the fields they care about (NCCL-EP uses
    ``nccl_comm`` / ``process_group`` + ``stream``; NIXL-EP uses
    ``tcp_store``). Carrying all of them here lets a single bootstrap config
    drive either backend.

    Communicator resolution (NCCL-EP), in priority order:

    * ``nccl_comm`` — an existing ``ncclComm_t`` (as an int). The Fleet
      *adopts* it (wraps without taking ownership; it is never destroyed or
      aborted by the Fleet), so a host — e.g. vLLM — can share the exact
      communicator its process group already owns.
    * ``process_group`` — a torch ``ProcessGroup`` to mirror. The Fleet
      creates a *fresh* NCCL communicator over that group's membership (the
      torch-version-robust pattern), letting the caller target a specific EP
      subgroup rather than the default ``WORLD`` group.
    * neither set — mirror the default process group.
    """

    world_size: int
    rank: int
    stream: int = 0  # int representation of a cudaStream_t; 0 = default stream
    nccl_comm: Optional[int] = (
        None  # int representation of ncclComm_t; None = derive from PG
    )
    # Torch process group to mirror when nccl_comm is not supplied. None =
    # default group. Adopting an existing nccl_comm takes precedence.
    process_group: Optional["torch.distributed.ProcessGroup"] = None
    tcp_store: Optional["torch.distributed.TCPStore"] = None

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} not in [0, {self.world_size})")


@dataclass(frozen=True)
class FleetParams:
    """Durable sizing for an EP Fleet.

    `num_channels`, `num_qp_per_rank`, `rdma_buffer_size` are exposed as
    :mod:`flashinfer.moe_ep.algo_knobs` Fleet-level knobs rather than top-level
    fields, since they're backend-specific tuning rather than contract-level
    sizing.
    """

    num_experts: int
    max_tokens_per_rank: int
    token_hidden_size: int
    dtype_bytes: int = 2  # bf16 default; FP8 path overrides
    algorithm: EpAlgorithm = EpAlgorithm.LOW_LATENCY
    # LL receive layout. Ignored by HT (always FLAT). RANK_MAJOR is LL-only.
    layout: EpLayout = EpLayout.EXPERT_MAJOR

    def __post_init__(self) -> None:
        for name in (
            "num_experts",
            "max_tokens_per_rank",
            "token_hidden_size",
            "dtype_bytes",
        ):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"FleetParams.{name} must be positive, got {v}")
        if (
            self.layout is EpLayout.RANK_MAJOR
            and self.algorithm is not EpAlgorithm.LOW_LATENCY
        ):
            raise ValueError(
                "FleetParams.layout=RANK_MAJOR is only valid with "
                "algorithm=LOW_LATENCY (HT uses the FLAT layout)."
            )


@dataclass(frozen=True)
class HandleParams:
    """Per-iteration handle inputs.

    ``topk_ids`` is the routing decision for the iteration; combine reweights
    by the matching ``topk_weights`` carried as a
    :class:`HandleAlgoKnobTopKWeights` knob (not a field here, so optional
    backends can elide it).
    """

    topk_ids: "torch.Tensor"


@dataclass(frozen=True)
class DispatchInputParams:
    """Inputs to :meth:`Handle.dispatch`. ``x`` is a list so backends can
    dispatch multiple tagged tensors in one call (tokens + scales)."""

    x: Sequence["torch.Tensor"]


@dataclass(frozen=True)
class DispatchOutput:
    """Outputs from :meth:`Handle.dispatch`.

    ``expert_tensors`` is the dispatched token tensor on the local rank
    (shape: ``[num_recv_tokens, hidden]``).

    ``recv_topk_idx`` / ``recv_topk_weights`` are the per-received-token routing
    returned by the LL RANK_MAJOR (and HT) layouts — ``[num_recv_tokens, top_k]``
    int64 / fp32. They are ``None`` for the LL EXPERT_MAJOR layout (whose rows are
    pre-assigned to experts by position, so no per-token routing is returned).

    ``expert_counts`` is the per-shard received-token count written by the library
    during dispatch — ``[num_local_experts]`` int32 for LL EXPERT_MAJOR (per-expert)
    or ``[world_size]`` int32 for LL RANK_MAJOR (per-source-rank). ``None`` when the
    layout does not expose it. It is a device tensor; reading it host-side forces a
    sync, so consumers that don't need it can ignore it at no cost.

    ``recv_total_counter`` is a scalar ``[1]`` int32/int64 device tensor holding the
    *actual* total received-token count for HT FLAT. It is populated only when the
    caller opts in via :class:`HandleAlgoKnobNumReceivedTokens` (which binds the
    counter at handle-create time so the HT metadata step fills it); otherwise
    ``None``. It lets an HT consumer trim its compute to ``recv_x[:actual_recv]``
    (the received tokens are front-packed) while the transport buffer stays sized to
    the static ``max_recv_tokens_per_rank`` budget (nccl-ep v0.1 does not resize the
    buffer itself). ``None`` → the consumer must fall back to the full static buffer.
    """

    expert_tensors: "torch.Tensor"
    recv_topk_idx: Optional["torch.Tensor"] = None
    recv_topk_weights: Optional["torch.Tensor"] = None
    expert_counts: Optional["torch.Tensor"] = None
    recv_total_counter: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class CombineInputParams:
    """Inputs to :meth:`Handle.combine`. ``out`` is an optional preallocated
    output buffer; backends MAY fill it in place to avoid an allocation."""

    x: Sequence["torch.Tensor"]
    out: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class CombineOutput:
    """Output of :meth:`Handle.combine`. The same tensor as
    ``CombineInputParams.out`` when it was provided."""

    x: "torch.Tensor"
