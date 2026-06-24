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
    ``nccl_comm`` + ``stream``; NIXL-EP uses ``tcp_store``). Carrying both
    here lets a single bootstrap config drive either backend.
    """

    world_size: int
    rank: int
    stream: int = 0  # int representation of a cudaStream_t; 0 = default stream
    nccl_comm: Optional[int] = (
        None  # int representation of ncclComm_t; None = derive from PG
    )
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
    (shape: ``[num_recv_tokens, hidden]``). ``num_tokens`` is the actual
    receive count (= ``max_tokens_per_rank`` in fixed-size LL mode; queried
    via ``ncclEpHandleGetNumRecvTokens`` otherwise).
    """

    expert_tensors: "torch.Tensor"
    num_tokens: int


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
