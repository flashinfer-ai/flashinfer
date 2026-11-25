"""MoEEpTensors — the bundle of tensors `MoEEpLayer.forward()` consumes.

`hidden_states` and `topk_ids` / `topk_weights` are required; the rest are
optional outputs the backend may populate during dispatch/combine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, eq=False)
class EplbRouting:
    """Caller-owned expert-parallel load-balancer state for logical routes.

    The map has shape ``[logical_experts, max_replicas]`` and contains global
    physical expert IDs. The replica-count vector has one entry per logical
    expert. Both tensors hold int32 or int64 indices. ``expert_load`` is an
    int32 counter per physical expert. ``should_record`` (bool) and
    ``num_unpadded_tokens`` (int32) are one-element tensors. All five tensors
    must be contiguous on one CUDA device. The load counters must have their
    own storage.

    Every call selects a replica using the low 32 bits of
    ``local_token_row * 2654435769`` modulo its replica count. Negative or
    out-of-range IDs and invalid replica counts produce masked routes.
    Counts accumulate in ``expert_load`` only when enabled and for rows below
    ``num_unpadded_tokens``. Mark any padding that should skip dispatch with
    ``-1`` in the input IDs.

    Keep the tensors and their addresses alive until all CUDA graphs that
    reference them are retired. Update tensor contents in place with stream
    ordering. Publish routing after the corresponding weights are ready.
    """

    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    expert_load: torch.Tensor
    should_record: torch.Tensor
    num_unpadded_tokens: torch.Tensor

    def __post_init__(self) -> None:
        import torch

        tensors = self._tensors()
        table, replicas, load, record, live = tensors
        if (
            table.ndim != 2
            or not all(table.shape)
            or replicas.shape != table.shape[:1]
            or load.ndim != 1
            or load.numel() == 0
            or table.dtype not in (torch.int32, torch.int64)
            or replicas.dtype not in (torch.int32, torch.int64)
            or load.dtype != torch.int32
            or record.dtype != torch.bool
            or live.dtype != torch.int32
            or record.ndim > 1
            or live.ndim > 1
            or record.numel() != 1
            or live.numel() != 1
            or not table.is_cuda
            or any(t.device != table.device or not t.is_contiguous() for t in tensors)
        ):
            raise ValueError("Invalid EPLB routing tensor shape, dtype or device")

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.logical_to_physical_map,
            self.logical_replica_count,
            self.expert_load,
            self.should_record,
            self.num_unpadded_tokens,
        )


@dataclass
class MoEEpTensors:
    """Per-iteration tensor bundle for MoE EP.

    Mutable on purpose: backends populate ``num_tokens_per_expert`` /
    ``recv_count`` in place during dispatch when an
    :class:`flashinfer.moe_ep.algo_knobs.HandleAlgoKnobNumReceivedTokens`
    knob targets them.
    """

    hidden_states: "torch.Tensor"
    topk_ids: "torch.Tensor"
    topk_weights: "torch.Tensor"
    scales: Optional["torch.Tensor"] = None
    fc1_alpha: Optional["torch.Tensor"] = None
    fc2_alpha: Optional["torch.Tensor"] = None
    fc1_norm_const: Optional["torch.Tensor"] = None
    # ``(num_tokens,)`` fp32 per-token fc1 activation scale (NVFP4 mega path,
    # requires ``enable_fc1_activation_per_token_scale`` on the kernel config):
    # multiplied into the dequantized fc1 output before the clamp / activation.
    fc1_activation_per_token_scale: Optional["torch.Tensor"] = None
    recv_count: Optional["torch.Tensor"] = None
    num_tokens_per_expert: Optional["torch.Tensor"] = None
    output: Optional["torch.Tensor"] = None
    # Largest ``num_tokens`` over the EP ranks for this step, the same on
    # every rank. ``MoEEpSplitLayer`` passes it to
    # :meth:`MoEEpCommunication.dispatch` as ``runtime_max_tokens_per_rank``,
    # which sizes the receive rows and the inner kernel's batch; ``None``
    # uses ``FleetParams.max_tokens_per_rank``. Fleet/Handle transports and
    # ``MoEEpMegaLayer`` ignore it.
    runtime_max_tokens_per_rank: Optional[int] = None
    # Shared-expert input when routed activations are already quantized in a
    # different format. BF16 is quantized by the backend; MXFP8 also needs scales.
    shared_hidden_states: Optional["torch.Tensor"] = None
    shared_scales: Optional["torch.Tensor"] = None

    @property
    def num_tokens(self) -> int:
        """Batch size for this forward (``hidden_states.shape[0]``)."""
        return self.hidden_states.shape[0]
