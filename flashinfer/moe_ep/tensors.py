"""MoEEpTensors — the bundle of tensors `MoEEpLayer.forward()` consumes.

`hidden_states` and `topk_ids` / `topk_weights` are required; the rest are
optional outputs the backend may populate during dispatch/combine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    import torch


class MegaMoeLoraOutput(NamedTuple):
    """Finalized MegaMoE hidden output plus unweighted post-SwiGLU snapshot for down-LoRA.

    Returned by :meth:`MoEEpMegaLayer.forward` when ``enable_fc1_lora_delta=True``.
    Unpacks as ``y, act, token_id, expert_offsets, src_rank``. Pad rows in ``act``
    have ``token_id == -1`` and ``src_rank == -1``.
    """

    output: "torch.Tensor"
    act: "torch.Tensor"
    token_id: "torch.Tensor"
    expert_offsets: "torch.Tensor"
    src_rank: "torch.Tensor"


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
    # ``(num_tokens, top_k, 2 * intermediate_size)`` bf16 unweighted FC1 LoRA
    # delta (Mn-bias, ``[..., :I]`` gate and ``[..., I:]`` up), added in fp32
    # before clamp / activation. Requires ``enable_fc1_lora_delta=True``.
    gemm1_lora_delta: Optional["torch.Tensor"] = field(default=None, kw_only=True)
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
