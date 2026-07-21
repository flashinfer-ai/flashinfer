"""MoEEpTensors — the bundle of tensors `MoEEpLayer.forward()` consumes.

`hidden_states` and `topk_ids` / `topk_weights` are required; the rest are
optional outputs the backend may populate during dispatch/combine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


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
    recv_count: Optional["torch.Tensor"] = None
    num_tokens_per_expert: Optional["torch.Tensor"] = None

    @property
    def num_tokens(self) -> int:
        """Batch size for this forward (``hidden_states.shape[0]``)."""
        return self.hidden_states.shape[0]
