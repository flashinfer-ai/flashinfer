"""Canonical MoE weight container for all moe_ep paths.

Users supply a single :class:`MoEWeightPack` via :attr:`FleetParams.weights`.
Split and mega kernel plugins materialize backend-specific layouts in
:meth:`~flashinfer.moe_ep_v2.core.kernel.base.SplitKernelBackend.preprocess_weights`
or the mega equivalent — callers never touch per-backend native views directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MoEWeightPack:
    """Per-rank expert weights in canonical logical layout.

    ``w13`` — gate+up projection ``[local_experts, 2*intermediate, hidden]``.
    ``w2``  — down projection ``[local_experts, hidden, intermediate]``.
    Optional scale tensors support quantized mega kernels.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
