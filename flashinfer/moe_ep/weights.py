"""Canonical MoE weight container for all moe_ep paths.

Users supply a single :class:`MoEWeightPack` via :attr:`FleetParams.weights`.
Split and mega kernel plugins materialize backend-specific layouts in
:meth:`~flashinfer.moe_ep.core.kernel.base.SplitKernelBackend.preprocess_weights`
or the mega equivalent — callers never touch per-backend native views directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MoEWeightPack:
    """Per-rank expert weights in canonical logical layout.

    ``w13`` — gate+up projection ``[local_experts, 2*intermediate, hidden]`` in
    bf16, or ``[..., hidden // 2]`` fp4 (``torch.int8`` / ``torch.uint8``) when
    ``w13_scale`` is supplied for mega kernels.
    ``w2``  — down projection ``[local_experts, hidden, intermediate]`` in bf16,
    or ``[..., intermediate // 2]`` fp4 when ``w2_scale`` is supplied.
    ``w13_scale`` / ``w2_scale`` — optional block scale factors. Mega DeepGEMM
    kernels expect ue8m0-packed ``torch.uint8`` scales with trailing dims
    ``hidden // 32`` and ``intermediate // 32`` respectively.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
