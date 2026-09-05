"""Fused MoE split kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig
    from ...overlap.combine import OverlapCombineFn


@dataclass
class FusedMoeKernelConfig:
    """Inner compute via :class:`flashinfer.fused_moe.layer.MoELayer`."""

    moe_config: "MoEConfig"
    kernel_name: str = "fused_moe"
    mxfp8_dispatch: bool = False
    overlap_combine_fn: Optional["OverlapCombineFn"] = None
