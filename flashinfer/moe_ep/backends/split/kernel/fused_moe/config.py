"""Fused MoE split kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig


@dataclass
class FusedMoeKernelConfig:
    """Inner compute via :class:`flashinfer.fused_moe.layer.MoELayer`."""

    moe_config: "MoEConfig"
    kernel_name: str = "fused_moe"
