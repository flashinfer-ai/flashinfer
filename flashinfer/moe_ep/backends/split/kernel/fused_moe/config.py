"""Fused MoE split kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig


@dataclass
class FusedMoeKernelConfig:
    """Inner compute via :class:`flashinfer.fused_moe.layer.MoELayer`.

    ``moe_config.backend`` controls the compute candidates independently of
    the EP transport. For NVFP4×NVFP4 or MXFP4×MXFP8, list both
    ``CuteDslConfig()`` and ``TrtllmFp4Config()`` in ``BackendOptions`` to enable
    MoELayer's existing backend selection, or list one to keep an explicit
    backend. Each candidate retains its native eligibility checks and requires
    a prepared weight view.
    Warm up each input shape eagerly before CUDA graph capture.
    """

    moe_config: "MoEConfig"
    kernel_name: str = "fused_moe"
    mxfp8_dispatch: bool = False
