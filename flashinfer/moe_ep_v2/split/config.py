"""Split-path config: comm backends + inner kernels."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import NcclEpConfig, NvepConfig


@dataclass
class IdentityConfig:
    """Inner compute stub — passes dispatched tokens through unchanged.

    ``require_weights`` reserves a slot for a future weight-layout pass;
    today it only requires ``FleetParams.weights`` and aliases them unchanged.
    """

    kernel_name: str = "identity"
    require_weights: bool = False


@dataclass
class FusedMoeKernelConfig:
    """Placeholder for a future ``flashinfer.fused_moe`` inner kernel."""

    kernel_name: str = "fused_moe"


@dataclass
class SplitConfig:
    """Dispatch → inner kernel → combine over NCCL-EP / NIXL-EP."""

    comm: object = field(default_factory=NcclEpConfig)
    kernel: object = field(default_factory=IdentityConfig)


NCCLEPConfig = NcclEpConfig

__all__ = [
    "FusedMoeKernelConfig",
    "IdentityConfig",
    "NCCLEPConfig",
    "NcclEpConfig",
    "NvepConfig",
    "SplitConfig",
]
