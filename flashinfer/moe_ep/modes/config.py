"""Mode-level config: comm + kernel composition for split and mega paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.split.comm.nccl_ep.config import NcclEpConfig
from ..backends.split.kernel.identity.config import IdentityConfig

__all__ = [
    "IdentityConfig",
    "MegaConfig",
    "SplitConfig",
]


@dataclass
class SplitConfig:
    """Dispatch → inner kernel → combine over NCCL-EP / NIXL-EP."""

    comm: object = field(default_factory=NcclEpConfig)
    kernel: object = field(default_factory=IdentityConfig)


@dataclass
class MegaConfig:
    """Fused expert-parallel mega kernel (symmetric memory)."""

    megakernel: object
    quantize_input: bool = True
    preprocess_weights: bool = True
    transformed_weights: object | None = None
