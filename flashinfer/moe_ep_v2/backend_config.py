"""Backward-compatible re-exports — prefer ``split.config`` / ``mega.config``."""

from .mega.config import DeepGemmMegaMoeConfig, MegaConfig
from .split.backends import NcclEpConfig, NvepConfig
from .split.config import (
    FusedMoeKernelConfig,
    IdentityConfig,
    NCCLEPConfig,
    SplitConfig,
)

__all__ = [
    "DeepGemmMegaMoeConfig",
    "FusedMoeKernelConfig",
    "IdentityConfig",
    "MegaConfig",
    "NCCLEPConfig",
    "NcclEpConfig",
    "NvepConfig",
    "SplitConfig",
]
