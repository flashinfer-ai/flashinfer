"""Comm backend config objects."""

from .nccl_ep.config import NcclEpConfig
from .nixl_ep.config import NvepConfig

NCCLEPConfig = NcclEpConfig

__all__ = ["NCCLEPConfig", "NcclEpConfig", "NvepConfig"]
