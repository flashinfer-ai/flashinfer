"""Comm backend config objects."""

from .nccl_ep.config import NcclEpConfig
from .nixl_ep.config import NvepConfig
from .nvlink_one_sided.config import NVLinkOneSidedConfig
from .nvlink_two_sided.config import NVLinkTwoSidedConfig

NCCLEPConfig = NcclEpConfig

__all__ = [
    "NCCLEPConfig",
    "NVLinkOneSidedConfig",
    "NVLinkTwoSidedConfig",
    "NcclEpConfig",
    "NvepConfig",
]
