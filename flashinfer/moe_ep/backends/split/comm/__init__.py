"""Comm backend config objects."""

from .nccl_ep.config import NcclEpConfig
from .nixl_ep.config import NvepConfig
from .nvlink_one_sided.config import NVLinkOneSidedConfig
from .nvlink_one_sided_cake.config import NVLinkOneSidedCakeConfig
from .nvlink_two_sided.config import NVLinkTwoSidedConfig

NCCLEPConfig = NcclEpConfig

__all__ = [
    "NCCLEPConfig",
    "NVLinkOneSidedCakeConfig",
    "NVLinkOneSidedConfig",
    "NVLinkTwoSidedConfig",
    "NcclEpConfig",
    "NvepConfig",
]
