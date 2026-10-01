"""Config selecting the NVLink one-sided backend with the Cake kernels."""

from __future__ import annotations

from dataclasses import dataclass

from ..nvlink_one_sided.config import NVLinkOneSidedConfig


@dataclass
class NVLinkOneSidedCakeConfig(NVLinkOneSidedConfig):
    """Options of :class:`CakeAlltoAll`; see
    :class:`NVLinkOneSidedConfig` for the fields.

    Pass to ``create_communication(..., backend=NVLinkOneSidedCakeConfig(...))``
    or ``MoEEpLayer(..., backend=SplitConfig(comm=NVLinkOneSidedCakeConfig()))``.
    """

    backend_name: str = "nvlink_one_sided_cake"
