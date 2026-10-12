"""Config selecting the NVLink one-sided backend with the Cake kernels."""

from __future__ import annotations

from dataclasses import dataclass

from ..nvlink_one_sided.config import NVLinkOneSidedConfig


@dataclass
class CakeAlltoAllConfig(NVLinkOneSidedConfig):
    """Options of :class:`CakeAlltoAll`; see
    :class:`NVLinkOneSidedConfig` for the fields.

    Pass to ``create_communication(..., backend=CakeAlltoAllConfig(...))``
    or ``MoEEpLayer(..., backend=SplitConfig(comm=CakeAlltoAllConfig()))``.
    """

    backend_name: str = "cake"
