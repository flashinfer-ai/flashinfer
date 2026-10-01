"""NVLink one-sided MoE communication with the generated Cake kernels.

Same protocol, workspace layout and interface as
:class:`NVLinkOneSidedAlltoAll`; only the kernels differ. The Cake kernels
are generated for Blackwell and require compute capability 10.0 or 10.3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .....core.comm.communication import MoEEpCommParams, register_communication
from ..nvlink_one_sided.communication import NVLinkOneSidedAlltoAll
from .config import NVLinkOneSidedCakeConfig

if TYPE_CHECKING:
    from .....config import BootstrapConfig

logger = logging.getLogger(__name__)

_CAKE_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))


@register_communication("nvlink_one_sided_cake")
class CakeAlltoAll(NVLinkOneSidedAlltoAll):
    """:class:`NVLinkOneSidedAlltoAll` running the generated Cake kernels."""

    alltoall_backend = "cake"

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NVLinkOneSidedCakeConfig] = None,
    ) -> None:
        super().__init__(
            bootstrap, params, NVLinkOneSidedCakeConfig() if config is None else config
        )

    @classmethod
    def is_platform_supported(cls) -> bool:
        if not super().is_platform_supported():
            return False
        import torch

        capability = torch.cuda.get_device_capability()
        if capability not in _CAKE_COMPUTE_CAPABILITIES:
            logger.error(
                "Cake MoE all-to-all kernels unavailable: they need compute "
                "capability 10.0 or 10.3, got %d.%d",
                *capability,
            )
            return False
        return True
