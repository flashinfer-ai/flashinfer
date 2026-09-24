"""Config selecting the NVLink two-sided communication backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ......comm.comm_backend import CommBackend


@dataclass
class NVLinkTwoSidedConfig:
    """Options of :class:`NVLinkTwoSidedCommunication`.

    Pass to ``create_communication(..., backend=NVLinkTwoSidedConfig())`` or
    ``MoEEpLayer(..., backend=SplitConfig(comm=NVLinkTwoSidedConfig()))``.
    ``comm_backend`` exchanges the MNNVL memory handles and defaults to
    torch.distributed over the bootstrap process group.
    """

    backend_name: str = "nvlink_two_sided"
    comm_backend: Optional["CommBackend"] = None
