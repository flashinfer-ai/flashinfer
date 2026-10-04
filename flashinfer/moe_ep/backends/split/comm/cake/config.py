"""Config selecting the Cake MoE all-to-all backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ......comm.abstractions import CommBackend


@dataclass
class CakeAlltoAllConfig:
    """Options of :class:`CakeAlltoAll`.

    Pass to ``create_communication(..., backend=CakeAlltoAllConfig(...))`` or
    ``MoEEpLayer(..., backend=SplitConfig(comm=CakeAlltoAllConfig()))``.

    The dispatch workspace holds each token's activations as described by
    ``MoEEpCommParams.dispatch_format`` plus its routing;
    ``extra_payload_bytes_per_token`` reserves room for anything else sent per
    token. ``eplb_stats_num_experts`` enables all-gathering EPLB statistics of
    that many experts during dispatch. ``enable_rank_mask`` compiles in
    rank-mask support for ``active_rank_mask``. ``use_low_precision_combine``
    sends combine payloads as FP8. ``comm_backend`` exchanges the MNNVL memory
    handles and defaults to torch.distributed over the bootstrap process group.
    """

    backend_name: str = "cake"
    extra_payload_bytes_per_token: int = 0
    eplb_stats_num_experts: int = 0
    enable_rank_mask: bool = False
    use_low_precision_combine: bool = False
    comm_backend: Optional["CommBackend"] = None
