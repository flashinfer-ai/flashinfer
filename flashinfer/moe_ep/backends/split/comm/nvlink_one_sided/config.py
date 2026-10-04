"""Config selecting the NVLink one-sided communication backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ......comm.abstractions import CommBackend


@dataclass
class NVLinkOneSidedConfig:
    """Options of :class:`NVLinkOneSidedAlltoAll`.

    Pass to ``create_communication(..., backend=NVLinkOneSidedConfig(...))`` or
    ``MoEEpLayer(..., backend=SplitConfig(comm=NVLinkOneSidedConfig(...)))``.

    The dispatch workspace holds each token's activations as described by
    ``MoEEpCommParams.dispatch_format`` plus its routing;
    ``extra_payload_bytes_per_token`` reserves room for anything else sent per
    token. ``eplb_stats_num_experts`` enables all-gathering EPLB statistics of
    that many experts during dispatch. ``enable_rank_mask`` compiles in
    rank-mask support for ``active_rank_mask``. ``use_low_precision_combine``
    sends combine payloads as FP8. ``comm_backend`` exchanges the MNNVL memory
    handles and defaults to torch.distributed over the bootstrap process group.

    ``cft`` selects CFT counted writes, which push payloads over the NVLink
    fabric with ``fabric.try_put.counted`` and let the receiver wait on
    hardware byte counters. They need compute capability 10.0 or newer, a
    FlashInfer build against CUDA 13.4+ and a driver of the 615 branch or newer
    that exports the CUDA logical-endpoint API; payload rows must be multiples
    of 16 bytes. ``None`` (default) uses them where every rank supports them,
    for steps whose busiest rank sends at most ``cft_max_tokens_for_dispatch``
    (dispatch) or ``cft_max_tokens_for_combine`` (combine) tokens; ``True``
    uses them for every step where supported; ``False`` never uses them. Only
    one workspace per process can bind CFT endpoints; further workspaces use
    the fence path.

    ``timeout_sec`` bounds every wait on a peer rank, in nominal seconds at a
    2 GHz SM clock; a wait that exceeds it traps the kernel. It is recorded
    into CUDA graphs at capture time.
    """

    backend_name: str = "nvlink_one_sided"
    extra_payload_bytes_per_token: int = 0
    eplb_stats_num_experts: int = 0
    enable_rank_mask: bool = False
    use_low_precision_combine: bool = False
    comm_backend: Optional["CommBackend"] = None
    cft: Optional[bool] = None
    cft_max_tokens_for_dispatch: int = 128
    cft_max_tokens_for_combine: int = 128
    timeout_sec: int = 300
