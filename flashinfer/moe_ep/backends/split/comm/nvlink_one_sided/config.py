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
    uses them for every step where supported; ``False`` never uses them. Steps
    the Lamport transport takes (see ``lamport``) do not use them. Only one
    workspace per process can bind CFT endpoints; further workspaces use the
    fence path.

    ``lamport`` selects the Lamport transport for small steps: one dispatch and
    one combine kernel in which every pushed vector carries its own arrival, so
    no rank waits for a completion flag or fence. It needs compute capability
    10.0 or 10.3, at most 64 EP ranks, a BF16 combine and a BF16, ``MXFP8``,
    ``DeepSeekFp8`` or ``NVFP4`` dispatch format. Where every rank supports it,
    it takes every step without ``eplb_local_stats`` whose busiest rank sends
    at most ``lamport_max_tokens`` tokens, ahead of CFT. A step's busiest-rank
    count is the ``runtime_max_tokens_per_rank`` passed to dispatch (the same
    on every rank, fixed in a captured CUDA graph), which ``MoEEpSplitLayer``
    passes from ``MoEEpTensors.runtime_max_tokens_per_rank``; a step that
    passes none is compared at the capacity, ``max_tokens_per_rank``. ``None``
    (default) turns it on unless ``cft=True``; ``True`` turns it on and cannot
    be combined with ``cft=True``; ``False`` leaves those steps to CFT or the
    fence path. A rank where it is off turns it off on every rank.
    ``lamport_max_tokens`` must be the same on every rank; it defaults to the
    largest step whose dispatch grid stays below one wave of resident blocks
    and whose Lamport regions stay within 64 MiB per rank (a speed bound, not
    a memory budget), then clamped to 159 to 512 tokens and to
    ``max_tokens_per_rank``. The regions hold ``ep_size * lamport_max_tokens``
    landing, combine and receive rows; rank 0 logs their size, about 420 MiB
    per rank at EP 64 for hidden size 7168 in BF16. Payload words equal to the
    transport's sentinel arrive rewritten to the same value (NaN stays NaN), so
    callers sending other bytes in BF16 rows turn it off, as the split layer
    does for kernels that pack their dispatch rows. A Lamport step places the
    rows it receives from rank ``src`` for its token ``t`` at
    ``src * tokens_per_rank + t``, ``tokens_per_rank`` being the step's
    ``runtime_max_tokens_per_rank``. Like the fence path, it needs every rank
    to pass ``topk_weights`` in a step or none (see
    ``NVLinkOneSidedAlltoAll.dispatch``).

    ``timeout_sec`` bounds every wait on a peer rank, in nominal seconds at a
    2 GHz SM clock; a wait that exceeds it traps the kernel. It is recorded
    into CUDA graphs at capture time.

    ``enable_pdl`` launches the dispatch and combine kernels with programmatic
    dependent launch; ``None`` (default) enables it on compute capability 9.0
    and newer.
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
    lamport: Optional[bool] = None
    lamport_max_tokens: Optional[int] = None
    timeout_sec: int = 300
    enable_pdl: Optional[bool] = None
