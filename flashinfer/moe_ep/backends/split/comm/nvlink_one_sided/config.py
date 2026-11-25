"""Config selecting the NVLink one-sided communication backend."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ......comm.abstractions import CommBackend

SyncProtocol = Literal["auto", "fence", "sentinel"]
SYNC_PROTOCOLS = ("auto", "fence", "sentinel")
# Under sync_protocol="auto", steps whose busiest rank sends at most this many
# tokens take the sentinel protocol and larger steps the fence protocol.
AUTO_SENTINEL_MAX_TOKENS = 256

# Deprecated fields and what each one does now.
_DEPRECATED_FIELDS = {
    "cft": "CFT counted writes are not available; cft=True selects "
    'sync_protocol="fence" and cft=False has no effect',
    "cft_max_tokens_for_dispatch": "it has no effect",
    "cft_max_tokens_for_combine": "it has no effect",
    "lamport": 'use sync_protocol; lamport=False selects sync_protocol="fence" '
    'and lamport=True keeps "auto"',
    "lamport_max_tokens": "it sets the largest step of the sentinel protocol "
    'under sync_protocol="auto"',
}


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

    ``sync_protocol`` selects how a rank learns that the rows of a step have
    arrived:

    * ``"fence"``: senders write their rows, then publish per-peer completion
      flags behind a system-scope fence, and receivers wait for the flags of
      every peer. Carries every step.
    * ``"sentinel"``: every landing row holds a sentinel word that no payload
      word takes, and every 16-byte vector a sender pushes replaces it, so
      receivers poll the data itself: no flags and no fence between data and
      its readiness. Takes every step it can carry, up to
      ``max_tokens_per_rank`` tokens.
    * ``"auto"`` (default): the sentinel protocol for steps whose busiest rank
      sends at most ``AUTO_SENTINEL_MAX_TOKENS`` (256) tokens, the fence
      protocol for the others.

    ``sync_protocol`` must be the same on every EP rank; constructing the
    communication raises ``ValueError`` otherwise. A step's busiest-rank count
    is the ``runtime_max_tokens_per_rank`` passed to dispatch (the same on
    every rank, fixed in a captured CUDA graph), which ``MoEEpSplitLayer``
    passes from ``MoEEpTensors.runtime_max_tokens_per_rank``; a step that
    passes none is compared at the capacity, ``max_tokens_per_rank``.

    The sentinel protocol needs compute capability 10.0, 10.3 or 10.7, at most
    64 EP ranks, a BF16 combine and a BF16, ``MXFP8``, ``DeepSeekFp8`` or
    ``NVFP4`` dispatch format. Where a rank does not support it, every rank
    uses the fence protocol (logged as a warning under ``"sentinel"``). It
    cannot carry steps with ``eplb_local_stats`` or with payloads outside the
    instance's format; those take the fence protocol. Payload words equal to
    its sentinel arrive rewritten to the same value (NaN stays NaN), so callers
    sending other bytes in BF16 rows use ``"fence"``, as the split layer does
    for kernels that pack their dispatch rows. A sentinel step places the rows
    it receives from rank ``src`` for its token ``t`` at
    ``src * tokens_per_rank + t``, ``tokens_per_rank`` being the step's
    ``runtime_max_tokens_per_rank``. Like the fence protocol, it needs every
    rank to pass ``topk_weights`` in a step or none (see
    ``NVLinkOneSidedAlltoAll.dispatch``). Its landing, combine and receive
    regions hold ``ep_size`` rows per token of its largest step, next to the
    fence protocol's regions in the workspace; rank 0 logs their size.

    ``timeout_sec`` bounds every wait on a peer rank, in nominal seconds at a
    2 GHz SM clock; a wait that exceeds it traps the kernel. It is recorded
    into CUDA graphs at capture time.

    ``enable_pdl`` launches the dispatch and combine kernels with programmatic
    dependent launch; ``None`` (default) enables it on compute capability 9.0
    and newer.

    Deprecated, each with a ``DeprecationWarning`` when set:

    * ``lamport``: ``False`` selects ``sync_protocol="fence"``; ``True`` keeps
      ``"auto"``.
    * ``lamport_max_tokens``: the largest step the sentinel protocol takes
      under ``"auto"``, the same on every rank.
    * ``cft``: CFT counted writes are not available; ``True`` selects
      ``sync_protocol="fence"``, ``False`` has no effect.
    * ``cft_max_tokens_for_dispatch``, ``cft_max_tokens_for_combine``: no
      effect.

    ``lamport``, ``lamport_max_tokens`` and ``cft`` cannot be combined with a
    ``sync_protocol`` other than ``"auto"``, nor ``lamport=True`` with
    ``cft=True``; either raises ``ValueError``.
    """

    backend_name: str = "nvlink_one_sided"
    extra_payload_bytes_per_token: int = 0
    eplb_stats_num_experts: int = 0
    enable_rank_mask: bool = False
    use_low_precision_combine: bool = False
    comm_backend: Optional["CommBackend"] = None
    cft: Optional[bool] = None
    cft_max_tokens_for_dispatch: Optional[int] = None
    cft_max_tokens_for_combine: Optional[int] = None
    lamport: Optional[bool] = None
    lamport_max_tokens: Optional[int] = None
    timeout_sec: int = 300
    enable_pdl: Optional[bool] = None
    sync_protocol: SyncProtocol = field(default="auto", kw_only=True)

    def __post_init__(self) -> None:
        if self.sync_protocol not in SYNC_PROTOCOLS:
            raise ValueError(
                f"sync_protocol must be one of {SYNC_PROTOCOLS}, "
                f"got {self.sync_protocol!r}"
            )
        for name, note in _DEPRECATED_FIELDS.items():
            if getattr(self, name) is not None:
                warnings.warn(
                    f"NVLinkOneSidedConfig.{name} is deprecated: {note}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        if self.sync_protocol != "auto" and any(
            getattr(self, name) is not None
            for name in ("lamport", "lamport_max_tokens", "cft")
        ):
            raise ValueError(
                "lamport, lamport_max_tokens and cft cannot be combined with "
                f"sync_protocol={self.sync_protocol!r}"
            )
        if self.lamport is True and self.cft is True:
            raise ValueError(
                "lamport=True and cft=True select different protocols; "
                "use sync_protocol"
            )

    def _effective_sync_protocol(self) -> SyncProtocol:
        """``sync_protocol`` with the deprecated ``lamport`` and ``cft`` applied."""
        if self.lamport is False or self.cft is True:
            return "fence"
        return self.sync_protocol
