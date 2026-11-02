"""NVLink one-sided MoE communication over MNNVL symmetric memory.

The dispatch kernel writes each token directly into the receive buffers of the
ranks that own its experts; the combine kernel reads the expert outputs back
from those ranks and reduces them locally. Where the platform supports it,
either direction can instead use CFT counted writes: the sender pushes over the
NVLink fabric with ``fabric.try_put.counted`` and the receiver waits on
hardware byte counters instead of per-round completion flags. Small steps can
use the Lamport transport (``include/flashinfer/comm/moe_alltoall_lamport.cuh``):
every pushed vector carries its own arrival, so neither direction waits for
flags.

One symmetric workspace, laid out by ``moe_a2a_get_workspace_layout``, holds
the round state, the dispatch receive buffers and the combine buffers of a
rank. Its size grows with ``ep_size * max_tokens_per_rank``; the Lamport
regions, appended by ``moe_a2a_lamport_extend_workspace_layout``, grow with
``ep_size * lamport_max_tokens``. Instances with the same layout share one
workspace.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Optional, Sequence

from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from ..nvlink_common import (
    check_round_owner,
    mnnvl_mapping_and_config,
    nvlink_platform_supported,
    remap_mnnvl_memory,
    unmap_mnnvl_memory,
)
from .config import NVLinkOneSidedConfig
from .kernels import get_nvlink_one_sided_alltoall_module, layout_constants

if TYPE_CHECKING:
    import torch

    from ......comm.abstractions import CommBackend
    from .....config import BootstrapConfig

logger = logging.getLogger(__name__)

_MAX_TIMEOUT_SEC = 24 * 60 * 60
_CFT_MIN_DRIVER_BRANCH = 615
# fabric.try_put.counted moves 16-byte chunks.
_CFT_ALIGNMENT_BYTES = 16
# Top-k values the kernels are specialized for.
_SUPPORTED_TOP_K = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22)
# Lamport transport: its payload format (layout constant suffix) per dispatch format.
_LAMPORT_FORMATS = {
    None: "BF16",
    "BF16": "BF16",
    "MXFP8": "MXFP8",
    "DeepSeekFp8": "FP8_BLOCK128",
    "NVFP4": "NVFP4",
}
# Default token limit of the Lamport transport (_default_lamport_max_tokens). Up to 159 tokens
# per rank the transport was faster than the fence transport in every configuration measured
# (GB300, EP 4 to 64, hidden 3584 to 7168, every payload format), so the default never goes below.
_LAMPORT_MIN_DEFAULT_MAX_TOKENS = 159
# Above it the default keeps the ep_size * limit rows a Lamport step polls, copies out and re-arms
# within this: a speed bound, not a memory budget; the largest that kept every GB300 case faster.
_LAMPORT_MAX_DEFAULT_REGION_BYTES = 64 << 20


def _driver_branch() -> Optional[int]:
    """Major version of the installed NVIDIA driver, or ``None`` if NVML fails."""
    import pynvml

    try:
        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None
    if isinstance(version, bytes):
        version = version.decode(errors="replace")
    match = re.match(r"\s*(\d+)", str(version))
    return int(match.group(1)) if match else None


def _cft_unsupported_reason(device_index: int, fabric_handles: bool) -> Optional[str]:
    """Why this rank cannot use CFT counted writes, or ``None`` if it can.

    ``fabric_handles`` tells whether the workspace will be shared through
    FABRIC handles; logical endpoints can only bind such memory. Never raises:
    every rank must reach the collective CFT agreement, and a rank whose probe
    fails falls back to the fence path with the others.
    """
    if not fabric_handles:
        return "the workspace is not shared through FABRIC handles"
    try:
        return _probe_cft_support(device_index)
    except Exception as exc:  # NVML, CUDA driver or cuda-python errors alike
        return f"the CFT support probe failed: {type(exc).__name__}: {exc}"


def _probe_cft_support(device_index: int) -> Optional[str]:
    import torch

    from ......comm.mnnvl import cuda

    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 10:
        return f"compute capability {major}.{minor} is below 10.0"
    branch = _driver_branch()
    if branch is None:
        return "the NVIDIA driver version could not be queried"
    if branch < _CFT_MIN_DRIVER_BRANCH:
        return f"driver branch {branch} is older than {_CFT_MIN_DRIVER_BRANCH}"
    try:
        attributes = (
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
        )
    except AttributeError:
        return "the installed cuda-python does not expose logical endpoints"
    for attribute in attributes:
        status, supported = cuda.cuDeviceGetAttribute(attribute, device_index)
        if status != cuda.CUresult.CUDA_SUCCESS:
            return f"querying {attribute.name} failed with {status.name}"
        if not supported:
            return f"the device does not support {attribute.name}"
    return None


class _LamportRow(NamedTuple):
    """Row geometry of a Lamport payload format, from the kernel module."""

    hidden_bytes: int  # one token's hidden states
    scale_bytes: int  # one token's scales (0 for BF16)
    blocks_per_token: int  # dispatch blocks per token slot
    bytes_per_row: int  # Lamport workspace bytes per (source rank, token) row


def _lamport_row(
    module: Any, index: Any, lamport_format: str, params: MoEEpCommParams
) -> _LamportRow:
    """Raises if the kernels do not take rows of ``params.hidden_size``."""
    return _LamportRow(
        *map(
            int,
            module.moe_a2a_lamport_row_bytes(
                getattr(index, f"LAMPORT_FORMAT_{lamport_format}"),
                params.hidden_size,
                params.top_k,
            ),
        )
    )


def _probe_lamport_support(
    module: Any,
    params: MoEEpCommParams,
    config: NVLinkOneSidedConfig,
    ep_size: int,
    index: Any,
) -> Optional[str]:
    import torch

    if not module.moe_a2a_lamport_available():
        return "the kernel module has no Lamport kernels for this GPU"
    if ep_size > index.LAMPORT_MAX_RANKS:
        return f"it supports at most {index.LAMPORT_MAX_RANKS} EP ranks"
    if params.top_k > index.LAMPORT_MAX_TOP_K:
        return f"it supports top_k <= {index.LAMPORT_MAX_TOP_K}"
    if params.token_dtype != torch.bfloat16:
        return "it combines BF16 expert outputs only"
    if config.use_low_precision_combine:
        return "it has no FP8 combine"
    if ((params.invalid_expert_id - index.LAMPORT_IDS_SENTINEL) & 0xFFFFFFFF) == 0:
        return "invalid_expert_id equals its expert-id sentinel as an int32"
    name = getattr(params.dispatch_format, "name", None)
    if name not in _LAMPORT_FORMATS:
        return f"it does not carry the dispatch format {name}"
    _lamport_row(module, index, _LAMPORT_FORMATS[name], params)
    return None


def _lamport_unsupported_reason(
    module: Any,
    params: MoEEpCommParams,
    config: NVLinkOneSidedConfig,
    ep_size: int,
    index: Any,
) -> Optional[str]:
    """Why this rank cannot use the Lamport transport, or ``None`` if it can.

    Never raises: every rank must reach the collective Lamport agreement, and a
    rank whose probe fails falls back to the other transports with the others.
    """
    try:
        return _probe_lamport_support(module, params, config, ep_size, index)
    except Exception as exc:  # CUDA queries and the kernels' row checks alike
        return f"the Lamport support probe failed: {type(exc).__name__}: {exc}"


def _default_lamport_max_tokens(
    ep_size: int, num_sms: int, row: _LamportRow, index: Any
) -> int:
    """Default token limit of the Lamport transport, in this order of precedence:

    1. at least ``_LAMPORT_MIN_DEFAULT_MAX_TOKENS`` (159);
    2. else the largest step whose dispatch grid stays below one full wave of
       resident blocks (from a full wave on, the fence transport catches up),
       at most the kernels' 512;
    3. lowered while the Lamport regions (landing, combine and receive rows,
       ``ep_size * limit`` of them) would exceed
       ``_LAMPORT_MAX_DEFAULT_REGION_BYTES`` per rank.

    The caller caps it by ``max_tokens_per_rank``.
    """
    below_one_wave = (
        index.LAMPORT_DISPATCH_BLOCKS_PER_SM * num_sms - 1
    ) // row.blocks_per_token
    by_region_bytes = _LAMPORT_MAX_DEFAULT_REGION_BYTES // (ep_size * row.bytes_per_row)
    return max(
        min(below_one_wave, index.LAMPORT_MAX_TOKENS_PER_RANK, by_region_bytes),
        _LAMPORT_MIN_DEFAULT_MAX_TOKENS,
    )


def _lamport_request(config: NVLinkOneSidedConfig) -> Optional[bool]:
    """``config.lamport``, with auto (``None``) off under ``cft=True``: an explicit
    CFT request keeps every step for CFT counted writes. (``lamport=True`` with
    ``cft=True`` is refused at construction.)"""
    if config.lamport is None and config.cft is True:
        return False
    return config.lamport


def _log_unavailable(
    transport: str, requested: Optional[bool], available: bool, reason: Optional[str]
) -> None:
    """Log why a transport asked for (warning) or left to auto (info) is off."""
    if requested is not False and not available:
        (logger.warning if requested else logger.info)(
            "NVLink one-sided %s disabled: %s",
            transport,
            reason or "another EP rank does not support it or turned it off",
        )


@register_communication("nvlink_one_sided")
class NVLinkOneSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink one-sided put/get kernels.

    ``hidden_states``, its optional scale factors, ``topk_ids`` and
    ``topk_weights`` travel as payloads of a single dispatch. Received rows
    beyond each source rank's token count get ``invalid_expert_id`` routing;
    after a Lamport step, row ``source_rank * tokens_per_rank + token`` holds
    that token and rows of tokens not routed here get ``invalid_expert_id``.
    See :class:`NVLinkOneSidedConfig` for CFT counted writes, the Lamport
    transport and timeouts.
    """

    # Workspaces shared by instances with the same layout, keyed by
    # (ep_rank, ep_size, layout).
    _WORKSPACES: ClassVar[dict[tuple, dict[str, Any]]] = {}
    # Key of the workspace bound to the process's CFT logical endpoints.
    _CFT_WORKSPACE_KEY: ClassVar[Optional[tuple]] = None

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NVLinkOneSidedConfig] = None,
    ) -> None:
        import torch

        from ......comm.mnnvl import MnnvlMemory
        from ......utils import device_support_pdl

        self.config = config = NVLinkOneSidedConfig() if config is None else config
        # Before any collective, so that every rank fails alike.
        if config.lamport is True and config.cft is True:
            raise ValueError(
                "lamport=True and cft=True both ask for every eligible step; "
                "set one of them to None or False"
            )
        # Collective platform check across the EP group; it runs before the
        # base class's local check so that no rank fails alone.
        mapping, mnnvl_config = mnnvl_mapping_and_config(bootstrap, config.comm_backend)
        super().__init__(bootstrap, params)

        if config.extra_payload_bytes_per_token < 0:
            raise ValueError("extra_payload_bytes_per_token must be non-negative")
        if not 0 <= config.eplb_stats_num_experts <= params.num_experts:
            raise ValueError("eplb_stats_num_experts must be in [0, num_experts]")
        if isinstance(config.timeout_sec, bool) or not isinstance(
            config.timeout_sec, int
        ):
            raise TypeError("timeout_sec must be an integer number of seconds")
        if not 0 < config.timeout_sec <= _MAX_TIMEOUT_SEC:
            raise ValueError(f"timeout_sec must be in 1..{_MAX_TIMEOUT_SEC}")
        if (
            config.cft_max_tokens_for_dispatch < 0
            or config.cft_max_tokens_for_combine < 0
        ):
            raise ValueError("CFT token-count limits must be non-negative")
        if config.lamport_max_tokens is not None and (
            isinstance(config.lamport_max_tokens, bool)
            or not isinstance(config.lamport_max_tokens, int)
            or config.lamport_max_tokens <= 0
        ):
            raise ValueError("lamport_max_tokens must be a positive integer or None")
        if params.top_k not in _SUPPORTED_TOP_K:
            raise ValueError(
                f"NVLinkOneSidedAlltoAll supports top_k in {_SUPPORTED_TOP_K}, "
                f"got {params.top_k}"
            )

        self._module = get_nvlink_one_sided_alltoall_module()
        self._index = index = layout_constants()
        if self.ep_size > index.MAX_RANKS:
            raise ValueError(
                f"NVLinkOneSidedAlltoAll supports at most {index.MAX_RANKS} EP "
                f"ranks, got {self.ep_size}"
            )

        MnnvlMemory.initialize()
        MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)
        comm = MnnvlMemory.get_comm(mapping)
        device = torch.device("cuda", torch.cuda.current_device())
        self._enable_pdl = (
            device_support_pdl(device)
            if config.enable_pdl is None
            else config.enable_pdl
        )

        # Every rank must agree on CFT and on the Lamport transport (each changes
        # the workspace layout; the CFT endpoint exchange is collective), on an
        # explicit Lamport limit and, for the default one, on one SM count: one
        # collective for all. Each rank votes its own decision, so a rank that
        # turns a transport off turns it off on every rank. The CFT handle type
        # is the one the workspace allocation will start with; _bind_cft checks
        # the one it ends with. Its precheck is collective, so every rank runs it.
        fabric_handles = MnnvlMemory.uses_fabric_handles(comm, device.index)
        cft_reason = lamport_reason = num_sms = None
        lamport = _lamport_request(config)
        if lamport is False and config.lamport is None:
            logger.info(
                "NVLink one-sided Lamport transport off: cft=True keeps every step "
                "for CFT counted writes"
            )
        if config.cft is not False:
            cft_reason = _cft_unsupported_reason(device.index, fabric_handles)
        if lamport is not False:
            lamport_reason = _lamport_unsupported_reason(
                self._module, params, config, self.ep_size, index
            )
            if lamport_reason is None:
                num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        votes = comm.allgather(
            (
                config.cft is not False and cft_reason is None,
                lamport is not False and lamport_reason is None,
                num_sms,
                config.lamport_max_tokens,
            )
        )
        if len({v[3] for v in votes}) > 1:
            raise ValueError(
                "lamport_max_tokens must be the same on every EP rank, got "
                + ", ".join(sorted({repr(v[3]) for v in votes}))
            )
        cft_capable = all(v[0] for v in votes)
        lamport_capable = all(v[1] for v in votes)
        _log_unavailable("CFT counted writes", config.cft, cft_capable, cft_reason)
        _log_unavailable("Lamport transport", lamport, lamport_capable, lamport_reason)

        tokens = self.ep_size * params.max_tokens_per_rank
        alignment = index.WORKSPACE_ALIGNMENT
        # Each dispatch payload (activations, scale factors, expert ids,
        # weights) starts on its own aligned boundary.
        dispatch_bytes = (
            tokens
            * (
                params.dispatch_bytes_per_token
                + 2 * params.top_k * 4
                + config.extra_payload_bytes_per_token
            )
            + index.MAX_PAYLOADS * alignment
        )
        # Expert outputs come back unquantized, at least 16 bits wide; the FP8
        # wire format only applies to the CFT receive inbox.
        combine_element_size = max(params.token_dtype.itemsize, 2)
        combine_input_bytes = tokens * params.hidden_size * combine_element_size
        combine_recv_bytes = (
            tokens
            * params.hidden_size
            * (1 if config.use_low_precision_combine else combine_element_size)
            if cft_capable
            else 0
        )
        metainfo = torch.empty(index.NUM_METAINFO_FIELDS, dtype=torch.int64)
        self._module.moe_a2a_get_workspace_layout(
            self.ep_size,
            params.max_tokens_per_rank,
            params.top_k,
            dispatch_bytes,
            combine_input_bytes,
            combine_recv_bytes,
            config.eplb_stats_num_experts,
            cft_capable,
            metainfo,
        )
        # Largest step of the Lamport transport; 0 when the instance does not use it.
        self.lamport_max_tokens = 0
        self._lamport_format: Optional[str] = None
        self._lamport_row: Optional[_LamportRow] = None
        self._lamport_workspace_bytes = 0
        if lamport_capable:
            lamport_format = _LAMPORT_FORMATS[
                getattr(params.dispatch_format, "name", None)
            ]
            row = _lamport_row(self._module, index, lamport_format, params)
            if config.lamport_max_tokens is None:  # every rank voted its SM count
                max_tokens = _default_lamport_max_tokens(
                    self.ep_size, min(v[2] for v in votes), row, index
                )
            else:
                max_tokens = min(
                    config.lamport_max_tokens, index.LAMPORT_MAX_TOKENS_PER_RANK
                )
            max_tokens = min(max_tokens, params.max_tokens_per_rank)
            size_before = int(metainfo[index.WORKSPACE_SIZE_INDEX])
            self._module.moe_a2a_lamport_extend_workspace_layout(
                metainfo,
                max_tokens,
                getattr(index, f"LAMPORT_FORMAT_{lamport_format}"),
                params.hidden_size,
            )
            self.lamport_max_tokens = max_tokens
            self._lamport_format = lamport_format
            self._lamport_row = row
            self._lamport_workspace_bytes = (
                int(metainfo[index.WORKSPACE_SIZE_INDEX]) - size_before
            )
        self._state = self._acquire_workspace(mapping, comm, metainfo, cft_capable)
        self.workspace: "torch.Tensor" = self._state["workspace"]
        self.metainfo: "torch.Tensor" = self._state["metainfo"]
        # Contiguous bytes of this rank's slice of the workspace.
        self._rank_bytes = self.workspace[self.ep_rank]
        self._round: Optional[dict[str, Any]] = None

    def _acquire_workspace(
        self,
        mapping: Any,
        comm: "CommBackend",
        metainfo: "torch.Tensor",
        cft_capable: bool,
    ) -> dict[str, Any]:
        import torch

        from ......comm.mnnvl import MnnvlMemory

        key = (self.ep_rank, self.ep_size, tuple(metainfo.tolist()))
        state = self._WORKSPACES.get(key)
        if state is None:
            size = int(metainfo[self._index.WORKSPACE_SIZE_INDEX])
            mnnvl_mem = MnnvlMemory(mapping, size)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            self._module.moe_a2a_initialize(
                workspace, metainfo, self.ep_rank, self.ep_size
            )
            # No peer may publish into this workspace until every rank has
            # cleared its own slice.
            comm.barrier()
            if self.lamport_max_tokens and self.ep_rank == 0:
                logger.info(
                    "NVLink one-sided Lamport transport: steps of up to %d tokens per "
                    "rank, %.1f MiB of workspace per rank",
                    self.lamport_max_tokens,
                    self._lamport_workspace_bytes / (1 << 20),
                )
            state = {
                "key": key,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                "views": {},
                "refcount": 0,
                # Instance whose dispatch awaits its combine on this workspace.
                "round_owner": None,
                "cft_ready": cft_capable
                and self._bind_cft(key, comm, mnnvl_mem, workspace, size),
            }
            self._WORKSPACES[key] = state
        state["refcount"] += 1
        return state

    def _bind_cft(
        self,
        key: tuple,
        comm: "CommBackend",
        mnnvl_mem: Any,
        workspace: "torch.Tensor",
        size: int,
    ) -> bool:
        """Bind this workspace to the process's CFT logical endpoints.

        Collective over the EP group; every rank returns the same result.
        """
        cls = type(self)
        if not mnnvl_mem.fabric_handles:
            logger.warning(
                "NVLink one-sided CFT counted writes disabled: the workspace fell "
                "back to POSIX file descriptors"
            )
            return False
        if cls._CFT_WORKSPACE_KEY is not None:
            logger.info(
                "NVLink one-sided CFT counted writes are bound to another workspace "
                "of this process; this one uses the fence path"
            )
            return False
        if not self._create_cft_endpoints(comm, mnnvl_mem, workspace, size):
            logger.warning(
                "NVLink one-sided CFT counted writes disabled: creating the logical "
                "endpoints failed on at least one EP rank"
            )
            return False
        cls._CFT_WORKSPACE_KEY = key
        return True

    def _create_cft_endpoints(
        self,
        comm: "CommBackend",
        mnnvl_mem: Any,
        workspace: "torch.Tensor",
        size: int,
    ) -> bool:
        """Create this rank's logical endpoint on its workspace slice and
        import every peer's.

        Collective over the EP group; every rank returns the same result and
        a failure leaves no endpoint behind.
        """
        import torch

        from ......comm.mnnvl import all_ranks_agree

        module = self._module
        handle = torch.empty(module.moe_a2a_cft_handle_bytes(), dtype=torch.uint8)
        created = module.moe_a2a_cft_create_endpoint(
            workspace,
            mnnvl_mem.local_mem_handle,
            size,
            self.ep_rank,
            self.ep_size,
            handle,
        )
        bound = all_ranks_agree(comm, created)
        if bound:
            handles = comm.allgather(handle.numpy().tobytes())
            all_handles = torch.frombuffer(
                bytearray(b"".join(handles)), dtype=torch.uint8
            )
            bound = all_ranks_agree(
                comm,
                module.moe_a2a_cft_import_endpoints(
                    workspace, self.ep_rank, all_handles
                ),
            )
        if not bound:
            module.moe_a2a_cft_destroy(workspace, self.ep_rank)
            return False
        comm.barrier()
        return True

    def _cft_endpoint_ids(self) -> "torch.Tensor":
        """Logical-endpoint ID per peer rank that CFT launches bake in."""
        import torch

        ids = torch.empty(self.ep_size, dtype=torch.int64)
        self._module.moe_a2a_cft_endpoint_ids(self.workspace, self.ep_rank, ids)
        return ids

    @classmethod
    def is_platform_supported(cls) -> bool:
        return nvlink_platform_supported()

    @property
    def _cft_enabled(self) -> bool:
        """Whether CFT counted writes are available to this instance's steps."""
        return self._state is not None and self._state["cft_ready"]

    def active_rank_mask(self, active_ranks: Sequence[int]) -> "torch.Tensor":
        """CPU ``uint64`` bitmask of ``active_ranks`` for ``active_rank_mask``."""
        import torch

        mask = 0
        for rank in active_ranks:
            rank = int(rank)
            if not 0 <= rank < self.ep_size:
                raise ValueError(f"rank {rank} out of range [0, {self.ep_size})")
            mask |= 1 << rank
        words = (self._index.MAX_RANKS + 63) // 64
        return torch.tensor(
            [(mask >> (64 * word)) & 0xFFFFFFFFFFFFFFFF for word in range(words)],
            dtype=torch.uint64,
        )

    def _workspace_view(
        self, offset: int, dtype: "torch.dtype", shape: tuple[int, ...]
    ) -> "torch.Tensor":
        """Cached ``shape`` view of this rank's workspace bytes at ``offset``."""
        views = self._state["views"]
        key = (offset, dtype, shape)
        view = views.get(key)
        if view is None:
            numel = 1
            for extent in shape:
                numel *= extent
            nbytes = numel * dtype.itemsize
            view = self._rank_bytes.narrow(0, offset, nbytes).view(dtype).view(shape)
            views[key] = view
        return view

    @property
    def lamport_enabled(self) -> bool:
        """Whether the Lamport transport is available to this instance's steps."""
        return self._state is not None and self.lamport_max_tokens > 0

    def _lamport_carries(
        self,
        tokens_per_rank: int,
        hidden_states: "torch.Tensor",
        hidden_states_scale: "torch.Tensor | None",
        topk_weights: "torch.Tensor | None",
        eplb_local_stats: "torch.Tensor | None",
    ) -> bool:
        """Whether the Lamport transport can carry this step: few enough tokens,
        no EPLB statistics and payloads in the instance's format.

        Depends only on values every rank passes alike, so all ranks take the
        same transport.
        """
        if tokens_per_rank > self.lamport_max_tokens or eplb_local_stats is not None:
            return False
        row = self._lamport_row

        def row_bytes(tensor: "torch.Tensor | None") -> int:
            if tensor is None:
                return 0
            return tensor.shape[1] * tensor.element_size() if tensor.dim() == 2 else -1

        if (row_bytes(hidden_states), row_bytes(hidden_states_scale)) != (
            row.hidden_bytes,
            row.scale_bytes,
        ):
            return False
        # Senders rewrite payload words equal to a sentinel the way the format
        # keeps its values (NaN stays NaN), so the dtype must be the format's.
        if self._lamport_format == "BF16":
            import torch

            if hidden_states.dtype != torch.bfloat16:
                return False
        elif hidden_states.element_size() != 1:  # quantized values are bytes
            return False
        return topk_weights is None or (
            topk_weights.dim() == 2
            and (topk_weights.shape[1], topk_weights.element_size())
            == (self.params.top_k, 4)
        )

    def _dispatch_transports(
        self,
        tokens_per_rank: int,
        payloads: "Sequence[torch.Tensor]",
        hidden_states: "torch.Tensor",
        hidden_states_scale: "torch.Tensor | None",
        topk_weights: "torch.Tensor | None",
        eplb_local_stats: "torch.Tensor | None",
    ) -> tuple[bool, bool]:
        """``(use_lamport, use_cft)`` of a dispatch.

        The Lamport transport takes every step it can carry, ahead of CFT: its
        combine receives and reduces each token in the same kernel as it lands.
        CFT counted writes take the other steps within their token limit when
        every payload row meets their alignment; the fence path the rest.
        """
        use_lamport = self._lamport_carries(
            tokens_per_rank,
            hidden_states,
            hidden_states_scale,
            topk_weights,
            eplb_local_stats,
        )
        use_cft = (
            not use_lamport
            and self._use_cft(tokens_per_rank, self.config.cft_max_tokens_for_dispatch)
            and all(
                p.shape[1] * p.element_size() % _CFT_ALIGNMENT_BYTES == 0
                for p in payloads
            )
        )
        return use_lamport, use_cft

    def _use_cft(self, tokens_per_rank: int, max_tokens: int) -> bool:
        if not self._cft_enabled:
            return False
        return self.config.cft is True or tokens_per_rank <= max_tokens

    def _combine_uses_cft(
        self, tokens_per_rank: int, hidden: int, element_size: int
    ) -> bool:
        """Whether a non-Lamport combine of this step uses CFT counted writes."""
        wire_bytes = hidden * (
            1 if self.config.use_low_precision_combine else element_size
        )
        return (
            self._use_cft(tokens_per_rank, self.config.cft_max_tokens_for_combine)
            and wire_bytes % _CFT_ALIGNMENT_BYTES == 0
        )

    def _check_rank_mask(self, active_rank_mask: "torch.Tensor | None") -> None:
        import torch

        if not self.config.enable_rank_mask:
            if active_rank_mask is not None:
                raise ValueError("active_rank_mask requires enable_rank_mask=True")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NVLinkOneSidedAlltoAll with enable_rank_mask=True cannot be "
                "captured into a CUDA graph: the active-rank mask is fixed at "
                "capture time."
            )

    def dispatch(
        self,
        hidden_states: "torch.Tensor",
        topk_ids: "torch.Tensor",
        topk_weights: "torch.Tensor | None" = None,
        *,
        hidden_states_scale: "torch.Tensor | None" = None,
        runtime_max_tokens_per_rank: Optional[int] = None,
        eplb_local_stats: "torch.Tensor | None" = None,
        active_rank_mask: "torch.Tensor | None" = None,
    ) -> MoEEpDispatchResult:
        """See :meth:`MoEEpCommunication.dispatch`.

        ``active_rank_mask`` is a CPU ``uint64`` mask from
        :meth:`active_rank_mask`; tokens routed to a masked-off rank are
        dropped. Requires ``enable_rank_mask=True``.

        A step takes the Lamport transport only if its
        ``runtime_max_tokens_per_rank`` (the capacity when ``None``) is at most
        ``lamport_max_tokens``; :class:`NVLinkOneSidedConfig` lists the other
        conditions.

        Every rank passes ``topk_weights`` in a step or none does (a rank
        without tokens passes an empty ``[0, top_k]`` tensor, not ``None``),
        as every rank passes the same ``runtime_max_tokens_per_rank``. A
        mismatch is not detected: on the fence path a rank that passes weights
        reads, for a peer that passed none, whatever an earlier step left in
        its weight rows; a Lamport step can also leave a stale weight row for a
        later step, or time out (``timeout_sec``).
        """
        import torch

        if self._state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        check_round_owner(self, self._state)
        self._check_rank_mask(active_rank_mask)
        params = self.params
        tokens_per_rank = self._resolve_runtime_max_tokens_per_rank(
            runtime_max_tokens_per_rank
        )
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        payloads = [hidden_states]
        if hidden_states_scale is not None:
            payloads.append(hidden_states_scale)
        expert_id_index = len(payloads)
        payloads.append(topk_ids)
        if topk_weights is not None:
            payloads.append(topk_weights)

        use_lamport, use_cft = self._dispatch_transports(
            tokens_per_rank,
            payloads,
            hidden_states,
            hidden_states_scale,
            topk_weights,
            eplb_local_stats,
        )
        recv_offsets, combine_offset, eplb_offset = self._module.moe_a2a_dispatch(
            topk_ids,
            payloads,
            self.workspace,
            self.metainfo,
            tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            params.top_k,
            params.num_experts,
            eplb_local_stats,
            use_cft,
            expert_id_index if use_cft or use_lamport else -1,
            params.invalid_expert_id,
            use_lamport,
            self.config.enable_rank_mask,
            active_rank_mask,
            self.config.timeout_sec,
            self._enable_pdl,
        )

        recv = [
            self._workspace_view(
                int(offset),
                payload.dtype,
                (self.ep_size, tokens_per_rank, payload.shape[1]),
            )
            for offset, payload in zip(recv_offsets, payloads, strict=True)
        ]
        if not (use_cft or use_lamport):
            # CFT and Lamport dispatches fill the padding rows' expert ids themselves.
            self._module.moe_a2a_sanitize_expert_ids(
                recv[expert_id_index],
                self.workspace,
                self.metainfo,
                self.ep_rank,
                params.invalid_expert_id,
                self._enable_pdl,
            )
        eplb_gathered_stats = None
        if eplb_local_stats is not None:
            eplb_gathered_stats = self._workspace_view(
                int(eplb_offset),
                torch.int32,
                (self.ep_size, eplb_local_stats.shape[0]),
            )
        self._round = {
            "tokens_per_rank": tokens_per_rank,
            "local_num_tokens": topk_ids.shape[0],
            "combine_offset": int(combine_offset),
            "lamport": use_lamport,
        }
        self._state["round_owner"] = self
        recv = [t.flatten(0, 1) for t in recv]
        return MoEEpDispatchResult(
            hidden_states=recv[0],
            hidden_states_scale=recv[1] if hidden_states_scale is not None else None,
            topk_ids=recv[expert_id_index],
            topk_weights=recv[expert_id_index + 1]
            if topk_weights is not None
            else None,
            tokens_per_rank=tokens_per_rank,
            eplb_gathered_stats=eplb_gathered_stats,
        )

    def get_combine_input_buffer(self, dtype: "torch.dtype") -> "torch.Tensor":
        """``[ep_size * tokens_per_rank, hidden_size]`` view of the combine
        input region of this rank's workspace."""
        if self._round is None:
            raise RuntimeError("get_combine_input_buffer called before dispatch")
        rows = self.ep_size * self._round["tokens_per_rank"]
        nbytes = rows * self.params.hidden_size * dtype.itemsize
        capacity = int(self.metainfo[self._index.COMBINE_INPUT_SIZE_INDEX])
        if nbytes > capacity:
            raise ValueError(
                f"a {dtype} combine input needs {nbytes} bytes, more than the "
                f"{capacity}-byte combine region"
            )
        return self._workspace_view(
            self._round["combine_offset"], dtype, (rows, self.params.hidden_size)
        )

    def combine(
        self,
        expert_output: "torch.Tensor",
        *,
        output: "torch.Tensor | None" = None,
        active_rank_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """See :meth:`MoEEpCommunication.combine`.

        ``active_rank_mask`` must match the mask passed to :meth:`dispatch`.
        """
        import torch

        state = self._round
        if state is None:
            raise RuntimeError("combine called before dispatch")
        self._check_rank_mask(active_rank_mask)
        tokens_per_rank = state["tokens_per_rank"]
        if expert_output.dim() == 2:
            payload = expert_output.view(self.ep_size, tokens_per_rank, -1)
        elif expert_output.dim() == 3:
            payload = expert_output
        else:
            raise ValueError(
                f"expert_output must be 2D or 3D, got shape {tuple(expert_output.shape)}"
            )
        hidden = payload.shape[-1]
        if output is None:
            output = torch.empty(
                state["local_num_tokens"],
                hidden,
                dtype=payload.dtype,
                device=payload.device,
            )
        # A Lamport dispatch is followed by a Lamport combine.
        use_lamport = state["lamport"]
        use_cft = not use_lamport and self._combine_uses_cft(
            tokens_per_rank, hidden, payload.element_size()
        )
        self._module.moe_a2a_combine(
            payload,
            state["local_num_tokens"],
            self.workspace,
            self.metainfo,
            tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.params.top_k,
            state["combine_offset"],
            False,
            self.config.use_low_precision_combine,
            use_cft,
            self.config.enable_rank_mask,
            active_rank_mask,
            output,
            self.config.timeout_sec,
            self._enable_pdl,
            use_lamport,
        )
        self._round = None
        self._state["round_owner"] = None
        return output

    def checkpoint_prepare(self) -> None:
        """Release the workspace's physical memory and CFT endpoints before a
        process checkpoint.

        A CUDA process checkpoint cannot save the workspace's MNNVL physical
        memory, which peer processes import, nor its CFT logical endpoints.
        This method releases both but keeps the workspace's virtual addresses
        reserved, so CUDA graphs captured on the workspace replay correctly
        once :meth:`checkpoint_restore` has backed it with new memory:

        .. code-block:: python

            communication.checkpoint_prepare()
            # checkpoint and restore the process
            communication.checkpoint_restore(comm_backend)

        Collective over the EP group: every rank calls it in the same order,
        after all work on the workspace has completed and only between
        rounds. Instances sharing the workspace may call it again, which does
        nothing. If it raises once unmapping has begun, the workspace cannot be
        reused: restart the rank rather than retry.
        """
        state = self._state
        if state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        if state["round_owner"] is not None:
            raise RuntimeError("checkpoint_prepare called between dispatch and combine")

        def release_cft() -> None:
            # Keep the endpoint IDs reserved: captured launches use them.
            if state["cft_ready"]:
                state["cft_endpoint_ids"] = self._cft_endpoint_ids()
                self._module.moe_a2a_cft_release_endpoints(self.workspace, self.ep_rank)

        unmap_mnnvl_memory(state["mnnvl_mem"], release_cft)

    def checkpoint_restore(self, comm_backend: "CommBackend") -> None:
        """Back the workspace with new memory after :meth:`checkpoint_prepare`,
        re-arm its Lamport regions and recreate its CFT endpoints.

        Collective over ``comm_backend``, a fresh backend that must span the
        EP group with the original ranks and world size. Every rank calls it
        in the same order. Does nothing when the workspace is mapped. The CFT
        endpoints are recreated under the IDs they had, which CUDA graphs
        captured before the checkpoint use; raises if that is not possible.
        If it raises once remapping has begun, the workspace cannot be reused:
        restart the rank rather than retry.
        """
        from ......comm.mnnvl import all_ranks_agree

        state = self._state
        if state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        if not remap_mnnvl_memory(state["mnnvl_mem"], comm_backend):
            return
        self._module.moe_a2a_initialize(
            self.workspace, self.metainfo, self.ep_rank, self.ep_size
        )
        # No peer may publish into this workspace until every rank has
        # cleared its own slice.
        comm_backend.barrier()
        self._round = None
        if not state["cft_ready"]:
            return
        previous_ids = state.pop("cft_endpoint_ids")
        size = int(self.metainfo[self._index.WORKSPACE_SIZE_INDEX])
        if not self._create_cft_endpoints(
            comm_backend, state["mnnvl_mem"], self.workspace, size
        ):
            state["cft_ready"] = False
            type(self)._CFT_WORKSPACE_KEY = None
            raise RuntimeError(
                "Could not recreate the NVLink one-sided CFT logical endpoints after "
                "the checkpoint; CUDA graphs that use CFT are not safe to replay"
            )
        if not all_ranks_agree(
            comm_backend, bool((self._cft_endpoint_ids() == previous_ids).all())
        ):
            raise RuntimeError(
                "NVLink one-sided CFT logical-endpoint IDs changed across the "
                "checkpoint; CUDA graphs captured before it are not safe to replay"
            )

    def destroy(self) -> None:
        """Release this instance's share of the workspace.

        The last instance on a workspace frees it, together with its CFT
        endpoints; every rank must have finished its last combine on it.
        """
        state = self._state
        if state is None:
            return
        import torch

        if state["round_owner"] is self:
            state["round_owner"] = None
        self._state = None
        self._round = None
        self.workspace = None
        self.metainfo = None
        self._rank_bytes = None
        state["refcount"] -= 1
        if state["refcount"] > 0:
            return
        torch.cuda.synchronize()
        cls = type(self)
        if state["cft_ready"]:
            self._module.moe_a2a_cft_destroy(state["workspace"], self.ep_rank)
            cls._CFT_WORKSPACE_KEY = None
        cls._WORKSPACES.pop(state["key"], None)
        state.clear()
