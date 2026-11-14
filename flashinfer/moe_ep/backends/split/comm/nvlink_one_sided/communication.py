"""NVLink one-sided MoE communication over MNNVL symmetric memory.

The dispatch kernel writes each token directly into the receive buffers of the
ranks that own its experts; the combine kernel reads the expert outputs back
from those ranks and reduces them locally. A step synchronizes by one of two
protocols (``NVLinkOneSidedConfig.sync_protocol``): the fence protocol publishes
per-peer completion flags behind a system-scope fence; under the sentinel
protocol (``include/flashinfer/comm/moe_alltoall_sentinel.cuh``) every pushed
vector carries its own arrival, so neither direction waits for flags.

One symmetric workspace, laid out by ``moe_a2a_get_workspace_layout``, holds
the round state, the dispatch receive buffers and the combine buffers of a
rank. Its size grows with ``ep_size * max_tokens_per_rank``; the sentinel
regions, appended by ``moe_a2a_sentinel_extend_workspace_layout``, grow with
``ep_size * sentinel_max_tokens``. Instances with the same layout share one
workspace.
"""

from __future__ import annotations

import logging
import warnings
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
from . import config as config_module
from .config import NVLinkOneSidedConfig
from .kernels import get_nvlink_one_sided_alltoall_module, layout_constants

if TYPE_CHECKING:
    import torch

    from ......comm.abstractions import CommBackend
    from .....config import BootstrapConfig

logger = logging.getLogger(__name__)

_MAX_TIMEOUT_SEC = 24 * 60 * 60
# Top-k values the kernels are specialized for.
_SUPPORTED_TOP_K = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22)
# Sentinel protocol: its payload format (layout constant suffix) per dispatch format.
_SENTINEL_FORMATS = {
    None: "BF16",
    "BF16": "BF16",
    "MXFP8": "MXFP8",
    "DeepSeekFp8": "FP8_BLOCK128",
    "NVFP4": "NVFP4",
}


class _SentinelRow(NamedTuple):
    """Row geometry of a sentinel payload format, from the kernel module."""

    hidden_bytes: int  # one token's hidden states
    scale_bytes: int  # one token's scales (0 for BF16)
    blocks_per_token: int  # dispatch blocks per token slot
    bytes_per_row: int  # sentinel workspace bytes per (source rank, token) row


def _sentinel_row(
    module: Any, index: Any, sentinel_format: str, params: MoEEpCommParams
) -> _SentinelRow:
    """Raises if the kernels do not take rows of ``params.hidden_size``."""
    return _SentinelRow(
        *map(
            int,
            module.moe_a2a_sentinel_row_bytes(
                getattr(index, f"SENTINEL_FORMAT_{sentinel_format}"),
                params.hidden_size,
                params.top_k,
            ),
        )
    )


def _probe_sentinel_support(
    module: Any,
    params: MoEEpCommParams,
    config: NVLinkOneSidedConfig,
    ep_size: int,
    index: Any,
) -> Optional[str]:
    import torch

    if not module.moe_a2a_sentinel_available():
        return "the kernel module has no sentinel kernels for this GPU"
    if ep_size > index.SENTINEL_MAX_RANKS:
        return f"it supports at most {index.SENTINEL_MAX_RANKS} EP ranks"
    if params.top_k > index.SENTINEL_MAX_TOP_K:
        return f"it supports top_k <= {index.SENTINEL_MAX_TOP_K}"
    if params.token_dtype != torch.bfloat16:
        return "it combines BF16 expert outputs only"
    if config.use_low_precision_combine:
        return "it has no FP8 combine"
    if ((params.invalid_expert_id - index.SENTINEL_IDS_SENTINEL) & 0xFFFFFFFF) == 0:
        return "invalid_expert_id equals its expert-id sentinel as an int32"
    name = getattr(params.dispatch_format, "name", None)
    if name not in _SENTINEL_FORMATS:
        return f"it does not carry the dispatch format {name}"
    _sentinel_row(module, index, _SENTINEL_FORMATS[name], params)
    return None


def _sentinel_unsupported_reason(
    module: Any,
    params: MoEEpCommParams,
    config: NVLinkOneSidedConfig,
    ep_size: int,
    index: Any,
) -> Optional[str]:
    """Why this rank cannot use the sentinel protocol, or ``None`` if it can.

    Never raises: every rank must reach the collective protocol agreement, and
    a rank whose probe fails falls back to the fence protocol with the others.
    """
    try:
        return _probe_sentinel_support(module, params, config, ep_size, index)
    except Exception as exc:  # CUDA queries and the kernels' row checks alike
        return f"the sentinel support probe failed: {type(exc).__name__}: {exc}"


@register_communication("nvlink_one_sided")
class NVLinkOneSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink one-sided put/get kernels.

    ``hidden_states``, its optional scale factors, ``topk_ids`` and
    ``topk_weights`` travel as payloads of a single dispatch. Received rows
    beyond each source rank's token count get ``invalid_expert_id`` routing;
    after a sentinel step, row ``source_rank * tokens_per_rank + token`` holds
    that token and rows of tokens not routed here get ``invalid_expert_id``.
    See :class:`NVLinkOneSidedConfig` for the sync protocols and timeouts.
    """

    # Workspaces shared by instances with the same layout, keyed by
    # (ep_rank, ep_size, layout).
    _WORKSPACES: ClassVar[dict[tuple, dict[str, Any]]] = {}

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

        # Every rank must agree on the protocol and, under "auto", on an explicit
        # sentinel limit (they decide each step's protocol and the workspace
        # layout) and on whether all of them support the sentinel protocol: one
        # collective for all.
        protocol = config._effective_sync_protocol()
        sentinel_reason = None
        if protocol != "fence":
            sentinel_reason = _sentinel_unsupported_reason(
                self._module, params, config, self.ep_size, index
            )
        votes = comm.allgather(
            (
                protocol,
                config.lamport_max_tokens if protocol == "auto" else None,
                protocol != "fence" and sentinel_reason is None,
            )
        )
        for i, name in enumerate(("sync_protocol", "lamport_max_tokens")):
            if len({v[i] for v in votes}) > 1:
                raise ValueError(
                    f"{name} must be the same on every EP rank, got "
                    + ", ".join(sorted({repr(v[i]) for v in votes}))
                )
        sentinel_capable = all(v[2] for v in votes)
        if protocol != "fence" and not sentinel_capable:
            (logger.warning if protocol == "sentinel" else logger.info)(
                "NVLink one-sided sentinel protocol disabled, every step takes the "
                "fence protocol: %s",
                sentinel_reason or "another EP rank does not support it",
            )

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
        # Expert outputs come back unquantized, at least 16 bits wide.
        combine_element_size = max(params.token_dtype.itemsize, 2)
        combine_input_bytes = tokens * params.hidden_size * combine_element_size
        metainfo = torch.empty(index.NUM_METAINFO_FIELDS, dtype=torch.int64)
        # The module's CFT counted-write path stays unused: no CFT receive inbox
        # or counters.
        self._module.moe_a2a_get_workspace_layout(
            self.ep_size,
            params.max_tokens_per_rank,
            params.top_k,
            dispatch_bytes,
            combine_input_bytes,
            0,
            config.eplb_stats_num_experts,
            False,
            metainfo,
        )
        self._sentinel_max_tokens = 0
        self._sentinel_format: Optional[str] = None
        self._sentinel_row: Optional[_SentinelRow] = None
        self._sentinel_workspace_bytes = 0
        if protocol == "sentinel":
            max_tokens = params.max_tokens_per_rank
        elif config.lamport_max_tokens is not None:
            max_tokens = config.lamport_max_tokens
        else:
            max_tokens = config_module.AUTO_SENTINEL_MAX_TOKENS
        max_tokens = min(max_tokens, params.max_tokens_per_rank)
        if sentinel_capable and max_tokens > 0:
            sentinel_format = _SENTINEL_FORMATS[
                getattr(params.dispatch_format, "name", None)
            ]
            row = _sentinel_row(self._module, index, sentinel_format, params)
            size_before = int(metainfo[index.WORKSPACE_SIZE_INDEX])
            self._module.moe_a2a_sentinel_extend_workspace_layout(
                metainfo,
                max_tokens,
                getattr(index, f"SENTINEL_FORMAT_{sentinel_format}"),
                params.hidden_size,
            )
            self._sentinel_max_tokens = max_tokens
            self._sentinel_format = sentinel_format
            self._sentinel_row = row
            self._sentinel_workspace_bytes = (
                int(metainfo[index.WORKSPACE_SIZE_INDEX]) - size_before
            )
        self._state = self._acquire_workspace(mapping, comm, metainfo)
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
            if self.sentinel_max_tokens and self.ep_rank == 0:
                logger.info(
                    "NVLink one-sided sentinel protocol: steps of up to %d tokens per "
                    "rank, %.1f MiB of workspace per rank",
                    self.sentinel_max_tokens,
                    self._sentinel_workspace_bytes / (1 << 20),
                )
            state = {
                "key": key,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                "views": {},
                # Instances sharing the workspace, and CFT endpoints bound to it
                # (cft.CftEndpoints); the last one to drop its reference frees it.
                "refcount": 0,
                # Instance whose dispatch awaits its combine on this workspace.
                "round_owner": None,
            }
            self._WORKSPACES[key] = state
        state["refcount"] += 1
        return state

    @classmethod
    def is_platform_supported(cls) -> bool:
        return nvlink_platform_supported()

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
    def sentinel_max_tokens(self) -> int:
        """Largest step, in tokens per rank, that the sentinel protocol takes; 0
        when this instance does not use it. Fixed at construction: the
        workspace's sentinel regions are sized for it."""
        return self._sentinel_max_tokens

    @property
    def sentinel_enabled(self) -> bool:
        """Whether the sentinel protocol is available to this instance's steps."""
        return self._state is not None and self.sentinel_max_tokens > 0

    @property
    def lamport_enabled(self) -> bool:
        """Deprecated alias of :attr:`sentinel_enabled`."""
        warnings.warn(
            "NVLinkOneSidedAlltoAll.lamport_enabled is deprecated; use sentinel_enabled.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sentinel_enabled

    @property
    def lamport_max_tokens(self) -> int:
        """Deprecated alias of :attr:`sentinel_max_tokens`."""
        warnings.warn(
            "NVLinkOneSidedAlltoAll.lamport_max_tokens is deprecated; use "
            "sentinel_max_tokens.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sentinel_max_tokens

    def _sentinel_carries(
        self,
        tokens_per_rank: int,
        hidden_states: "torch.Tensor",
        hidden_states_scale: "torch.Tensor | None",
        topk_weights: "torch.Tensor | None",
        eplb_local_stats: "torch.Tensor | None",
    ) -> bool:
        """Whether this step takes the sentinel protocol: few enough tokens, no
        EPLB statistics and payloads in the instance's format. The fence
        protocol takes the other steps.

        Depends only on values every rank passes alike, so all ranks take the
        same protocol.
        """
        if tokens_per_rank > self.sentinel_max_tokens or eplb_local_stats is not None:
            return False
        row = self._sentinel_row

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
        if self._sentinel_format == "BF16":
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

        A step takes the sentinel protocol only if its
        ``runtime_max_tokens_per_rank`` (the capacity when ``None``) is at most
        ``sentinel_max_tokens``; :class:`NVLinkOneSidedConfig` lists the other
        conditions.

        Every rank passes ``topk_weights`` in a step or none does (a rank
        without tokens passes an empty ``[0, top_k]`` tensor, not ``None``),
        as every rank passes the same ``runtime_max_tokens_per_rank``. A
        mismatch is not detected: on the fence path a rank that passes weights
        reads, for a peer that passed none, whatever an earlier step left in
        its weight rows; a sentinel step can also leave a stale weight row for a
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

        use_sentinel = self._sentinel_carries(
            tokens_per_rank,
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
            False,
            expert_id_index if use_sentinel else -1,
            params.invalid_expert_id,
            use_sentinel,
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
        if not use_sentinel:
            # Sentinel dispatches fill the padding rows' expert ids themselves.
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
            "sentinel": use_sentinel,
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
        # A sentinel dispatch is followed by a sentinel combine.
        use_sentinel = state["sentinel"]
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
            False,
            self.config.enable_rank_mask,
            active_rank_mask,
            output,
            self.config.timeout_sec,
            self._enable_pdl,
            use_sentinel,
        )
        self._round = None
        self._state["round_owner"] = None
        return output

    def checkpoint_prepare(self) -> None:
        """Release the workspace's physical memory before a process checkpoint.

        A CUDA process checkpoint cannot save the workspace's MNNVL physical
        memory, which peer processes import. This method releases it but keeps
        the workspace's virtual addresses reserved, so CUDA graphs captured on
        the workspace replay correctly once :meth:`checkpoint_restore` has
        backed it with new memory:

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
        unmap_mnnvl_memory(state["mnnvl_mem"])

    def checkpoint_restore(self, comm_backend: "CommBackend") -> None:
        """Back the workspace with new memory after :meth:`checkpoint_prepare`
        and re-arm its sentinel regions.

        Collective over ``comm_backend``, a fresh backend that must span the
        EP group with the original ranks and world size. Every rank calls it
        in the same order. Does nothing when the workspace is mapped. If it
        raises once remapping has begun, the workspace cannot be reused:
        restart the rank rather than retry.
        """
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

    def destroy(self) -> None:
        """Release this instance's share of the workspace.

        The last instance on a workspace frees it, unless CFT endpoints are
        still bound to it; every rank must have finished its last combine on
        it.
        """
        state = self._state
        if state is None:
            return
        if state["round_owner"] is self:
            state["round_owner"] = None
        self._state = None
        self._round = None
        self.workspace = None
        self.metainfo = None
        self._rank_bytes = None
        self._release_workspace(state)

    @classmethod
    def _release_workspace(cls, state: dict[str, Any]) -> None:
        """Drop one reference to a workspace; the last one frees it."""
        import torch

        state["refcount"] -= 1
        if state["refcount"] > 0:
            return
        torch.cuda.synchronize()
        cls._WORKSPACES.pop(state["key"], None)
        state.clear()
