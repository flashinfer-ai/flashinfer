"""NVLink one-sided MoE communication over MNNVL symmetric memory.

The dispatch kernel writes each token directly into the receive buffers of the
ranks that own its experts; the combine kernel reads the expert outputs back
from those ranks and reduces them locally. Where the platform supports it,
either direction can instead use CFT counted writes: the sender pushes over the
NVLink fabric with ``fabric.try_put.counted`` and the receiver waits on
hardware byte counters instead of per-round completion flags.

One symmetric workspace, laid out by ``moe_a2a_get_workspace_layout``, holds
the round state, the dispatch receive buffers and the combine buffers of a
rank. Its size grows with ``ep_size * max_tokens_per_rank``. Instances with the
same layout share one workspace.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence

from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from ..nvlink_common import (
    mnnvl_mapping_and_config,
    nvlink_platform_supported,
    remap_mnnvl_memory,
    unmap_mnnvl_memory,
)
from .config import NVLinkOneSidedConfig
from .kernels import get_nvlink_one_sided_module, layout_constants

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
_SUPPORTED_TOP_K = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22)


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


def _cft_unsupported_reason(device_index: int) -> Optional[str]:
    """Why this rank cannot use CFT counted writes, or ``None`` if it can."""
    import torch

    from ......comm.mnnvl import cuda, is_mnnvl_fabric_supported

    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 10:
        return f"compute capability {major}.{minor} is below 10.0"
    branch = _driver_branch()
    if branch is None:
        return "the NVIDIA driver version could not be queried"
    if branch < _CFT_MIN_DRIVER_BRANCH:
        return f"driver branch {branch} is older than {_CFT_MIN_DRIVER_BRANCH}"
    try:
        if not is_mnnvl_fabric_supported(device_index):
            return "the workspace cannot be allocated with fabric handles"
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


@register_communication("nvlink_one_sided")
class NVLinkOneSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink one-sided put/get kernels.

    ``hidden_states``, its optional scale factors, ``topk_ids`` and
    ``topk_weights`` travel as payloads of a single dispatch. Received rows
    beyond each source rank's token count get ``invalid_expert_id`` routing.
    See :class:`NVLinkOneSidedConfig` for CFT counted writes and timeouts.
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

        from ......comm.mnnvl import MnnvlMemory, all_ranks_agree
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
        if (
            config.cft_max_tokens_for_dispatch < 0
            or config.cft_max_tokens_for_combine < 0
        ):
            raise ValueError("CFT token-count limits must be non-negative")
        if params.top_k not in _SUPPORTED_TOP_K:
            raise ValueError(
                f"NVLinkOneSidedAlltoAll supports top_k in {_SUPPORTED_TOP_K}, "
                f"got {params.top_k}"
            )

        self._module = get_nvlink_one_sided_module()
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
        self._enable_pdl = device_support_pdl(device)

        # Every rank must agree on CFT: it changes the workspace layout and
        # the endpoint exchange is collective.
        reason = None
        if config.cft is not False:
            reason = _cft_unsupported_reason(device.index)
        cft_capable = all_ranks_agree(comm, config.cft is not False and reason is None)
        if config.cft is not False and not cft_capable:
            log = logger.warning if config.cft else logger.info
            log(
                "NVLink one-sided CFT counted writes disabled: %s",
                reason or "another EP rank does not support them",
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
            state = {
                "key": key,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                "views": {},
                "refcount": 0,
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
    def cft_enabled(self) -> bool:
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

    def _use_cft(self, tokens_per_rank: int, max_tokens: int) -> bool:
        if not self.cft_enabled:
            return False
        return self.config.cft is True or tokens_per_rank <= max_tokens

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
        max_tokens_per_rank: Optional[int] = None,
        eplb_local_stats: "torch.Tensor | None" = None,
        active_rank_mask: "torch.Tensor | None" = None,
    ) -> MoEEpDispatchResult:
        """See :meth:`MoEEpCommunication.dispatch`.

        ``active_rank_mask`` is a CPU ``uint64`` mask from
        :meth:`active_rank_mask`; tokens routed to a masked-off rank are
        dropped. Requires ``enable_rank_mask=True``.
        """
        import torch

        if self._state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        if self._round is not None:
            raise RuntimeError("dispatch called twice without an intervening combine")
        self._check_rank_mask(active_rank_mask)
        params = self.params
        tokens_per_rank = (
            params.max_tokens_per_rank
            if max_tokens_per_rank is None
            else max_tokens_per_rank
        )
        if not 0 < tokens_per_rank <= params.max_tokens_per_rank:
            raise ValueError(
                f"max_tokens_per_rank={tokens_per_rank} must be in "
                f"(0, {params.max_tokens_per_rank}]"
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

        use_cft = self._use_cft(
            tokens_per_rank, self.config.cft_max_tokens_for_dispatch
        ) and all(
            p.shape[1] * p.element_size() % _CFT_ALIGNMENT_BYTES == 0 for p in payloads
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
            expert_id_index if use_cft else -1,
            params.invalid_expert_id,
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
        if not use_cft:
            # CFT dispatch fills the padding rows' expert ids itself.
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
        }
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
        wire_bytes = hidden * (
            1 if self.config.use_low_precision_combine else payload.element_size()
        )
        use_cft = (
            self._use_cft(tokens_per_rank, self.config.cft_max_tokens_for_combine)
            and wire_bytes % _CFT_ALIGNMENT_BYTES == 0
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
        )
        self._round = None
        return output

    def checkpoint_prepare(self) -> None:
        """Release the workspace's physical memory and CFT endpoints, e.g. so
        that the process can be checkpointed.

        The virtual addresses stay reserved: CUDA graphs captured on the
        workspace replay correctly after :meth:`checkpoint_restore`. Collective
        over the EP group and only valid between rounds. Instances sharing the
        workspace may call it again, which does nothing.
        """
        state = self._state
        if state is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        if self._round is not None:
            raise RuntimeError("checkpoint_prepare called between dispatch and combine")

        def release_cft() -> None:
            # Keep the endpoint IDs reserved: captured launches use them.
            if state["cft_ready"]:
                state["cft_endpoint_ids"] = self._cft_endpoint_ids()
                self._module.moe_a2a_cft_release_endpoints(self.workspace, self.ep_rank)

        unmap_mnnvl_memory(state["mnnvl_mem"], release_cft)

    def checkpoint_restore(self, comm_backend: "CommBackend") -> None:
        """Back the workspace with new memory after :meth:`checkpoint_prepare`
        and recreate its CFT endpoints.

        Collective over ``comm_backend``, which must span the EP group with
        the original ranks. Does nothing when the workspace is mapped. The CFT
        endpoints are recreated under the IDs they had, which CUDA graphs
        captured before the checkpoint use; raises if that is not possible.
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
