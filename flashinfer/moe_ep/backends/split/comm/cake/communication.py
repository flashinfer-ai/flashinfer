"""NVLink one-sided MoE communication with the generated Cake kernels.

The Cake kernels are generated for Blackwell (compute capability 10.0 or
10.3) and implement the one-sided put/get protocol of the ``moe_a2a_*`` ops in
:mod:`flashinfer.comm.trtllm_moe_alltoall`, selected with ``backend="cake"``:
the dispatch kernel writes each token into the receive buffers of the ranks
that own its experts and the combine kernel reads the expert outputs back and
reduces them locally. The symmetric workspace grows with
``ep_size * max_tokens_per_rank``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from ..nvlink_common import mnnvl_mapping_and_config, nvlink_platform_supported
from .config import CakeAlltoAllConfig

if TYPE_CHECKING:
    import torch

    from ......comm.mapping import Mapping
    from .....config import BootstrapConfig

logger = logging.getLogger(__name__)

_CAKE_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_BACKEND = "cake"


@register_communication("cake")
class CakeAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with the generated Cake kernels.

    ``hidden_states``, its optional scale factors, ``topk_ids`` and
    ``topk_weights`` travel as payloads of a single dispatch. Received rows
    beyond each source rank's token count get ``invalid_expert_id`` routing.
    """

    # Workspaces shared by instances with the same geometry, keyed by
    # (ep_rank, ep_size, max_tokens_per_rank, bytes per rank, EPLB experts).
    _WORKSPACES: ClassVar[dict[tuple, dict[str, Any]]] = {}

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[CakeAlltoAllConfig] = None,
    ) -> None:
        self.config = config = CakeAlltoAllConfig() if config is None else config
        # Collective platform check across the EP group; it runs before the
        # base class's local check so that no rank fails alone.
        mapping, mnnvl_config = mnnvl_mapping_and_config(bootstrap, config.comm_backend)
        super().__init__(bootstrap, params)
        from ......comm.mnnvl import MnnvlMemory
        from ......comm.trtllm_moe_alltoall import (
            moe_a2a_get_workspace_size_per_rank,
        )

        if config.extra_payload_bytes_per_token < 0:
            raise ValueError("extra_payload_bytes_per_token must be non-negative")
        if config.eplb_stats_num_experts < 0:
            raise ValueError("eplb_stats_num_experts must be non-negative")

        # Dispatch carries the activations, int32 expert ids and FP32 weights.
        dispatch_bytes_per_token = (
            params.dispatch_bytes_per_token
            + params.top_k * 4
            + params.top_k * 4
            + config.extra_payload_bytes_per_token
        )
        # Expert outputs come back unquantized, at least 16 bits wide.
        combine_bytes_per_token = params.hidden_size * max(
            params.token_dtype.itemsize, 2
        )
        workspace_size_per_rank = moe_a2a_get_workspace_size_per_rank(
            self.ep_size,
            params.max_tokens_per_rank,
            dispatch_bytes_per_token,
            combine_bytes_per_token,
            config.eplb_stats_num_experts,
            backend=_BACKEND,
        )

        MnnvlMemory.initialize()
        if mnnvl_config is not None:
            MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)  # type: ignore[attr-defined]
        self._state: Optional[dict[str, Any]] = self._acquire_workspace(
            mapping, workspace_size_per_rank
        )
        self.workspace = self._state["workspace"]
        self.metainfo = self._state["metainfo"]
        self._round: Optional[dict[str, Any]] = None

    @classmethod
    def is_platform_supported(cls) -> bool:
        if not nvlink_platform_supported():
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

    def _acquire_workspace(
        self, mapping: "Mapping", workspace_size_per_rank: int
    ) -> dict[str, Any]:
        import torch

        from ......comm.mnnvl import MnnvlMemory
        from ......comm.trtllm_moe_alltoall import moe_a2a_initialize

        key = (
            self.ep_rank,
            self.ep_size,
            self.params.max_tokens_per_rank,
            workspace_size_per_rank,
            self.config.eplb_stats_num_experts,
        )
        state = self._WORKSPACES.get(key)
        if state is None:
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.params.max_tokens_per_rank,
                self.config.eplb_stats_num_experts,
                backend=_BACKEND,
            )
            # No peer may publish into this workspace until every rank has
            # cleared its own slice.
            MnnvlMemory.allocated_map[mnnvl_mem.ptr].comm.barrier()
            state = {
                "key": key,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                # Receive views of the shared workspace, reused across rounds.
                "views": {},
                "refcount": 0,
            }
            self._WORKSPACES[key] = state
        state["refcount"] += 1
        return state

    def _live_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("CakeAlltoAll has been destroyed")
        return self._state

    def _check_rank_mask(self, active_rank_mask: "torch.Tensor | None") -> None:
        if active_rank_mask is not None and not self.config.enable_rank_mask:
            raise ValueError("active_rank_mask requires enable_rank_mask=True")

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
        :func:`flashinfer.comm.moe_a2a_active_rank_mask`; tokens routed to a
        masked-off rank are dropped. Requires ``enable_rank_mask=True``.
        """
        import torch

        from ......comm.trtllm_moe_alltoall import (
            moe_a2a_dispatch,
            moe_a2a_sanitize_expert_ids,
        )

        state = self._live_state()
        if self._round is not None:
            raise RuntimeError("dispatch called twice without an intervening combine")
        tokens_per_rank = (
            self.params.max_tokens_per_rank
            if max_tokens_per_rank is None
            else max_tokens_per_rank
        )
        if not 0 < tokens_per_rank <= self.params.max_tokens_per_rank:
            raise ValueError(
                f"max_tokens_per_rank={tokens_per_rank} must be in "
                f"(0, {self.params.max_tokens_per_rank}]"
            )
        if eplb_local_stats is not None and eplb_local_stats.shape != (
            self.config.eplb_stats_num_experts,
        ):
            raise ValueError(
                "eplb_local_stats must have shape "
                f"({self.config.eplb_stats_num_experts},), got "
                f"{tuple(eplb_local_stats.shape)}"
            )
        self._check_rank_mask(active_rank_mask)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        payloads = [hidden_states]
        if hidden_states_scale is not None:
            payloads.append(hidden_states_scale)
        expert_id_index = len(payloads)
        payloads.append(topk_ids)
        if topk_weights is not None:
            payloads.append(topk_weights)

        recv, combine_offset, eplb_gathered_stats = moe_a2a_dispatch(
            topk_ids,
            payloads,
            self.workspace,
            self.metainfo,
            tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.params.top_k,
            self.params.num_experts,
            eplb_local_stats=eplb_local_stats,
            enable_rank_mask=self.config.enable_rank_mask,
            active_rank_mask=active_rank_mask,
            backend=_BACKEND,
            recv_view_cache=state["views"],
        )
        moe_a2a_sanitize_expert_ids(
            recv[expert_id_index],
            self.workspace,
            self.metainfo,
            self.ep_rank,
            self.params.invalid_expert_id,
            backend=_BACKEND,
        )
        self._round = {
            "local_num_tokens": topk_ids.shape[0],
            "tokens_per_rank": tokens_per_rank,
            "combine_offset": combine_offset,
            "combine_buffer": None,
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
        payload region of this rank's workspace."""
        from ......comm.trtllm_moe_alltoall import (
            moe_a2a_wrap_payload_tensor_in_workspace,
        )

        self._live_state()
        round_state = self._round
        if round_state is None:
            raise RuntimeError("get_combine_input_buffer called before dispatch")
        rows = self.ep_size * round_state["tokens_per_rank"]
        start = round_state["combine_offset"]
        end = start + rows * self.params.hidden_size * dtype.itemsize
        buffer = moe_a2a_wrap_payload_tensor_in_workspace(
            self.workspace[self.ep_rank, :], [rows], start, end, dtype
        )
        round_state["combine_buffer"] = buffer
        return buffer

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
        from ......comm.trtllm_moe_alltoall import moe_a2a_combine

        self._live_state()
        round_state = self._round
        if round_state is None:
            raise RuntimeError("combine called before dispatch")
        self._check_rank_mask(active_rank_mask)
        tokens_per_rank = round_state["tokens_per_rank"]
        if expert_output.dim() == 2:
            payload = expert_output.view(self.ep_size, tokens_per_rank, -1)
        elif expert_output.dim() == 3:
            payload = expert_output
        else:
            raise ValueError(
                f"expert_output must be 2D or 3D, got shape {tuple(expert_output.shape)}"
            )
        buffer = round_state["combine_buffer"]
        in_workspace = (
            buffer is not None and expert_output.data_ptr() == buffer.data_ptr()
        )
        combined = moe_a2a_combine(
            payload,
            round_state["local_num_tokens"],
            self.workspace,
            self.metainfo,
            tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.params.top_k,
            round_state["combine_offset"],
            in_workspace,
            use_low_precision=self.config.use_low_precision_combine,
            enable_rank_mask=self.config.enable_rank_mask,
            active_rank_mask=active_rank_mask,
            output=output,
            backend=_BACKEND,
        )
        self._round = None
        return combined

    def destroy(self) -> None:
        """Release this instance's share of the workspace.

        The last instance on a workspace frees it; every rank must have
        finished its last combine on it.
        """
        state = self._state
        if state is None:
            return
        import torch

        self._state = None
        self._round = None
        self.workspace = None
        self.metainfo = None
        state["refcount"] -= 1
        if state["refcount"] > 0:
            return
        torch.cuda.synchronize()
        type(self)._WORKSPACES.pop(state["key"], None)
        state.clear()
