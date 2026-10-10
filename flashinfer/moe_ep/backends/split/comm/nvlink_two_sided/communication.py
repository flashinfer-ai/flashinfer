"""NVLink two-sided MoE communication over MNNVL FIFO channels.

Sender and receiver kernels exchange packets through per-peer FIFOs in
symmetric memory, like a collective. A prepare step first exchanges the
routing (expert ids, weights, EPLB statistics) and per-rank token counts; the
token payloads then follow as all-to-all-v transfers. The symmetric footprint
depends on the number of channels, not on the token count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from ..nvlink_common import mnnvl_mapping_and_config, nvlink_platform_supported
from .config import NVLinkTwoSidedConfig

if TYPE_CHECKING:
    import torch

    from ......comm.mapping import Mapping
    from .....config import BootstrapConfig


@register_communication("nvlink_two_sided")
class NVLinkTwoSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink two-sided all-to-all-v kernels.

    Requires ``num_experts`` divisible by 4. EPLB statistics, when passed to
    :meth:`dispatch`, must cover all ``num_experts`` experts.
    """

    # FIFO and prepare workspaces keyed by (ep_rank, ep_size). The FIFO
    # positions persist in the workspace across calls, so a workspace lives as
    # long as the process and every instance of the same group shares it.
    _WORKSPACES: ClassVar[dict[tuple[int, int], dict[str, Any]]] = {}

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NVLinkTwoSidedConfig] = None,
    ) -> None:
        self.config = NVLinkTwoSidedConfig() if config is None else config
        # Collective platform check across the EP group; it runs before the
        # base class's local check so that no rank fails alone.
        mapping, mnnvl_config = mnnvl_mapping_and_config(
            bootstrap, self.config.comm_backend
        )
        super().__init__(bootstrap, params)
        from ......comm.mnnvl import MnnvlMemory

        if params.num_experts % 4 != 0:
            raise ValueError(
                "NVLinkTwoSidedAlltoAll requires num_experts divisible by 4, "
                f"got {params.num_experts}"
            )
        MnnvlMemory.initialize()
        if mnnvl_config is not None:
            MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)  # type: ignore[attr-defined]
        workspaces = self._acquire_workspaces(mapping)
        self._workspace: "torch.Tensor | None" = workspaces["workspace"]
        self._prepare_workspace: "torch.Tensor | None" = workspaces["prepare_workspace"]
        self._state: Optional[dict[str, Any]] = None

    @classmethod
    def is_platform_supported(cls) -> bool:
        return nvlink_platform_supported()

    def _acquire_workspaces(self, mapping: "Mapping") -> dict[str, Any]:
        import torch

        from ......comm.mnnvl import MnnvlMemory
        from ......comm.trtllm_alltoall import (
            get_moe_commworkspace_size_per_rank,
            get_moe_prepare_workspace_size_per_rank,
        )

        key = (self.ep_rank, self.ep_size)
        workspaces = self._WORKSPACES.get(key)
        if workspaces is None:
            memory = MnnvlMemory(
                mapping, get_moe_commworkspace_size_per_rank(self.ep_size)
            )
            prepare_memory = MnnvlMemory(
                mapping, get_moe_prepare_workspace_size_per_rank(self.ep_size)
            )
            workspaces = {
                "memory": memory,
                "prepare_memory": prepare_memory,
                "workspace": memory.as_torch_strided_tensor(torch.uint64),
                "prepare_workspace": prepare_memory.as_torch_strided_tensor(
                    torch.uint64
                ),
            }
            self._WORKSPACES[key] = workspaces
        return workspaces

    def _alltoallv(
        self,
        x: "torch.Tensor",
        output: "torch.Tensor",
        send_cumsum: "torch.Tensor",
        send_indices: "torch.Tensor",
        recv_cumsum: "torch.Tensor",
        recv_indices: "torch.Tensor",
    ) -> "torch.Tensor":
        """Move the rows of ``x`` selected by the send indices to the receive
        indices of ``output`` on the peer ranks; rows nobody sends to keep
        their contents."""
        from ......comm.trtllm_alltoall import moe_comm

        if x.dim() != 2:
            raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
        moe_comm(
            x,
            send_cumsum,
            send_indices,
            output,
            recv_cumsum,
            recv_indices,
            self._workspace,
            self.ep_rank,
            self.ep_size,
        )
        return output

    def dispatch(
        self,
        hidden_states: "torch.Tensor",
        topk_ids: "torch.Tensor",
        topk_weights: "torch.Tensor | None" = None,
        *,
        hidden_states_scale: "torch.Tensor | None" = None,
        max_tokens_per_rank: Optional[int] = None,
        eplb_local_stats: "torch.Tensor | None" = None,
    ) -> MoEEpDispatchResult:
        import torch

        from ......comm.trtllm_alltoall import moe_prepare

        if self._workspace is None:
            raise RuntimeError("NVLinkTwoSidedAlltoAll has been destroyed")
        if self._state is not None:
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
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if topk_weights is not None and topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)

        num_experts = self.params.num_experts
        (
            recv_topk_ids,
            recv_topk_weights,
            send_cumsum,
            send_indices,
            recv_cumsum,
            recv_indices,
            backward_recv_indices,
            gathered_stats,
        ) = moe_prepare(
            topk_ids,
            topk_weights,
            eplb_local_stats,
            self._prepare_workspace,
            tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            num_experts,
            num_experts,
            self.params.top_k,
        )
        # The prepare kernel fills rows past each source rank's token count
        # with the slot count (== num_experts) as the invalid id.
        if self.params.invalid_expert_id != num_experts:
            recv_topk_ids.masked_fill_(
                recv_topk_ids == num_experts, self.params.invalid_expert_id
            )

        # Received rows are laid out rank-major, tokens_per_rank per source.
        recv_rows = self.ep_size * tokens_per_rank
        recv_hidden = self._alltoallv(
            hidden_states,
            hidden_states.new_empty(recv_rows, hidden_states.shape[1]),
            send_cumsum,
            send_indices,
            recv_cumsum,
            recv_indices,
        )
        recv_scale = None
        if hidden_states_scale is not None:
            recv_scale = self._alltoallv(
                hidden_states_scale,
                hidden_states_scale.new_empty(recv_rows, hidden_states_scale.shape[1]),
                send_cumsum,
                send_indices,
                recv_cumsum,
                recv_indices,
            )

        self._state = {
            "send_cumsum": send_cumsum,
            "recv_cumsum": recv_cumsum,
            "recv_indices": recv_indices,
            "backward_recv_indices": backward_recv_indices,
            "local_num_tokens": hidden_states.shape[0],
        }
        return MoEEpDispatchResult(
            hidden_states=recv_hidden,
            hidden_states_scale=recv_scale,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            tokens_per_rank=tokens_per_rank,
            eplb_gathered_stats=gathered_stats,
        )

    def combine(
        self,
        expert_output: "torch.Tensor",
        *,
        output: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        import torch

        state = self._state
        if state is None:
            raise RuntimeError("combine called before dispatch")
        if expert_output.dim() == 3:
            expert_output = expert_output.flatten(0, 1)
        top_k = self.params.top_k
        num_tokens = state["local_num_tokens"]
        # Send each received row back to the top-k slot it came from, then
        # reduce over the slots; slots without a row stay zero.
        per_slot = self._alltoallv(
            expert_output,
            expert_output.new_zeros(num_tokens * top_k, expert_output.shape[1]),
            state["recv_cumsum"],
            state["recv_indices"],
            state["send_cumsum"],
            state["backward_recv_indices"],
        )
        self._state = None
        combined = torch.sum(per_slot.view(num_tokens, top_k, -1), dim=1)
        if output is not None:
            output.copy_(combined)
            return output
        return combined

    def destroy(self) -> None:
        # The workspaces stay with the process for later instances.
        self._workspace = None
        self._prepare_workspace = None
        self._state = None
