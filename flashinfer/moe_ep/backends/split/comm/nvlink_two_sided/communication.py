"""NVLink two-sided MoE communication over MNNVL FIFO channels.

Sender and receiver kernels exchange packets through per-peer FIFOs in
symmetric memory, like a collective. A prepare step first exchanges the
routing (expert ids, weights, EPLB statistics) and per-rank token counts; the
token payloads then follow as all-to-all-v transfers. The symmetric footprint
depends on the number of channels, not on the token count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

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

    from .....config import BootstrapConfig


@register_communication("nvlink_two_sided")
class NVLinkTwoSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink two-sided all-to-all-v kernels.

    Requires ``num_experts`` divisible by 4. EPLB statistics, when passed to
    :meth:`dispatch`, must cover all ``num_experts`` experts.
    """

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NVLinkTwoSidedConfig] = None,
    ) -> None:
        super().__init__(bootstrap, params)
        from ......comm.mnnvl import MnnvlMemory
        from ......comm.trtllm_alltoall import MnnvlMoe

        if params.num_experts % 4 != 0:
            raise ValueError(
                "NVLinkTwoSidedAlltoAll requires num_experts divisible by 4, "
                f"got {params.num_experts}"
            )
        self.config = NVLinkTwoSidedConfig() if config is None else config
        mapping, mnnvl_config = mnnvl_mapping_and_config(
            bootstrap, self.config.comm_backend
        )
        MnnvlMemory.initialize()
        self._workspace = MnnvlMoe.get_moe_workspaces(mapping, mnnvl_config)
        self._prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
            mapping, mnnvl_config
        )
        self._state: Optional[dict[str, Any]] = None

    @classmethod
    def is_platform_supported(cls) -> bool:
        return nvlink_platform_supported()

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

        from ......comm.trtllm_alltoall import MnnvlMoe

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
        alltoall_info, recv_topk_ids, recv_topk_weights, gathered_stats = (
            MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
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
        )
        # The prepare kernel fills rows past each source rank's token count
        # with the slot count (== num_experts) as the invalid id.
        if self.params.invalid_expert_id != num_experts:
            recv_topk_ids.masked_fill_(
                recv_topk_ids == num_experts, self.params.invalid_expert_id
            )

        recv_hidden = MnnvlMoe.mnnvl_moe_alltoallv(
            hidden_states, alltoall_info, self._workspace, self.ep_rank, self.ep_size
        )
        recv_scale = None
        if hidden_states_scale is not None:
            recv_scale = MnnvlMoe.mnnvl_moe_alltoallv(
                hidden_states_scale,
                alltoall_info,
                self._workspace,
                self.ep_rank,
                self.ep_size,
            )

        self._state = {
            "alltoall_info": alltoall_info,
            "local_num_tokens": hidden_states.shape[0],
            "tokens_per_rank": tokens_per_rank,
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
        from ......comm.trtllm_alltoall import MnnvlMoe

        state = self._state
        if state is None:
            raise RuntimeError("combine called before dispatch")
        if expert_output.dim() == 3:
            expert_output = expert_output.flatten(0, 1)
        combined = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            expert_output,
            state["alltoall_info"],
            self._workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=self.params.top_k,
            token_count=state["local_num_tokens"],
        )
        self._state = None
        if output is not None:
            output.copy_(combined)
            return output
        return combined

    def destroy(self) -> None:
        # MnnvlMoe caches its workspaces process-wide.
        self._workspace = None
        self._prepare_workspace = None
        self._state = None
