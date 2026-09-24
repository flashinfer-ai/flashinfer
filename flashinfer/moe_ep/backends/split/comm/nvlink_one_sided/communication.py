"""NVLink one-sided MoE communication over MNNVL symmetric memory.

The dispatch kernel writes each token directly into the receive buffers of the
ranks that own its experts; the combine kernel reads the expert outputs back
from those ranks and reduces them locally. Both sides use the throughput
all-to-all primitive :class:`flashinfer.comm.MoeAlltoAll`, whose symmetric
workspace grows with ``ep_size * max_tokens_per_rank``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .....core.comm.communication import (
    MoEEpCommParams,
    MoEEpCommunication,
    MoEEpDispatchResult,
    register_communication,
)
from ..nvlink_common import mnnvl_mapping_and_config, nvlink_platform_supported
from .config import NVLinkOneSidedConfig

if TYPE_CHECKING:
    import torch

    from ......comm.trtllm_moe_alltoall import MoeAlltoAll
    from .....config import BootstrapConfig


@register_communication("nvlink_one_sided")
class NVLinkOneSidedAlltoAll(MoEEpCommunication):
    """Rank-major dispatch/combine with NVLink one-sided put/get kernels.

    ``hidden_states``, its optional scale factors, ``topk_ids`` and
    ``topk_weights`` travel as payloads of a single dispatch. Received rows
    beyond each source rank's token count get ``invalid_expert_id`` routing.
    """

    def __init__(
        self,
        bootstrap: "BootstrapConfig",
        params: MoEEpCommParams,
        config: Optional[NVLinkOneSidedConfig] = None,
    ) -> None:
        super().__init__(bootstrap, params)
        from ......comm.trtllm_moe_alltoall import (
            MoeAlltoAll,
            moe_a2a_get_workspace_size_per_rank,
        )

        self.config = NVLinkOneSidedConfig() if config is None else config
        if self.config.extra_payload_bytes_per_token < 0:
            raise ValueError("extra_payload_bytes_per_token must be non-negative")

        itemsize = params.token_dtype.itemsize
        # Dispatch carries the token row, int32 expert ids and FP32 weights.
        dispatch_bytes_per_token = (
            params.hidden_size * itemsize
            + params.top_k * 4
            + params.top_k * 4
            + self.config.extra_payload_bytes_per_token
        )
        # Expert outputs come back unquantized, at least 16 bits wide.
        combine_bytes_per_token = params.hidden_size * max(itemsize, 2)
        workspace_size_per_rank = moe_a2a_get_workspace_size_per_rank(
            self.ep_size,
            params.max_tokens_per_rank,
            dispatch_bytes_per_token,
            combine_bytes_per_token,
            self.config.eplb_stats_num_experts,
            backend=self.config.kernel,
        )

        mapping, mnnvl_config = mnnvl_mapping_and_config(
            bootstrap, self.config.comm_backend
        )
        self._alltoall: Optional[MoeAlltoAll] = MoeAlltoAll(
            mapping,
            max_num_tokens=params.max_tokens_per_rank,
            top_k=params.top_k,
            num_experts=params.num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
            mnnvl_config=mnnvl_config,
            eplb_stats_num_experts=self.config.eplb_stats_num_experts,
            enable_rank_mask=self.config.enable_rank_mask,
            backend=self.config.kernel,
        )
        self._tokens_per_rank: Optional[int] = None
        self._combine_buffer: "torch.Tensor | None" = None

    @classmethod
    def is_platform_supported(cls) -> bool:
        return nvlink_platform_supported()

    @property
    def alltoall(self) -> "MoeAlltoAll":
        """The underlying all-to-all primitive (checkpointing, metainfo access)."""
        if self._alltoall is None:
            raise RuntimeError("NVLinkOneSidedAlltoAll has been destroyed")
        return self._alltoall

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

        alltoall = self.alltoall
        if self._tokens_per_rank is not None:
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

        payloads = [hidden_states]
        if hidden_states_scale is not None:
            payloads.append(hidden_states_scale)
        expert_id_index = len(payloads)
        payloads.append(topk_ids)
        if topk_weights is not None:
            payloads.append(topk_weights)

        recv = alltoall.dispatch(
            topk_ids,
            payloads,
            tokens_per_rank,
            invalid_token_expert_id=self.params.invalid_expert_id,
            expert_id_payload_index=expert_id_index,
            eplb_local_stats=eplb_local_stats,
            active_rank_mask=active_rank_mask,
        )
        self._tokens_per_rank = tokens_per_rank
        recv = [t.flatten(0, 1) for t in recv]
        return MoEEpDispatchResult(
            hidden_states=recv[0],
            hidden_states_scale=recv[1] if hidden_states_scale is not None else None,
            topk_ids=recv[expert_id_index],
            topk_weights=recv[expert_id_index + 1]
            if topk_weights is not None
            else None,
            tokens_per_rank=tokens_per_rank,
            eplb_gathered_stats=alltoall.eplb_gathered_stats,
        )

    def get_combine_input_buffer(self, dtype: "torch.dtype") -> "torch.Tensor":
        """``[ep_size * tokens_per_rank, hidden_size]`` view of the combine
        payload region of this rank's workspace."""
        if self._tokens_per_rank is None:
            raise RuntimeError("get_combine_input_buffer called before dispatch")
        buffer = self.alltoall.get_combine_payload_tensor_in_workspace(
            self._tokens_per_rank, self.params.hidden_size, dtype
        ).flatten(0, 1)
        self._combine_buffer = buffer
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
        alltoall = self.alltoall
        tokens_per_rank = self._tokens_per_rank
        if tokens_per_rank is None:
            raise RuntimeError("combine called before dispatch")
        if expert_output.dim() == 2:
            payload = expert_output.view(self.ep_size, tokens_per_rank, -1)
        elif expert_output.dim() == 3:
            payload = expert_output
        else:
            raise ValueError(
                f"expert_output must be 2D or 3D, got shape {tuple(expert_output.shape)}"
            )
        in_workspace = (
            self._combine_buffer is not None
            and expert_output.data_ptr() == self._combine_buffer.data_ptr()
        )
        combined = alltoall.combine(
            payload,
            tokens_per_rank,
            payload_in_workspace=in_workspace,
            output=output,
            use_low_precision=self.config.use_low_precision_combine,
            active_rank_mask=active_rank_mask,
        )
        self._tokens_per_rank = None
        self._combine_buffer = None
        return combined

    def destroy(self) -> None:
        # The symmetric workspace is cached process-wide by MoeAlltoAll and
        # shared with other instances of the same geometry.
        self._alltoall = None
        self._tokens_per_rank = None
        self._combine_buffer = None
