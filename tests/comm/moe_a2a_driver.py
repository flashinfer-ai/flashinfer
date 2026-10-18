# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Test driver for the kernel-level operations of the Cake MoE all-to-all.

``MoeA2ADriver`` allocates and initializes an MNNVL workspace, shared by
drivers with the same geometry, and runs the ``dispatch``,
``sanitize_expert_ids`` and ``combine`` operations of
:mod:`flashinfer.moe_ep.backends.split.comm.cake.kernels` on it, carrying the
per-round state that combine needs from dispatch.
"""

from typing import Optional

import torch

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory
from flashinfer.moe_ep.backends.split.comm.cake import kernels as cake_kernels
from flashinfer.moe_ep.backends.split.comm.cake.kernels import metainfo_index
from flashinfer.tllm_enums import SfLayout

__all__ = ["MoeA2ADriver", "metainfo_index", "workspace_bytes_per_rank"]


def workspace_bytes_per_rank(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    extra_payload_bytes_per_token: int = 0,
    eplb_stats_num_experts: int = 0,
) -> int:
    """Workspace bytes per rank for 16-bit hidden states, int32 expert ids and
    FP32 weights, plus ``extra_payload_bytes_per_token``."""
    return cake_kernels.workspace_size_per_rank(
        ep_size,
        max_num_tokens,
        hidden_size * 2 + top_k * 4 + top_k * 4 + extra_payload_bytes_per_token,
        hidden_size * 2,
        eplb_stats_num_experts,
    )


class MoeA2ADriver:
    # Workspaces keyed by (bytes per rank, ep_rank, ep_size, max tokens,
    # EPLB experts).
    _WORKSPACES: dict = {}

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        workspace_size_per_rank: int,
        mnnvl_config: Optional[MnnvlConfig] = None,
        eplb_stats_num_experts: int = 0,
        enable_rank_mask: bool = False,
    ):
        MnnvlMemory.initialize()
        if mnnvl_config:
            MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.enable_rank_mask = enable_rank_mask
        key = (
            workspace_size_per_rank,
            self.ep_rank,
            self.ep_size,
            max_num_tokens,
            eplb_stats_num_experts,
        )
        if key not in self._WORKSPACES:
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = cake_kernels.initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                max_num_tokens,
                eplb_stats_num_experts,
            )
            # No peer may publish into this workspace until every rank has
            # cleared its own slice.
            MnnvlMemory.allocated_map[mnnvl_mem.ptr].comm.barrier()
            self._WORKSPACES[key] = (mnnvl_mem, workspace, metainfo)
        self.mnnvl_mem, self.workspace, self.metainfo = self._WORKSPACES[key]
        self._recv_view_cache: dict = {}
        self._round: Optional[dict] = None
        self.eplb_gathered_stats: Optional[torch.Tensor] = None

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        input_payloads: list,
        runtime_max_tokens_per_rank: int,
        invalid_token_expert_id: Optional[int] = None,
        expert_id_payload_index: Optional[int] = None,
        eplb_local_stats: Optional[torch.Tensor] = None,
        active_rank_mask: Optional[torch.Tensor] = None,
    ) -> list:
        assert self._round is None, "dispatch called twice without combine"
        recv, combine_offset, self.eplb_gathered_stats = cake_kernels.dispatch(
            token_selected_experts,
            input_payloads,
            self.workspace,
            self.metainfo,
            runtime_max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_experts,
            eplb_local_stats=eplb_local_stats,
            enable_rank_mask=self.enable_rank_mask,
            active_rank_mask=active_rank_mask,
            recv_view_cache=self._recv_view_cache,
        )
        if invalid_token_expert_id is not None:
            cake_kernels.sanitize_expert_ids(
                recv[expert_id_payload_index],
                self.workspace,
                self.metainfo,
                self.ep_rank,
                invalid_token_expert_id,
            )
        self._round = {
            "local_num_tokens": token_selected_experts.size(0),
            "combine_offset": combine_offset,
        }
        return recv

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self._round is not None, "dispatch must precede the combine payload"
        start = self._round["combine_offset"]
        end = (
            start
            + self.ep_size * runtime_max_tokens_per_rank * hidden_size * dtype.itemsize
        )
        return cake_kernels.wrap_workspace(
            self.workspace[self.ep_rank, :],
            [self.ep_size, runtime_max_tokens_per_rank],
            start,
            end,
            dtype,
        )

    def combine(
        self,
        payload: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        output_scales: Optional[torch.Tensor] = None,
        output_scalar_scale: float = 1.0,
        sf_layout: SfLayout = SfLayout.layout_linear,
        output: Optional[torch.Tensor] = None,
        *,
        use_low_precision: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self._round is not None, "combine called before dispatch"
        combined = cake_kernels.combine(
            payload,
            self._round["local_num_tokens"],
            self.workspace,
            self.metainfo,
            runtime_max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self._round["combine_offset"],
            payload_in_workspace,
            output_dtype,
            output_scales,
            output_scalar_scale,
            sf_layout,
            output,
            use_low_precision=use_low_precision,
            enable_rank_mask=self.enable_rank_mask,
            active_rank_mask=active_rank_mask,
        )
        self._round = None
        self.eplb_gathered_stats = None
        return combined
