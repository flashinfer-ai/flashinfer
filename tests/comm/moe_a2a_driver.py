# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Test driver for the functional MoE all-to-all ops of flashinfer.comm.

``MoeA2ADriver`` allocates and initializes an MNNVL workspace, shared by
drivers with the same geometry and backend, and runs ``moe_a2a_dispatch``,
``moe_a2a_sanitize_expert_ids`` and ``moe_a2a_combine`` on it, carrying the
per-round state that combine needs from dispatch.
"""

from typing import Optional

import torch

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory
from flashinfer.comm.trtllm_moe_alltoall import (
    MoeAlltoAllBackend,
    get_moe_alltoall_module,
    moe_a2a_combine,
    moe_a2a_dispatch,
    moe_a2a_get_workspace_size_per_rank,
    moe_a2a_initialize,
    moe_a2a_sanitize_expert_ids,
    moe_a2a_wrap_payload_tensor_in_workspace,
)
from flashinfer.tllm_enums import SfLayout


def workspace_bytes_per_rank(
    ep_size: int,
    top_k: int,
    max_num_tokens: int,
    hidden_size: int,
    extra_payload_bytes_per_token: int = 0,
    eplb_stats_num_experts: int = 0,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
) -> int:
    """Workspace bytes per rank for 16-bit hidden states, int32 expert ids and
    FP32 weights, plus ``extra_payload_bytes_per_token``."""
    return moe_a2a_get_workspace_size_per_rank(
        ep_size,
        max_num_tokens,
        hidden_size * 2 + top_k * 4 + top_k * 4 + extra_payload_bytes_per_token,
        hidden_size * 2,
        eplb_stats_num_experts,
        backend=backend,
    )


def metainfo_index(backend: MoeAlltoAllBackend = "trtllm") -> dict:
    """Metainfo field indices, keyed by name without the ``MOE_A2A_`` prefix."""
    names, values = get_moe_alltoall_module(backend).moe_a2a_get_metainfo_index_pairs()
    return {
        name.removeprefix("MOE_A2A_"): int(value)
        for name, value in zip(names, values, strict=True)
    }


class MoeA2ADriver:
    # Workspaces keyed by (bytes per rank, ep_rank, ep_size, max tokens,
    # EPLB experts, backend).
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
        *,
        backend: MoeAlltoAllBackend = "trtllm",
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
        self.backend = backend
        key = (
            workspace_size_per_rank,
            self.ep_rank,
            self.ep_size,
            max_num_tokens,
            eplb_stats_num_experts,
            backend,
        )
        if key not in self._WORKSPACES:
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                max_num_tokens,
                eplb_stats_num_experts,
                backend=backend,
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
        recv, combine_offset, self.eplb_gathered_stats = moe_a2a_dispatch(
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
            backend=self.backend,
            recv_view_cache=self._recv_view_cache,
        )
        if invalid_token_expert_id is not None:
            moe_a2a_sanitize_expert_ids(
                recv[expert_id_payload_index],
                self.workspace,
                self.metainfo,
                self.ep_rank,
                invalid_token_expert_id,
                backend=self.backend,
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
        return moe_a2a_wrap_payload_tensor_in_workspace(
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
        combined = moe_a2a_combine(
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
            backend=self.backend,
        )
        self._round = None
        self.eplb_gathered_stats = None
        return combined
