"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from ..jit import gen_comm_alltoall_module
from ..utils import register_custom_op
from .mapping import Mapping
from .mnnvl import MnnvlMemory, MnnvlConfig


@functools.cache
def get_comm_alltoall_module():
    module = gen_comm_alltoall_module().build_and_load()

    @register_custom_op(
        "flashinfer::moe_comm_prepare_indices",
        mutates_args=[],
    )
    def moe_comm_prepare_indices(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cum_sum: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = gathered_target_rank_ids.device
        max_send_ranks_per_token = max(top_k, ep_size)
        local_gather_indices = torch.empty(
            (max_token_count_per_rank * ep_size), device=device, dtype=torch.int
        )
        send_rank_count_cum_sum = torch.empty(
            (ep_size,), device=device, dtype=torch.int
        )
        send_rank_local_indices = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token),
            device=device,
            dtype=torch.int,
        )
        recv_rank_count_cum_sum = torch.empty((ep_size), device=device, dtype=torch.int)
        recv_rank_local_indices = torch.empty(
            (max_token_count_per_rank * ep_size), device=device, dtype=torch.int
        )
        backward_recv_rank_local_indice = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token),
            device=device,
            dtype=torch.int,
        )
        module.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cum_sum,
            local_gather_indices,
            send_rank_count_cum_sum,
            send_rank_local_indices,
            recv_rank_count_cum_sum,
            recv_rank_local_indices,
            backward_recv_rank_local_indice,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )
        return (
            local_gather_indices,
            send_rank_count_cum_sum,
            send_rank_local_indices,
            recv_rank_count_cum_sum,
            recv_rank_local_indices,
            backward_recv_rank_local_indice,
        )

    @register_custom_op(
        "flashinfer::moe_local_gather",
        mutates_args=["local_expert_ids", "local_scales"],
    )
    def moe_local_gather(
        recv_rank_cum_sum: torch.Tensor,
        local_gather_indices: torch.Tensor,
        gathered_expert_ids: torch.Tensor,
        gathered_scales: torch.Tensor,
        local_expert_ids: torch.Tensor,
        local_scales: torch.Tensor,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        module.moe_local_gather(
            recv_rank_cum_sum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

    @register_custom_op(
        "flashinfer::moe_comm",
        mutates_args=["output"],
    )
    def moe_comm(
        input: torch.Tensor,
        send_rank_cum_sum: torch.Tensor,
        send_indices: torch.Tensor,
        output: torch.Tensor,
        recv_rank_cum_sum: torch.Tensor,
        recv_indices: torch.Tensor,
        all_workspaces: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        module.moe_comm(
            input,
            send_rank_cum_sum,
            send_indices,
            output,
            recv_rank_cum_sum,
            recv_indices,
            all_workspaces,
            ep_rank,
            ep_size,
        )

    @register_custom_op(
        "flashinfer::set_moe_max_usable_sm_count",
        mutates_args=[],
    )
    def set_moe_max_usable_sm_count(
        max_sm_count: int,
    ) -> None:
        module.set_moe_max_usable_sm_count(max_sm_count)

    @register_custom_op(
        "flashinfer::get_moe_commworkspace_size_per_rank",
        mutates_args=[],
    )
    def get_moe_commworkspace_size_per_rank(
        ep_size: int,
    ) -> int:
        return module.get_moe_commworkspace_size_per_rank(ep_size)

    @register_custom_op(
        "flashinfer::get_moe_prepare_workspace_size_per_rank",
        mutates_args=[],
    )
    def get_moe_prepare_workspace_size_per_rank(
        ep_size: int,
    ) -> int:
        return module.get_moe_prepare_workspace_size_per_rank(ep_size)

    @register_custom_op(
        "flashinfer::moe_prepare",
        mutates_args=[],
    )
    def moe_prepare(
        experts_ids: torch.Tensor,
        scales: Optional[torch.Tensor],
        experts_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        attrs = {"dtype": torch.int32, "device": experts_ids.device}

        prepared_local_expert_ids = torch.empty(
            (max_token_count_per_rank * ep_size, top_k), **attrs
        )
        send_rank_count_cum_sum = torch.empty((ep_size,), **attrs)
        recv_rank_count_cum_sum = torch.empty((ep_size,), **attrs)

        gather_recv_rank_indices = torch.empty(
            (max_token_count_per_rank * ep_size,), **attrs
        )
        recv_rank_indices = torch.empty((max_token_count_per_rank * ep_size,), **attrs)

        max_send_ranks_per_token = max(ep_size, top_k)

        gather_backward_recv_rank_indices = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token,), **attrs
        )
        backward_recv_rank_indices = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token,), **attrs
        )
        gather_send_rank_indices = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token,), **attrs
        )
        send_rank_indices = torch.empty(
            (max_token_count_per_rank * max_send_ranks_per_token,), **attrs
        )
        if scales is not None:
            prepared_local_scales = torch.empty(
                (max_token_count_per_rank * ep_size, top_k),
                dtype=torch.float32,
                device=attrs["device"],
            )
        else:
            prepared_local_scales = None
        if experts_statics is not None:
            gathered_expert_statics = torch.empty((ep_size, expert_count), **attrs)
        else:
            gathered_expert_statics = None

        module.moe_prepare(
            experts_ids,
            scales,
            experts_statics,
            workspace,
            prepared_local_expert_ids,
            send_rank_count_cum_sum,
            recv_rank_count_cum_sum,
            gather_recv_rank_indices,
            recv_rank_indices,
            gather_backward_recv_rank_indices,
            backward_recv_rank_indices,
            gather_send_rank_indices,
            send_rank_indices,
            prepared_local_scales,
            gathered_expert_statics,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )
        return (
            prepared_local_expert_ids,
            prepared_local_scales,
            send_rank_count_cum_sum,
            gather_send_rank_indices,
            recv_rank_count_cum_sum,
            gather_recv_rank_indices,
            gather_backward_recv_rank_indices,
            gathered_expert_statics,
        )

    return SimpleNamespace(
        moe_comm_prepare_indices=moe_comm_prepare_indices,
        moe_local_gather=moe_local_gather,
        moe_comm=moe_comm,
        set_moe_max_usable_sm_count=set_moe_max_usable_sm_count,
        get_moe_commworkspace_size_per_rank=get_moe_commworkspace_size_per_rank,
        get_moe_prepare_workspace_size_per_rank=get_moe_prepare_workspace_size_per_rank,
        moe_prepare=moe_prepare,
    )


def moe_comm_prepare_indices(
    gathered_target_rank_ids: torch.Tensor,
    real_rank_token_count_cum_sum: Optional[torch.Tensor],
    max_token_count_per_rank: int,
    expert_count: int,
    top_k: int,
    ep_rank: int,
    ep_size: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return get_comm_alltoall_module().moe_comm_prepare_indices(
        gathered_target_rank_ids,
        real_rank_token_count_cum_sum,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )


def moe_local_gather(
    recv_rank_cum_sum: torch.Tensor,
    local_gather_indices: torch.Tensor,
    gathered_expert_ids: torch.Tensor,
    gathered_scales: torch.Tensor,
    local_expert_ids: torch.Tensor,
    local_scales: torch.Tensor,
    max_token_count_per_rank: int,
    expert_count: int,
    top_k: int,
    ep_rank: int,
    ep_size: int,
) -> None:
    get_comm_alltoall_module().moe_local_gather(
        recv_rank_cum_sum,
        local_gather_indices,
        gathered_expert_ids,
        gathered_scales,
        local_expert_ids,
        local_scales,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )


def moe_comm(
    input: torch.Tensor,
    send_rank_cum_sum: torch.Tensor,
    send_indices: torch.Tensor,
    output: torch.Tensor,
    recv_rank_cum_sum: torch.Tensor,
    recv_indices: torch.Tensor,
    all_workspaces: torch.Tensor,
    ep_rank: int,
    ep_size: int,
) -> None:
    get_comm_alltoall_module().moe_comm(
        input,
        send_rank_cum_sum,
        send_indices,
        output,
        recv_rank_cum_sum,
        recv_indices,
        all_workspaces,
        ep_rank,
        ep_size,
    )


def set_moe_max_usable_sm_count(
    max_sm_count: int,
) -> None:
    get_comm_alltoall_module().set_moe_max_usable_sm_count(max_sm_count)


def get_moe_commworkspace_size_per_rank(
    ep_size: int,
) -> int:
    return get_comm_alltoall_module().get_moe_commworkspace_size_per_rank(ep_size)


def get_moe_prepare_workspace_size_per_rank(
    ep_size: int,
) -> int:
    return get_comm_alltoall_module().get_moe_prepare_workspace_size_per_rank(ep_size)


def moe_prepare(
    experts_ids: torch.Tensor,
    scales: Optional[torch.Tensor],
    experts_statics: Optional[torch.Tensor],
    workspace: torch.Tensor,
    max_token_count_per_rank: int,
    ep_rank: int,
    ep_size: int,
    expert_count: int,
    slot_count: int,
    top_k: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return get_comm_alltoall_module().moe_prepare(
        experts_ids,
        scales,
        experts_statics,
        workspace,
        max_token_count_per_rank,
        ep_rank,
        ep_size,
        expert_count,
        slot_count,
        top_k,
    )


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping, config: Optional[MnnvlConfig] = None):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = get_moe_commworkspace_size_per_rank(mapping.tp_size)
        if config:
            MnnvlMemory.set_comm_from_config(mapping, config)  # type: ignore[attr-defined]
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(
            torch.uint64
        )
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(
        mapping: Mapping, config: Optional[MnnvlConfig] = None
    ):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = get_moe_prepare_workspace_size_per_rank(
            mapping.tp_size
        )
        if config:
            MnnvlMemory.set_comm_from_config(mapping, config)  # type: ignore[attr-defined]
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, (
            "expert_count should be divisible by ep_size"
        )
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        scales: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            prepared_local_experts,
            prepared_local_scales,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = moe_prepare(
            expert_ids,
            scales,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return (
            alltoall_info,
            prepared_local_experts,
            prepared_local_scales,
            gathered_expert_statics,
        )

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: torch.Tensor,
        gathered_expert_ids: torch.Tensor,
        gathered_scales: torch.Tensor,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count,
            top_k,
            dtype=torch.int32,
            device=torch.device("cuda"),
        )
        local_scales = torch.empty(
            local_token_allocation_count,
            top_k,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ):
        assert x.dim() == 2, "only 2D tensor supported, please reshape."
        output_tensor = torch.empty(
            alltoall_info.local_token_allocation_count,
            x.shape[1],
            dtype=x.dtype,
            device=torch.device("cuda"),
        )
        moe_comm(
            x,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.send_rank_local_indices,
            output_tensor,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return output_tensor

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensor = torch.zeros(
            token_count * top_k, x.shape[1], dtype=x.dtype, device=torch.device("cuda")
        )
        moe_comm(
            x,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            output_tensor,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return torch.sum(
            output_tensor.reshape(token_count, top_k, x.shape[1]), dim=1, keepdim=False
        )
