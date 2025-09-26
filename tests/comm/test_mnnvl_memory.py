# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import socket

import pynvml
import pytest
import torch

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory, MpiComm
from flashinfer.comm.trtllm_alltoall import MnnvlMoe, MoEAlltoallInfo

pynvml.nvmlInit()


@pytest.mark.skipif(
    not MnnvlMemory.supports_mnnvl(),
    reason="Mnnvl memory is not supported on this platform",
)
class TestMnnvlMemory:
    @pytest.fixture(autouse=True)
    def setup(self):
        # get num of task per node
        hostname = socket.gethostname()
        self.comm = MpiComm()
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        all_hostnames = self.comm.allgather(hostname)
        local_ntasks_per_node = all_hostnames.count(hostname)
        all_ntasks_per_node = self.comm.allgather(local_ntasks_per_node)
        uniform_ntasks = all(x == all_ntasks_per_node[0] for x in all_ntasks_per_node)
        assert uniform_ntasks, "Not all nodes has same ntasks_per_node"
        self.local_world_size = local_ntasks_per_node
        self.local_rank = self.rank % self.local_world_size
        local_dev_count = torch.cuda.device_count()
        assert self.local_world_size <= local_dev_count, (
            "ntasks_per_node should be less than local device count"
        )
        torch.cuda.set_device(self.local_rank)
        MnnvlMemory.initialize()
        self.mapping = Mapping(
            self.world_size, self.rank, self.local_world_size, tp_size=self.world_size
        )

    @staticmethod
    def align_memory(size: int):
        align_size = 2 * 1024 * 1024
        return (size + align_size - 1) // align_size * align_size

    @pytest.mark.skipif(
        not MnnvlMemory.supports_mnnvl(),
        reason="Mnnvl memory is not supported on this platform",
    )
    def test_mnnvl_memory(self):
        # allocate un-aligned memory
        allocate0_size = 4 * 1024 * 1024 - 3 * 1024
        mnnvl_memory0 = MnnvlMemory(self.mapping, allocate0_size)
        allocate0_size_aligned = TestMnnvlMemory.align_memory(allocate0_size)
        assert MnnvlMemory.current_mem_offset == allocate0_size_aligned

        tensor0 = mnnvl_memory0.as_torch_strided_tensor(torch.int32)
        numel_per_rank = allocate0_size // 4
        tensor0[(self.rank + 1) % self.world_size] = torch.arange(
            start=self.rank, end=self.rank + numel_per_rank, device="cuda"
        )
        self.comm.Barrier()
        for r in range(self.world_size):
            torch.equal(
                tensor0[(r + 1) % self.world_size],
                torch.arange(start=r, end=r + numel_per_rank, device="cuda"),
            )

        allocate1_size = 30 * 1024 * 1024 - 2 * 1024
        mnnvl_memory1 = MnnvlMemory(self.mapping, allocate1_size)
        allocate1_size_aligned = TestMnnvlMemory.align_memory(allocate1_size)
        assert (
            MnnvlMemory.current_mem_offset
            == allocate0_size_aligned + allocate1_size_aligned
        )
        tensor1 = mnnvl_memory1.as_torch_strided_tensor(torch.float32)
        numel_per_rank = allocate1_size // 4
        tensor1[(self.rank + 5) % self.world_size] = torch.arange(
            start=self.rank,
            end=self.rank + numel_per_rank,
            dtype=torch.float32,
            device="cuda",
        )
        self.comm.Barrier()
        for r in range(self.world_size):
            torch.equal(
                tensor1[(r + 5) % self.world_size],
                torch.arange(
                    start=r, end=r + numel_per_rank, dtype=torch.float32, device="cuda"
                ),
            )
        self.comm.Barrier()
        del tensor0, mnnvl_memory0
        self.comm.Barrier()

        large_allocation2_size = 768 * 1024 * 1024
        large_mnnvl_memory2 = MnnvlMemory(self.mapping, large_allocation2_size)
        allocate2_size_aligned = TestMnnvlMemory.align_memory(large_allocation2_size)
        assert MnnvlMemory.current_mem_offset == allocate2_size_aligned
        assert large_mnnvl_memory2.rank_stride == (1 << 30)

        del tensor1

    @pytest.mark.skipif(
        not MnnvlMemory.supports_mnnvl(),
        reason="Mnnvl memory is not supported on this platform",
    )
    def test_moe_alltoall_multi_rank_single_gpu(self):
        torch.cuda.set_device(self.rank)
        max_world_size = 8
        assert self.world_size <= max_world_size, (
            f"should run with world_size at most {max_world_size}"
        )
        torch.manual_seed(self.world_size)
        input_entry_per_rank, vector_dim, dtype = 128, 256, torch.float16

        # Create a random input tensor
        input_tensor = torch.randn(
            input_entry_per_rank * self.world_size,
            vector_dim,
            dtype=dtype,
            device=torch.device("cuda"),
        )
        ref_output_tensor = torch.zeros(
            input_entry_per_rank * self.world_size,
            vector_dim,
            dtype=dtype,
            device=torch.device("cuda"),
        )
        target_rank_ids = torch.randint(
            0,
            self.world_size,
            (input_entry_per_rank * self.world_size,),
            dtype=torch.int32,
            device=torch.device("cuda"),
        )

        input_tensors_all_ranks = list(torch.split(input_tensor, input_entry_per_rank))
        target_rank_ids_all_ranks = list(
            torch.split(target_rank_ids, input_entry_per_rank)
        )

        send_ids_all_ranks = []
        send_counts_all_ranks = []
        send_cumsum_all_ranks = []
        send_start_end_all_ranks = []

        # each rank do its own local compute to get how to send data to other ranks.
        for rank in range(self.world_size):
            send_start_end = []
            local_target_rank_ids = target_rank_ids_all_ranks[rank]
            sorted_local_target_rank_ids, local_send_id = torch.sort(
                local_target_rank_ids
            )
            local_send_id = local_send_id.to(torch.int32)
            padded_sorted_local_target_rank_ids = torch.cat(
                (
                    sorted_local_target_rank_ids,
                    torch.arange(
                        self.world_size, dtype=torch.int32, device=torch.device("cuda")
                    ),
                )
            )
            unique_target_rank_ids, local_send_counts = torch.unique(
                padded_sorted_local_target_rank_ids, return_counts=True
            )
            local_send_counts = local_send_counts.to(torch.int32)
            assert unique_target_rank_ids.numel() == self.world_size, (
                "unique_target_rank_ids must be equal to world_size"
            )
            local_send_counts -= 1  # remove padding
            local_send_cumsum = torch.cumsum(local_send_counts, dim=0).to(torch.int32)
            send_ids_all_ranks.append(local_send_id)
            send_counts_all_ranks.append(local_send_counts)
            send_cumsum_all_ranks.append(local_send_cumsum)
            local_send_cumsum_cpu = local_send_cumsum.cpu().tolist()
            for i in range(len(local_send_cumsum_cpu)):
                send_start_end.append(
                    (
                        local_send_cumsum_cpu[i - 1] if i > 0 else 0,
                        local_send_cumsum_cpu[i],
                    )
                )
            send_start_end_all_ranks.append(send_start_end)

        recv_ids_all_ranks = []
        recv_cumsum_all_ranks = []

        ref_output_tensors_all_ranks = []

        total_recv_all_ranks_cpu = []
        output_indice_offset = 0

        output_start_current_rank = 0
        # each rank do compute based on other ranks' send counts to get how to receive data from other ranks.
        for rank in range(self.world_size):
            local_recv_counts = torch.zeros(
                self.world_size, dtype=torch.int32, device=torch.device("cuda")
            )
            for other_rank in range(self.world_size):
                local_recv_counts[other_rank] = send_counts_all_ranks[other_rank][rank]
                local_recv_count_pair = local_recv_counts[other_rank].cpu().item()
                send_rank_start_end = send_start_end_all_ranks[other_rank][rank]
                ref_output_tensor[
                    output_indice_offset : output_indice_offset + local_recv_count_pair
                ] = input_tensors_all_ranks[other_rank][
                    send_ids_all_ranks[other_rank][
                        send_rank_start_end[0] : send_rank_start_end[1]
                    ]
                ]
                output_indice_offset += local_recv_count_pair
            local_recv_cumsum = torch.cumsum(local_recv_counts, dim=0).to(torch.int32)
            recv_cumsum_all_ranks.append(local_recv_cumsum)
            total_recv_count = local_recv_cumsum[-1].cpu()
            total_recv_all_ranks_cpu.append(total_recv_count)
            ref_output_tensors_all_ranks.append(
                ref_output_tensor[
                    output_start_current_rank : output_start_current_rank
                    + total_recv_count
                ]
            )
            output_start_current_rank += total_recv_count
            local_recv_ids = torch.arange(
                total_recv_count, dtype=torch.int32, device=torch.device("cuda")
            )
            recv_ids_all_ranks.append(local_recv_ids)

        alltoall_info = MoEAlltoallInfo(
            None,
            send_cumsum_all_ranks[self.rank],
            send_ids_all_ranks[self.rank],
            recv_cumsum_all_ranks[self.rank],
            recv_ids_all_ranks[self.rank],
            None,
            ref_output_tensors_all_ranks[self.rank].shape[0],
        )

        alltoall_workspace = MnnvlMoe.get_moe_workspaces(self.mapping)

        self.comm.Barrier()

        output = MnnvlMoe.mnnvl_moe_alltoallv(
            input_tensors_all_ranks[self.rank],
            alltoall_info,
            alltoall_workspace,
            self.rank,
            self.world_size,
        )

        self.comm.Barrier()

        torch.testing.assert_close(
            output, ref_output_tensors_all_ranks[self.rank], atol=1e-5, rtol=1e-5
        )
