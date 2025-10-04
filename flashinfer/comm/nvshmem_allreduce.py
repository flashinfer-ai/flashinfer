"""
Copyright (c) 2023 by FlashInfer team.

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

from typing import Optional

import torch
from torch.distributed import ProcessGroup

from .nvshmem import get_nvshmem_module


class NVSHMEMAllReduce:
    """
    An AllReduce implementation for Single-Node and Multi-Node NVLink communication.
    This class handles NVLINK-specific allreduce operations, optimized for NVLink-enabled clusters.
    Note: Requires an active torch.distributed process group to be initialized
    prior to creating an instance of this class.

    Args:
        local_rank (int): The local rank of the current process.
        world_size (int): The total number of processes in the distributed group.
        max_buffer_elements (int): The maximum number of elements that can be stored in
        the buffer. This is used to allocate memory in nvshmem symm heap. set to the
        largest tensor size you will be reducing.
        dtype (torch.dtype): The data type of the tensors to be reduced.
        device (torch.device): The device on which the tensors are located.
        group (torch.distributed.ProcessGroup, optional): The torch.distributed process group to use.
        should_init (bool, optional): Whether to initialize nvshmem. Defaults to True.
    Raises:
        RuntimeError: If nvshmem fails to initialize.
    """

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        max_buffer_elements: int,
        dtype: torch.dtype,
        device: torch.device,
        group: Optional[ProcessGroup] = None,
        should_init: bool = True,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = device
        self.max_buffer_elements = max_buffer_elements
        self.group = group
        self.nvshmem_module = get_nvshmem_module()

        self.should_init = should_init
        if self.should_init:
            self.init_nvshmem()

        # assert PE and world size match
        my_pe = self.nvshmem_module.nvshmem_my_pe()
        n_pes = self.nvshmem_module.nvshmem_n_pes()
        if my_pe != local_rank:
            print(
                f"WARNING: Rank {local_rank}: PE mismatch! Expected PE {local_rank}, got PE {my_pe}",
                flush=True,
            )
        if n_pes != world_size:
            print(
                f"WARNING: Rank {local_rank}: World size mismatch! Expected {world_size}, got {n_pes}",
                flush=True,
            )

        # allocate memory in nvshmem symm heap
        self.symm_buffer_input = self.nvshmem_module.nvshmem_malloc(
            [max_buffer_elements],
            self.dtype,
            self.device.index,
        )
        self.symm_buffer_output = self.nvshmem_module.nvshmem_malloc(
            [max_buffer_elements],
            self.dtype,
            self.device.index,
        )
        torch.distributed.barrier(self.group)

    def init_nvshmem(self):
        uid = torch.zeros(
            self.nvshmem_module.nvshmem_unique_id_size(),
            dtype=torch.uint8,
            device="cpu",
        )
        if self.local_rank == 0:
            self.nvshmem_module.nvshmem_get_unique_id(uid)
        torch.distributed.broadcast(uid, src=0)
        torch.distributed.barrier(self.group)
        init_status = self.nvshmem_module.nvshmem_init(
            uid, self.local_rank, self.world_size
        )
        torch.cuda.synchronize()
        if init_status != 0:
            raise RuntimeError("Failed to initialize nvshmem")

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        self.nvshmem_module.nvshmem_allreduce_on_stream_with_copy(
            self.symm_buffer_output,
            self.symm_buffer_input,
            out,
            inp,
            inp.numel(),
        )

    def shutdown(self):
        del self.symm_buffer_input
        del self.symm_buffer_output
        torch.distributed.barrier(self.group)
        torch.cuda.synchronize()
        if self.should_init:
            self.nvshmem_module.nvshmem_finalize()
