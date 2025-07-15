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

import nvshmem.core
import torch
import torch.distributed as dist
from cuda.core.experimental import Device
from torch.distributed import ProcessGroup

from .nvshmem import PyTorchStreamWrapper


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
        self.dev = Device(self.device.index)
        self.dev.set_current()
        self.pt_stream = torch.cuda.current_stream()
        self.stream = PyTorchStreamWrapper(self.pt_stream)
        self.should_init = should_init
        if self.should_init:
            self.init_nvshmem()

        # assert PE and world size match
        my_pe = nvshmem.core.my_pe()
        n_pes = nvshmem.core.n_pes()
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
        self.symm_buffer_input = nvshmem.core.tensor(
            (max_buffer_elements,),
            dtype=self.dtype,
        )
        self.symm_buffer_output = nvshmem.core.tensor(
            (max_buffer_elements,),
            dtype=self.dtype,
        )

    def init_nvshmem(self):
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if self.local_rank == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = nvshmem.core.get_unique_id()
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]
        torch.distributed.broadcast_object_list(
            broadcast_objects, src=0, group=self.group
        )
        torch.distributed.barrier(self.group)
        nvshmem.core.init(
            device=self.dev,
            uid=broadcast_objects[0],
            rank=self.local_rank,
            nranks=self.world_size,
            initializer_method="uid",
        )

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        numel = inp.numel()
        input_buffer = self.symm_buffer_input.narrow(0, 0, numel)
        output_buffer = self.symm_buffer_output.narrow(0, 0, numel)
        input_buffer.copy_(inp)
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=self.stream)
        nvshmem.core.reduce(
            nvshmem.core.Teams.TEAM_WORLD,
            output_buffer,
            input_buffer,
            "sum",
            stream=self.stream,
        )
        out.copy_(output_buffer)

    def shutdown(self):
        nvshmem.core.free_tensor(self.symm_buffer_input)
        nvshmem.core.free_tensor(self.symm_buffer_output)
        if self.should_init:
            nvshmem.core.finalize()
