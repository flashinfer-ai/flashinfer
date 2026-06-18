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

import logging
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup

import nvshmem.core
from cuda.core import Buffer, Device

logger = logging.getLogger(__name__)

_TORCH_TO_NVSHMEM_DTYPE = {
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
}


def _buffer_view(tensor: torch.Tensor, nelems: int) -> Buffer:
    """Create a Buffer view over the first ``nelems`` elements of an NVSHMEM tensor.

    nvshmem.core.reduce() determines element count from Buffer.size, so passing
    a full pre-allocated symmetric buffer wastes bandwidth when only a prefix is
    needed. This creates a Buffer with the correct size while reusing the same
    pointer (which is already tracked by nvshmem4py's memory manager).

    TODO(flashinfer): Remove this once nvshmem4py exposes register_external_buffer
    with call_register=False through its public API, allowing proper buffer views.
    See: https://github.com/NVIDIA/nvshmem/blob/devel/nvshmem4py/nvshmem/core/nvshmem_types.py
    """
    buf, _, _ = nvshmem.core.tensor_get_buffer(tensor)
    return Buffer.from_handle(
        ptr=int(buf.handle),
        size=nelems * tensor.element_size(),
        mr=buf.memory_resource,
    )


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
        self.global_rank = torch.distributed.get_rank(group)
        self.world_size = world_size
        self.dtype = dtype
        self.device = device
        self.max_buffer_elements = max_buffer_elements
        self.group = group

        self.should_init = should_init
        if self.should_init:
            self.init_nvshmem()

        # assert PE and world size match
        pe = nvshmem.core.my_pe()
        num_pes = nvshmem.core.n_pes()
        if pe != self.global_rank:
            logger.warning(
                "Rank %d: PE mismatch! Expected PE %d, got PE %d",
                self.global_rank,
                self.global_rank,
                pe,
            )
        if num_pes != world_size:
            logger.warning(
                "Rank %d: World size mismatch! Expected %d, got %d",
                self.global_rank,
                world_size,
                num_pes,
            )

        # allocate memory in nvshmem symm heap
        self.symm_buffer_input = nvshmem.core.tensor(
            (max_buffer_elements,), dtype=self.dtype
        )
        self.symm_buffer_output = nvshmem.core.tensor(
            (max_buffer_elements,), dtype=self.dtype
        )
        torch.distributed.barrier(self.group)

    def init_nvshmem(self):
        if self.global_rank == 0:
            uid = nvshmem.core.get_unique_id(empty=False)
        else:
            uid = nvshmem.core.get_unique_id(empty=True)

        # Broadcast uid._data across ranks via torch.distributed
        uid_bytes = np.frombuffer(uid._data.tobytes(), dtype=np.uint8)
        uid_tensor = torch.from_numpy(uid_bytes.copy()).to(dtype=torch.uint8)
        torch.distributed.broadcast(uid_tensor, src=0, group=self.group)

        # Reconstruct uid from broadcasted bytes
        uid._data[:] = np.frombuffer(
            uid_tensor.numpy().tobytes(), dtype=uid._data.dtype
        )

        torch.distributed.barrier(self.group)

        # local_rank selects the GPU; global_rank identifies the PE
        device = Device(self.local_rank)
        nvshmem.core.init(
            device=device,
            uid=uid,
            rank=self.global_rank,
            nranks=self.world_size,
            initializer_method="uid",
        )
        torch.cuda.synchronize()

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        stream = torch.cuda.current_stream()
        nelems = inp.numel()

        # Copy local input to symmetric buffer
        self.symm_buffer_input[:nelems].copy_(inp)
        # Barrier before reduce
        nvshmem.core.barrier_all(stream=stream)
        # Sum reduce across all PEs (only the first nelems elements)
        src_view = _buffer_view(self.symm_buffer_input, nelems)
        dst_view = _buffer_view(self.symm_buffer_output, nelems)
        nvshmem.core.collective_on_buffer(
            "reduce",
            nvshmem.core.Teams.TEAM_WORLD,
            dst_view,
            src_view,
            dtype=_TORCH_TO_NVSHMEM_DTYPE[self.dtype],
            op="sum",
            stream=stream,
        )
        # Copy result back to local output
        out.copy_(self.symm_buffer_output[:nelems])
        stream.synchronize()

    def shutdown(self):
        nvshmem.core.free_tensor(self.symm_buffer_input)
        nvshmem.core.free_tensor(self.symm_buffer_output)
        torch.distributed.barrier(self.group)
        torch.cuda.synchronize()
        if self.should_init:
            nvshmem.core.finalize()
