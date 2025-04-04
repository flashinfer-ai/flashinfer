"""
Copyright (c) 2024 by FlashInfer team.

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

from types import SimpleNamespace

import torch

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import register_custom_op

_comm_module = None
_comm_module_sm90 = None # todo(yingyi): place holder for sm90


def get_comm_module():
    global _comm_module
    if _comm_module is None:
        if has_prebuilt_ops:
            _kernels = torch.ops.flashinfer_kernels
            module = _kernels
        else:
            module = load_cuda_ops(
                "comm",
                [
                FLASHINFER_CSRC_DIR / "comm.cu",
                FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
                FLASHINFER_CSRC_DIR / "customAllReduceKernels.cu"
            ],
        )

        # torch library for all

        @register_custom_op("flashinfer::all_reduce", mutates_args=["data", "workspace_buffer"])
        def all_reduce(data: torch.Tensor, workspace_buffer: torch.Tensor, world_size: int, rank: int, num_ctas: int) -> None:
            module.all_reduce(data, workspace_buffer, world_size, rank, num_ctas)

        _comm_module = SimpleNamespace(all_reduce=all_reduce) # todo(yingyi): 

    return _comm_module


class NetWrapper:

    def __init__(self):
        self.num_ctas = None
        # need to specify device for workspace buffer?
        self.workspace_buffer = torch.empty((1024 * 1024,), dtype=torch.int8)

    def plan(self, num_ctas):
        self.num_ctas = num_ctas

    def all_reduce(self, data: torch.Tensor):
        if self.num_ctas is None:
            # raise RuntimeError("NetWrapper.plan() must be called before kernel execution")
            pass
        get_comm_module().all_reduce(
            data, self.workspace_buffer, 
            torch.distributed.get_world_size(),
            torch.distributed.get_rank(),
            self.num_ctas,
        )

def all_reduce_sm_constraint(data: torch.Tensor, reduce_op: int, num_ctas: int = None, fusion_op: int = None):
    r"""AllReduce with SM constraint

    Parameters
    ----------
    data: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3, fp16, bf16
        todo(yingyi): add dtype support check
    reduce_op: int
        Reduction operation, todo(yingyi): add reduction op mapping
    num_ctas: int
        Number of CTAs to use
    fusion_op: int
        todo(yingyi): add fusion op support
    
    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> data = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> flashinfer.all_reduce_sm_constraint...) todo(yingyi)
    ...
    """

    if data == None:
        raise ValueError("data is None")

    world_size = 0 # todo(yingyi): get world size: pp+mp+dp+ep calculation or mpi?
    rank = 0 # todo(yingyi): get rank

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # todo(yingyi): default 1/3 of total SMs
    num_ctas = NUM_SMS if num_ctas is None else min(NUM_SMS // 3, num_ctas) 

    # todo(yingyi): workspace buffer palce holder
    # import workspace constructor from trtllm: benchmarks/python/all_reduce.py
    workspace_buffer = torch.empty((1024 * 1024,), dtype=data.dtype, device=data.device)
    get_comm_module().all_reduce_sm_constraint(data, workspace_buffer, world_size, rank, num_ctas, fusion_op)
    

