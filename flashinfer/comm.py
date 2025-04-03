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

from .jit import FLASHINFER_CSRC_DIR, load_cuda_ops
from .utils import register_custom_op

_comm_module = None


def get_comm_module():
    global _comm_module
    if _comm_module is None:
        module = load_cuda_ops(
            "comm",
            [
                FLASHINFER_CSRC_DIR / "comm.cu",
                FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
                FLASHINFER_CSRC_DIR / "customAllReduceKernels.cu"
            ],
        )

        @register_custom_op("flashinfer::all_reduce", mutates_args="data")
        def all_reduce(data: torch.Tensor, reduce_op: int, num_ctas: int = 0) -> None:
            module.all_reduce(data, reduce_op, num_ctas)

        _comm_module = SimpleNamespace(all_reduce=all_reduce)

    return _comm_module


class NetWrapper:

    def __init__(self):
        self.num_ctas = None

    def plan(self, num_ctas):
        self.num_ctas = num_ctas

    def all_reduce(self, data: torch.Tensor, reduce_op: int):
        if self.num_ctas is None:
            raise RuntimeError("NetWrapper.plan() must be called before kernel execution")
        get_comm_module().all_reduce(data, reduce_op, self.num_ctas)
