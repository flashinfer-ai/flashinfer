"""
Copyright (c) 2026 by FlashInfer team.

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
from types import SimpleNamespace
from typing import List

import torch

from ..jit.comm import gen_pcie_ipc_ag_rs_module
from ..utils import register_custom_op


@functools.cache
def get_pcie_ipc_ag_rs_module():
    """Build and register the standalone AllGather/ReduceScatter module."""
    module = gen_pcie_ipc_ag_rs_module().build_and_load()

    @register_custom_op(
        "flashinfer::pcie_ipc_all_gather_workspace_size", mutates_args=[]
    )
    def all_gather_workspace_size(
        world_size: int,
        max_numel: int,
        element_size: int,
        max_blocks: int,
        enable_copy_engine: bool,
    ) -> int:
        return module.pcie_ipc_all_gather_workspace_size(
            world_size, max_numel, element_size, max_blocks, enable_copy_engine
        )

    @register_custom_op(
        "flashinfer::pcie_ipc_all_gather_init", mutates_args=["ipc_ptrs"]
    )
    def all_gather_init(
        ipc_ptrs: List[int],
        rank: int,
        max_numel: int,
        element_size: int,
        max_blocks: int,
        enable_copy_engine: bool,
    ) -> int:
        return module.pcie_ipc_all_gather_init(
            ipc_ptrs,
            rank,
            max_numel,
            element_size,
            max_blocks,
            enable_copy_engine,
        )

    @register_custom_op(
        "flashinfer::pcie_ipc_all_gather_dispose", mutates_args=["handle"]
    )
    def all_gather_dispose(handle: int) -> None:
        module.pcie_ipc_all_gather_dispose(handle)

    @register_custom_op("flashinfer::pcie_ipc_all_gather", mutates_args=["out"])
    def all_gather(
        handle: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        blocks: int,
        threads: int,
        variant: int,
    ) -> None:
        module.pcie_ipc_all_gather(handle, inp, out, blocks, threads, variant)

    @register_custom_op(
        "flashinfer::pcie_ipc_reduce_scatter_workspace_size", mutates_args=[]
    )
    def reduce_scatter_workspace_size(
        world_size: int, max_numel: int, element_size: int, max_blocks: int
    ) -> int:
        return module.pcie_ipc_reduce_scatter_workspace_size(
            world_size, max_numel, element_size, max_blocks
        )

    @register_custom_op(
        "flashinfer::pcie_ipc_reduce_scatter_init", mutates_args=["ipc_ptrs"]
    )
    def reduce_scatter_init(
        ipc_ptrs: List[int],
        rank: int,
        max_numel: int,
        element_size: int,
        max_blocks: int,
    ) -> int:
        return module.pcie_ipc_reduce_scatter_init(
            ipc_ptrs, rank, max_numel, element_size, max_blocks
        )

    @register_custom_op(
        "flashinfer::pcie_ipc_reduce_scatter_dispose", mutates_args=["handle"]
    )
    def reduce_scatter_dispose(handle: int) -> None:
        module.pcie_ipc_reduce_scatter_dispose(handle)

    @register_custom_op("flashinfer::pcie_ipc_reduce_scatter", mutates_args=["out"])
    def reduce_scatter(
        handle: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        blocks: int,
        threads: int,
        variant: int,
    ) -> None:
        module.pcie_ipc_reduce_scatter(handle, inp, out, blocks, threads, variant)

    return SimpleNamespace(
        all_gather_workspace_size=all_gather_workspace_size,
        all_gather_init=all_gather_init,
        all_gather_dispose=all_gather_dispose,
        all_gather=all_gather,
        reduce_scatter_workspace_size=reduce_scatter_workspace_size,
        reduce_scatter_init=reduce_scatter_init,
        reduce_scatter_dispose=reduce_scatter_dispose,
        reduce_scatter=reduce_scatter,
    )


__all__ = ["get_pcie_ipc_ag_rs_module"]
