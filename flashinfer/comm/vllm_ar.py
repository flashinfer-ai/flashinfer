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
from types import SimpleNamespace
from typing import List, Tuple

import torch

from ..jit.comm import gen_vllm_comm_module
from ..utils import register_custom_op


@functools.cache
def get_vllm_comm_module():
    module = gen_vllm_comm_module().build_and_load()

    # torch library for all
    @register_custom_op(
        "flashinfer::init_custom_ar",
        mutates_args=["ipc_ptrs", "rank_data", "rank", "full_nvlink"],
    )
    def init_custom_ar(
        ipc_ptrs: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        return module.init_custom_ar(ipc_ptrs, rank_data, rank, full_nvlink)

    @register_custom_op("flashinfer::dispose", mutates_args=["fa"])
    def dispose(fa: int) -> None:
        module.dispose(fa)

    @register_custom_op("flashinfer::get_graph_buffer_ipc_meta", mutates_args=["fa"])
    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        output_bytes, output_offsets = module.get_graph_buffer_ipc_meta(fa)
        return list(output_bytes), list(output_offsets)

    @register_custom_op(
        "flashinfer::register_buffer", mutates_args=["fa", "fake_ipc_ptrs"]
    )
    def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
        return module.register_buffer(fa, fake_ipc_ptrs)

    @register_custom_op(
        "flashinfer::register_graph_buffers",
        mutates_args=["fa", "handles", "offsets"],
    )
    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        module.register_graph_buffers(fa, handles, offsets)

    @register_custom_op("flashinfer::meta_size", mutates_args=[])
    def meta_size() -> int:
        return module.meta_size()

    @register_custom_op(
        "flashinfer::all_reduce",
        mutates_args=["out", "reg_buffer", "reg_buffer_sz_bytes"],
    )
    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
        num_ctas: int,
    ) -> None:
        module.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas)

    return SimpleNamespace(
        init_custom_ar=init_custom_ar,
        dispose=dispose,
        get_graph_buffer_ipc_meta=get_graph_buffer_ipc_meta,
        register_buffer=register_buffer,
        register_graph_buffers=register_graph_buffers,
        meta_size=meta_size,
        all_reduce=all_reduce,
    )


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    r"""Initialize the vLLM custom all-reduce backend.

    Parameters
    ----------
    ipc_tensors : list[int]
        IPC pointers to the per-rank communication buffers.
    rank_data : torch.Tensor
        Scratch tensor (one per rank) used for metadata exchange.
    rank : int
        Current rank within the all-reduce world.
    full_nvlink : bool
        ``True`` when every pair of ranks is connected via NVLink (enables
        the fully-NVLink-optimized kernel path).

    Returns
    -------
    int
        Opaque handle (``fa``) to be passed to subsequent ``vllm_ar`` calls.
    """
    return get_vllm_comm_module().init_custom_ar(
        ipc_tensors, rank_data, rank, full_nvlink
    )


def dispose(fa: int) -> None:
    r"""Release the resources held by a vLLM custom all-reduce handle.

    Parameters
    ----------
    fa : int
        Handle returned by :func:`init_custom_ar`.
    """
    get_vllm_comm_module().dispose(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
    num_ctas: int,
) -> None:
    r"""Perform an out-of-place all-reduce via the vLLM custom kernel.

    Parameters
    ----------
    fa : int
        Handle returned by :func:`init_custom_ar`.
    inp : torch.Tensor
        Input tensor (rank-local contribution).
    out : torch.Tensor
        Pre-allocated output tensor (receives the reduced result).
    reg_buffer : int
        Device pointer to the registered buffer used by the kernel.
    reg_buffer_sz_bytes : int
        Size of ``reg_buffer`` in bytes.
    num_ctas : int
        Number of CTAs to launch.  Upper bound is 36; small values are
        usually enough to saturate NVLink bandwidth.
    """
    get_vllm_comm_module().all_reduce(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas
    )


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
    r"""Return IPC metadata for graph-capture buffers.

    Parameters
    ----------
    fa : int
        Handle returned by :func:`init_custom_ar`.

    Returns
    -------
    Tuple[list[int], list[int]]
        ``(output_bytes, output_offsets)`` describing the registered
        graph-capture buffers.  Used to publish IPC handles to peer ranks.
    """
    return get_vllm_comm_module().get_graph_buffer_ipc_meta(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    r"""Register a peer's IPC-shared buffer with the local all-reduce handle.

    Parameters
    ----------
    fa : int
        Handle returned by :func:`init_custom_ar`.
    fake_ipc_ptrs : list[int]
        Per-rank IPC pointers obtained from each peer.
    """
    return get_vllm_comm_module().register_buffer(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    r"""Register graph-capture buffers across the all-reduce world.

    Parameters
    ----------
    fa : int
        Handle returned by :func:`init_custom_ar`.
    handles : list[list[int]]
        Per-rank IPC handles published via :func:`get_graph_buffer_ipc_meta`.
    offsets : list[list[int]]
        Per-rank offsets matching ``handles``.
    """
    get_vllm_comm_module().register_graph_buffers(fa, handles, offsets)


def meta_size() -> int:
    r"""Return the size of the vLLM all-reduce metadata structure in bytes."""
    return get_vllm_comm_module().meta_size()
