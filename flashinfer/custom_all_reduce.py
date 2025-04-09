# refer to sgl-kernel/python/sgl_kernel/allreduce.py from sglang
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
from typing import List, Tuple

import torch

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import register_custom_op

_comm_module = None
_comm_module_sm90 = None  # todo(yingyi): place holder for sm90


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
                    FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
                    FLASHINFER_CSRC_DIR / "custom_all_reduce.cu",
                ],
            )

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

        @register_custom_op(
            "flashinfer::get_graph_buffer_ipc_meta", mutates_args=["fa"]
        )
        def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
            return module.get_graph_buffer_ipc_meta(fa)

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

        _comm_module = SimpleNamespace(
            init_custom_ar=init_custom_ar,
            dispose=dispose,
            get_graph_buffer_ipc_meta=get_graph_buffer_ipc_meta,
            register_buffer=register_buffer,
            register_graph_buffers=register_graph_buffers,
            meta_size=meta_size,
            all_reduce=all_reduce,
        )

    return _comm_module


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    return get_comm_module().init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)


def dispose(fa: int) -> None:
    get_comm_module().dispose(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
    num_ctas: int,
) -> None:
    get_comm_module().all_reduce(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas
    )


def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
    return get_comm_module().get_graph_buffer_ipc_meta(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    return get_comm_module().register_buffer(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    get_comm_module().register_graph_buffers(fa, handles, offsets)


def meta_size() -> int:
    return get_comm_module().meta_size()
