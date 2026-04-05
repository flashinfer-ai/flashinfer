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

CuTe-DSL Kernel Compilation and Caching
=======================================

This module provides functions to compile and cache CuTe-DSL RoPE kernels.
These match the original compilation patterns from flashinfer/cute_dsl/rope.py.
"""

import functools
from typing import Callable, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from flashinfer.cute_dsl.utils import get_cutlass_dtype


def get_cutlass_idtype(idtype_str: str):
    """Convert idtype string to cutlass dtype."""
    if idtype_str == "int32":
        return Int32
    elif idtype_str == "int64":
        return Int64
    else:
        raise ValueError(f"Unsupported idtype: {idtype_str}")


from .kernels import (
    RopeKernelNonInterleavedVec,
    RopeKernelInterleavedVec,
    RopeKernelSeqHeads,
    RopeKernelWithIndptr,
    RopeKernelCosSinCache,
    RopeKernelCosSinCacheSeqHeads,
)


@functools.lru_cache(maxsize=64)
def _get_compiled_kernel_with_indptr(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
    idtype_str: str = "int32",
) -> Callable:
    """Get or compile a cached RoPE kernel that accepts indptr/offsets directly.

    Parameters
    ----------
    idtype_str : str
        Index type for indptr/offsets: "int32" or "int64". Default "int32".
    """
    dtype = get_cutlass_dtype(dtype_str)
    idtype = get_cutlass_idtype(idtype_str)

    kernel_obj = RopeKernelWithIndptr(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim, interleave=interleave
    )

    sym_nnz = cute.sym_int()
    sym_batch_size = cute.sym_int()
    sym_indptr_size = cute.sym_int()  # batch_size + 1
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    # indptr/offsets alignment: 4 for int32, 8 for int64
    idx_align = 4 if idtype_str == "int32" else 8
    indptr_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_indptr_size,), assumed_align=idx_align
    )
    offsets_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_batch_size,), assumed_align=idx_align
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        indptr_fake,
        offsets_fake,
        Int32(1),  # batch_size
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        Float32(1.0),  # rope_rcp_scale
        Float32(1.0),  # rope_rcp_theta
        Float32(0.0),  # smooth_a
        Float32(1.0),  # smooth_b
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        indptr: torch.Tensor,
        offsets: torch.Tensor,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        rope_rcp_scale: float,
        rope_rcp_theta: float,
        smooth_a: float,
        smooth_b: float,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            indptr,
            offsets,
            Int32(batch_size),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
            Float32(rope_rcp_scale),
            Float32(rope_rcp_theta),
            Float32(smooth_a),
            Float32(smooth_b),
        )

    return tensor_api


@functools.lru_cache(maxsize=64)
def _get_compiled_kernel_seq_heads(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
    idtype_str: str = "int32",
) -> Callable:
    """Get or compile a cached sequential-head RoPE kernel.

    This kernel is optimized for large workloads - it loops over heads
    instead of parallelizing, reusing sin/cos computation across heads.

    Parameters
    ----------
    idtype_str : str
        Index type for pos_ids: "int32" or "int64". Default "int32".
    """
    dtype = get_cutlass_dtype(dtype_str)
    idtype = get_cutlass_idtype(idtype_str)

    kernel_obj = RopeKernelSeqHeads(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim, interleave=interleave
    )

    sym_nnz = cute.sym_int()
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    # pos_ids alignment: 4 for int32, 8 for int64
    pos_ids_align = 4 if idtype_str == "int32" else 8
    pos_ids_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_nnz,), assumed_align=pos_ids_align
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        pos_ids_fake,
        Int32(1),  # nnz
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        Float32(1.0),  # rope_rcp_scale
        Float32(1.0),  # rope_rcp_theta
        Float32(0.0),  # smooth_a
        Float32(1.0),  # smooth_b
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        pos_ids: torch.Tensor,
        nnz: int,
        num_qo_heads: int,
        num_kv_heads: int,
        rope_rcp_scale: float,
        rope_rcp_theta: float,
        smooth_a: float,
        smooth_b: float,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            pos_ids,
            Int32(nnz),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
            Float32(rope_rcp_scale),
            Float32(rope_rcp_theta),
            Float32(smooth_a),
            Float32(smooth_b),
        )

    return tensor_api


@functools.lru_cache(maxsize=64)
def _get_compiled_kernel(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
    idtype_str: str = "int32",
) -> Callable:
    """Get or compile a cached RoPE kernel (parallel-heads variant).

    Uses 128-bit vectorized loads/stores for optimal performance.
    This variant parallelizes over heads - use for small workloads.
    For large workloads, use _get_compiled_kernel_seq_heads instead.

    Parameters
    ----------
    idtype_str : str
        Index type for pos_ids: "int32" or "int64". Default "int32".
    """
    dtype = get_cutlass_dtype(dtype_str)
    idtype = get_cutlass_idtype(idtype_str)

    kernel_obj: Union[RopeKernelInterleavedVec, RopeKernelNonInterleavedVec]
    if interleave:
        # Interleaved mode: pairs are adjacent, perfect for 128-bit loads
        kernel_obj = RopeKernelInterleavedVec(
            dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
        )
    else:
        # Non-interleaved: load main + pair vectors using 128-bit loads
        # (like CUDA C++ vec_apply_llama_rope_cos_sin)
        kernel_obj = RopeKernelNonInterleavedVec(
            dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim
        )

    sym_nnz = cute.sym_int()
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    # pos_ids alignment: 4 for int32, 8 for int64
    pos_ids_align = 4 if idtype_str == "int32" else 8
    pos_ids_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_nnz,), assumed_align=pos_ids_align
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        pos_ids_fake,
        Int32(1),  # nnz
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        Float32(1.0),  # rope_rcp_scale
        Float32(1.0),  # rope_rcp_theta
        Float32(0.0),  # smooth_a
        Float32(1.0),  # smooth_b
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        pos_ids: torch.Tensor,
        nnz: int,
        num_qo_heads: int,
        num_kv_heads: int,
        rope_rcp_scale: float,
        rope_rcp_theta: float,
        smooth_a: float,
        smooth_b: float,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            pos_ids,
            Int32(nnz),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
            Float32(rope_rcp_scale),
            Float32(rope_rcp_theta),
            Float32(smooth_a),
            Float32(smooth_b),
        )

    return tensor_api


@functools.lru_cache(maxsize=64)
def _get_compiled_cos_sin_cache_kernel(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
    idtype_str: str = "int32",
) -> Callable:
    """Get or compile a cached RoPE kernel for cos_sin_cache.

    Uses head-parallel approach where each thread block processes one (token, head)
    pair. While this loads cos/sin redundantly for each head, it maximizes GPU
    parallelism which is critical for small workloads (decode). The vectorized
    loads (ld_global_v4_f32) minimize the memory access overhead.

    Parameters
    ----------
    idtype_str : str
        Index type for pos_ids: "int32" or "int64". Default "int32".
    """
    dtype = get_cutlass_dtype(dtype_str)
    idtype = get_cutlass_idtype(idtype_str)

    # Use head-parallel kernel - better for small workloads due to parallelism
    kernel_obj = RopeKernelCosSinCache(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim, interleave=interleave
    )

    sym_nnz = cute.sym_int()
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()
    sym_max_seq_len = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    cos_sin_cache_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_max_seq_len, rotary_dim),
        stride_order=(1, 0),
        assumed_align=4,
    )
    # pos_ids alignment: 4 for int32, 8 for int64
    pos_ids_align = 4 if idtype_str == "int32" else 8
    pos_ids_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_nnz,), assumed_align=pos_ids_align
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        cos_sin_cache_fake,
        pos_ids_fake,
        Int32(1),  # nnz
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        nnz: int,
        num_qo_heads: int,
        num_kv_heads: int,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            Int32(nnz),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
        )

    return tensor_api


@functools.lru_cache(maxsize=64)
def _get_compiled_cos_sin_cache_kernel_seq_heads(
    head_dim: int,
    rotary_dim: int,
    interleave: bool,
    dtype_str: str,
    idtype_str: str = "int32",
) -> Callable:
    """Get or compile a sequential-heads RoPE kernel for cos_sin_cache.

    Uses sequential-heads approach where each thread block processes all heads
    for a token, loading cos/sin from cache ONCE and reusing across heads.
    This reduces memory traffic and is better for large workloads (prefill)
    where memory bandwidth is the bottleneck.

    Parameters
    ----------
    idtype_str : str
        Index type for pos_ids: "int32" or "int64". Default "int32".
    """
    dtype = get_cutlass_dtype(dtype_str)
    idtype = get_cutlass_idtype(idtype_str)

    # Use sequential-heads kernel - better for large workloads (memory-bound)
    kernel_obj = RopeKernelCosSinCacheSeqHeads(
        dtype=dtype, head_dim=head_dim, rotary_dim=rotary_dim, interleave=interleave
    )

    sym_nnz = cute.sym_int()
    sym_num_qo_heads = cute.sym_int()
    sym_num_kv_heads = cute.sym_int()
    sym_max_seq_len = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_rope_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_nnz, sym_num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    cos_sin_cache_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_max_seq_len, rotary_dim),
        stride_order=(1, 0),
        assumed_align=4,
    )
    # pos_ids alignment: 4 for int32, 8 for int64
    pos_ids_align = 4 if idtype_str == "int32" else 8
    pos_ids_fake = cute.runtime.make_fake_compact_tensor(
        idtype, (sym_nnz,), assumed_align=pos_ids_align
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        q_rope_fake,
        k_rope_fake,
        cos_sin_cache_fake,
        pos_ids_fake,
        Int32(1),  # nnz
        Int32(1),  # num_qo_heads
        Int32(1),  # num_kv_heads
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        nnz: int,
        num_qo_heads: int,
        num_kv_heads: int,
    ) -> None:
        compiled_kernel(
            q,
            k,
            q_rope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            Int32(nnz),
            Int32(num_qo_heads),
            Int32(num_kv_heads),
        )

    return tensor_api


__all__ = [
    "_get_compiled_kernel",
    "_get_compiled_kernel_seq_heads",
    "_get_compiled_kernel_with_indptr",
    "_get_compiled_cos_sin_cache_kernel",
    "_get_compiled_cos_sin_cache_kernel_seq_heads",
]
