# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Torch custom-op bindings for dense GEMM."""

from typing import Optional
from b12x._lib.utils.dtypes import cutlass_to_torch_dtype
from b12x._lib.utils.dtypes import get_cutlass_dtype
import torch
from b12x._lib.dense_gemm.buffers import _empty_dense_gemm_output
from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
from b12x._lib.dense_gemm.config import (
    _DenseGemmPolicy,
    _use_direct_m1_wo_a_inputs,
    _use_direct_sfa_live16,
)
from b12x._lib.dense_gemm.reductions import _reduce_split_k2_bf16


def _dense_gemm_launch_flat(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    target_occupancy_override: Optional[int],
    stream_int: Optional[int],
) -> None:
    b_tile_major = b_tensor_gpu.ndim == 5
    policy = _DenseGemmPolicy(
        single_work_tile_per_cta=single_work_tile_per_cta,
        direct_one_m_tile_scheduler=direct_one_m_tile_scheduler,
        use_m1_non_tma=use_m1_non_tma,
        split_k_slices=split_k_slices,
        split_k_atomic_bf16=split_k_atomic_bf16,
        large_m_unroll=large_m_unroll,
    )
    compiled = _get_compiled_dense_gemm(
        n=n,
        k=k,
        l=l,
        c_l=c_l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=get_cutlass_dtype(ab_dtype),
        sf_dtype=get_cutlass_dtype(sf_dtype),
        c_dtype=get_cutlass_dtype(c_dtype),
        alpha_dtype=get_cutlass_dtype(alpha_dtype),
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=(mma_tile_m, mma_tile_n),
        cluster_shape_mn=(cluster_shape_m, cluster_shape_n),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path=load_path,
        swap_ab=swap_ab,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        alpha_is_one=alpha_is_one,
        direct_sfa_live16=_use_direct_sfa_live16(
            m=int(a_tensor_gpu.shape[0]),
            n=n,
            k=k,
            l=l,
            sf_vec_size=sf_vec_size,
            tile_k=tile_k,
            mma_tiler_mn=(mma_tile_m, mma_tile_n),
            load_path=load_path,
            swap_ab=swap_ab,
            b_tile_major=b_tile_major,
            sfb_k_reuse=sfb_k_reuse,
            alpha_is_one=alpha_is_one,
            is_mxfp8=ab_dtype == "float8_e4m3fn",
        ),
        direct_m1_wo_a_inputs=_use_direct_m1_wo_a_inputs(
            m=int(a_tensor_gpu.shape[0]),
            n=n,
            k=k,
            l=l,
            sf_vec_size=sf_vec_size,
            tile_k=tile_k,
            mma_tiler_mn=(mma_tile_m, mma_tile_n),
            load_path=load_path,
            swap_ab=swap_ab,
            b_tile_major=b_tile_major,
            sfb_k_reuse=sfb_k_reuse,
            is_mxfp8=ab_dtype == "float8_e4m3fn",
        ),
        target_occupancy_override=target_occupancy_override,
    )
    compiled(
        a_tensor_gpu=a_tensor_gpu,
        b_tensor_gpu=b_tensor_gpu,
        sfa_tensor_gpu=sfa_tensor_gpu,
        sfb_tensor_gpu=sfb_tensor_gpu,
        c_tensor_gpu=c_tensor_gpu,
        alpha_tensor_gpu=alpha_tensor_gpu,
        stream_int=stream_int,
    )


@torch.library.custom_op(
    "b12x::dense_gemm_launch",
    mutates_args=("c_tensor_gpu",),
)
def _dense_gemm_launch_op(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    target_occupancy_override: Optional[int],
    stream_int: Optional[int],
) -> None:
    _dense_gemm_launch_flat(
        a_tensor_gpu,
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        c_tensor_gpu,
        alpha_tensor_gpu,
        n,
        k,
        l,
        c_l,
        ab_dtype,
        sf_dtype,
        c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        target_occupancy_override,
        stream_int,
    )


@_dense_gemm_launch_op.register_fake
def _dense_gemm_launch_fake(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    target_occupancy_override: Optional[int],
    stream_int: Optional[int],
) -> None:
    return None


@torch.library.custom_op(
    "b12x::dense_gemm_launch_functional",
    mutates_args=(),
)
def _dense_gemm_launch_functional_op(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    kernel_c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    kernel_c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> torch.Tensor:
    m = int(a_tensor_gpu.shape[0])
    out = _empty_dense_gemm_output(
        m,
        n,
        l,
        dtype=cutlass_to_torch_dtype(get_cutlass_dtype(c_dtype)),
        device=a_tensor_gpu.device,
    )
    split_k_output = int(split_k_slices) > 1
    if split_k_output and split_k_atomic_bf16:
        c_tensor_gpu = out
        out.zero_()
    elif split_k_output:
        split_storage = torch.empty(
            (split_k_slices, m, n),
            dtype=torch.float32,
            device=a_tensor_gpu.device,
        )
        c_tensor_gpu = split_storage.permute(1, 2, 0)
    else:
        c_tensor_gpu = out

    _dense_gemm_launch_flat(
        a_tensor_gpu,
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        c_tensor_gpu,
        alpha_tensor_gpu,
        n,
        k,
        l,
        kernel_c_l,
        ab_dtype,
        sf_dtype,
        kernel_c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        None,
        stream_int,
    )
    if split_k_output and not split_k_atomic_bf16:
        _reduce_split_k2_bf16(c_tensor_gpu, out, m=m, n=n)
    return out


@_dense_gemm_launch_functional_op.register_fake
def _dense_gemm_launch_functional_fake(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    kernel_c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    kernel_c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> torch.Tensor:
    del (
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        alpha_tensor_gpu,
        k,
        kernel_c_l,
        ab_dtype,
        sf_dtype,
        kernel_c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        stream_int,
    )
    return _empty_dense_gemm_output(
        int(a_tensor_gpu.shape[0]),
        n,
        l,
        dtype=cutlass_to_torch_dtype(get_cutlass_dtype(c_dtype)),
        device=a_tensor_gpu.device,
    )
