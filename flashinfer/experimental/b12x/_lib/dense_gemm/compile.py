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

"""Dense GEMM launch wrappers and compilation factories."""

from b12x._lib.quant.block_codec import BLOCK_CODECS
from typing import Callable
from cutlass import Int32
from b12x._lib.compiler import KernelCompileSpec
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.quant.block_codec import block_codec
import cuda.bindings.driver as cuda
from b12x._lib.utils.device import cuda_stream_from_int_or_current
from b12x._lib.utils.device import cuda_stream_to_int
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.utils.dtypes import cutlass_to_torch_dtype
from b12x._lib.utils.device import get_num_sm
from b12x._lib.utils.pointers import make_ptr
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
import torch
from b12x._lib.dense_gemm.buffers import _cached_alpha_one, _empty_dense_gemm_output
from b12x._lib.dense_gemm.config import (
    _PACKED_B_EXPAND_AHEAD,
    _B12X_DENSE_ATOM_24,
    _DENSE_FUSED_QUANT,
    _DenseGemmPolicy,
    _dense_gemm_policy_for,
    _max_active_clusters_for,
    _tile_major_cluster_limit,
)
from b12x._lib.dense_gemm.kernel import DenseGemmKernel
from b12x._lib.dense_gemm.policy import _select_default_dense_gemm_plan


class _DenseGemmLaunch:
    def __init__(
        self,
        n: int,
        k: int,
        l: int,
        c_l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_k: int,
        tile_k: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        policy: _DenseGemmPolicy,
        sm_count: int,
        sm_version: str,
        load_path: str,
        swap_ab: bool,
        sfb_k_reuse: bool,
        b_tile_major: bool = False,
        quantize_c: bool = False,
        alpha_is_one: bool = False,
        direct_sfa_live16: bool = False,
        direct_m1_wo_a_inputs: bool = False,
        target_occupancy: int = 1,
        plain_fp8: bool = False,
        block_fp8: bool = False,
        weight_only: Optional[str] = None,
        mma_warps: Optional[int] = None,
        alpha_reciprocal: bool = False,
        input_k: Optional[int] = None,
    ):
        self._weight_only = weight_only
        self._mma_warps = mma_warps
        self._alpha_reciprocal = alpha_reciprocal
        self._input_k = k if input_k is None else input_k
        self._storage_k = k // (
            block_codec(weight_only).pack_factor
            if weight_only in BLOCK_CODECS
            else 2
            if weight_only == "nvfp4"
            else 1
        )
        self._n = n
        self._k = k
        self._l = l
        self._c_l = c_l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_k = mma_k
        self._tile_k = tile_k
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn
        self._policy = policy
        self._sm_count = sm_count
        self._sm_version = sm_version
        self._load_path = load_path
        self._swap_ab = swap_ab
        # This experimental atom choice changes generated code. Capture the
        # import-time setting on the launch so both in-process resolution and
        # the persistent object cache distinguish it.
        self._atom_shape_24 = _B12X_DENSE_ATOM_24
        self._sfb_k_reuse = sfb_k_reuse
        self._b_tile_major = b_tile_major
        self._quantize_c = quantize_c
        self._alpha_is_one = alpha_is_one
        self._direct_sfa_live16 = direct_sfa_live16
        self._direct_m1_wo_a_inputs = direct_m1_wo_a_inputs
        self._target_occupancy = target_occupancy
        self._plain_fp8 = bool(plain_fp8)
        self._block_fp8 = bool(block_fp8)
        if b_tile_major:
            if (n, k, l) == (1024, 4096, 4):
                self._b_tile_n = 64
            elif (n, k, l) == (4096, 4096, 1):
                self._b_tile_n = 128
            else:
                raise ValueError(
                    "tile-major B is restricted to production WO-A/WO-B shapes, "
                    f"got {(n, k, l)}"
                )
        else:
            self._b_tile_n = 0
        if not DenseGemmKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
            load_path=load_path,
            swap_ab=swap_ab,
            block_fp8=block_fp8,
            weight_only=weight_only,
        ):
            raise TypeError(
                "dense_gemm launch is unsupported with "
                f"{ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
                f"{mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {l}, "
                f"{a_major}, {b_major}, {c_major}, "
                f"load_path={load_path}, swap_ab={swap_ab}"
            )

        self._max_active_clusters = (
            _max_active_clusters_for(self._cluster_shape_mn, sm_count)
            * self._target_occupancy
        )
        if (
            mma_tiler_mn == (32, 64)
            and tile_k == 128
            and b_tile_major
            and sfb_k_reuse
            and alpha_is_one
        ):
            self._max_active_clusters = _tile_major_cluster_limit(
                self._max_active_clusters,
                n=n,
                l=l,
                tile_n=mma_tiler_mn[1],
            )

    def compile_key(self) -> tuple[object, ...]:
        """Return every value that can specialize the generated kernel."""

        key = (
            self._n,
            self._k,
            self._l,
            self._c_l,
            self._a_major,
            self._b_major,
            self._c_major,
            self._ab_dtype,
            self._sf_dtype,
            self._c_dtype,
            self._alpha_dtype,
            self._sf_vec_size,
            self._mma_k,
            self._tile_k,
            self._mma_tiler_mn,
            self._cluster_shape_mn,
            self._policy,
            self._sm_count,
            self._max_active_clusters,
            self._sm_version,
            self._load_path,
            self._swap_ab,
            self._atom_shape_24,
            self._sfb_k_reuse,
            self._b_tile_major,
            self._b_tile_n,
            self._quantize_c,
            self._alpha_is_one,
            self._direct_sfa_live16,
            self._direct_m1_wo_a_inputs,
            self._target_occupancy,
            self._plain_fp8,
        )

        return (
            key
            if self._weight_only is None
            else key
            + (
                self._weight_only,
                self._mma_warps,
                self._alpha_reciprocal,
                self._input_k,
            )
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        quant_c_values_ptr: cute.Pointer,
        quant_c_scale_rows_ptr: cute.Pointer,
        quant_c_scale_mma_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (m, self._input_k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        if cutlass.const_expr(self._weight_only in BLOCK_CODECS):
            descriptor_tile_n = 128 if self._n % 128 == 0 else 8
            spec = block_codec(self._weight_only)
            b_layout = cute.make_layout(
                (
                    (descriptor_tile_n, self._n // descriptor_tile_n),
                    (spec.payload_bytes, self._k // spec.block_weights),
                    self._l,
                ),
                stride=(
                    (spec.payload_bytes, descriptor_tile_n * self._storage_k),
                    (1, descriptor_tile_n * spec.payload_bytes),
                    self._n * self._storage_k,
                ),
            )
        elif cutlass.const_expr(self._b_tile_major):
            b_layout = cute.make_layout(
                (
                    (self._b_tile_n, self._n // self._b_tile_n),
                    (128, self._k // 128),
                    self._l,
                ),
                stride=(
                    (128, self._b_tile_n * self._k),
                    (1, self._b_tile_n * 128),
                    self._n * self._k,
                ),
            )
        else:
            b_layout = cute.make_ordered_layout(
                (self._n, self._storage_k, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            )
        b_tensor = cute.make_tensor(b_ptr, layout=b_layout)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, self._n, self._c_l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        quant_c_width = self._n * self._l
        quant_c_chunks = max(1, (quant_c_width + 31) // 32)
        quant_c_k_tiles = max(1, (quant_c_width + 127) // 128)
        quant_c_values_tensor = cute.make_tensor(
            quant_c_values_ptr,
            layout=cute.make_ordered_layout((m, quant_c_width), order=(1, 0)),
        )
        quant_c_scale_rows_tensor = cute.make_tensor(
            quant_c_scale_rows_ptr,
            layout=cute.make_ordered_layout((m, quant_c_chunks), order=(1, 0)),
        )
        quant_c_scale_mma_tensor = cute.make_tensor(
            quant_c_scale_mma_ptr,
            layout=cute.make_layout(
                (32, 4, 1, 4, quant_c_k_tiles, 1),
                stride=(
                    16,
                    4,
                    quant_c_k_tiles * 512,
                    1,
                    512,
                    quant_c_k_tiles * 512,
                ),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        if cutlass.const_expr(self._weight_only is not None):
            scale_bytes = (
                ((self._n + 127) // 128)
                * ((self._k // self._sf_vec_size + 3) // 4)
                * 512
            )
            if cutlass.const_expr(self._weight_only in BLOCK_CODECS):
                scale_bytes = (
                    self._n
                    * (self._k // block_codec(self._weight_only).block_weights)
                    * 2
                    if self._weight_only != "iq2_xs"
                    else ((self._n + 127) // 128) * (self._k // 256) * 1280
                )
            sfa_tensor = cute.make_tensor(
                sfa_ptr,
                layout=cute.make_layout(
                    (
                        max(1, block_codec(self._weight_only).lut_bytes())
                        if self._weight_only in BLOCK_CODECS
                        else 1,
                    )
                ),
            )
            sfb_tensor = cute.make_tensor(
                sfb_ptr, layout=cute.make_layout((scale_bytes,))
            )
        elif cutlass.const_expr(self._block_fp8):
            sfa_tensor = cute.make_tensor(
                sfa_ptr,
                layout=cute.make_ordered_layout(
                    (m, self._k // 128, self._l), order=(1, 0, 2)
                ),
            )
            sfb_tensor = cute.make_tensor(
                sfb_ptr,
                layout=cute.make_ordered_layout(
                    (self._n // 128, self._k // 128, self._l),
                    order=(1, 0, 2),
                ),
            )
        else:
            sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
            sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=policy.large_m_unroll,
            # M=1 FP8 benefits from normal TMA loads for A/SFA on the
            # standalone tiny-M profile. Keep C on the direct epilogue path;
            # the normal TMA store did not beat it in the DSV4F TP=2 GPU5 run.
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma and not self._swap_ab,
            use_m1_non_tma_sfa=False,
            load_path=self._load_path,
            swap_ab=self._swap_ab,
            sfb_k_reuse=self._sfb_k_reuse,
            atom_shape_24=self._atom_shape_24,
            b_tile_major=self._b_tile_major,
            quantize_c=self._quantize_c,
            alpha_is_one=self._alpha_is_one,
            direct_sfa_live16=self._direct_sfa_live16,
            direct_m1_wo_a_inputs=self._direct_m1_wo_a_inputs,
            target_occupancy=self._target_occupancy,
            plain_fp8=self._plain_fp8,
            block_fp8=self._block_fp8,
            weight_only=self._weight_only,
            mma_warps=self._mma_warps,
            alpha_reciprocal=self._alpha_reciprocal,
        )(
            a_tensor,
            a_tensor,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            quant_c_values_tensor,
            quant_c_scale_rows_tensor,
            quant_c_scale_mma_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


class _DenseGemmMxfp6Launch(_DenseGemmLaunch):
    """MX-FP6 (W6A6/W6A8) launch: FP6 codes in Float8E4M3FN byte-containers.

    Adds the per-operand MX sub-formats, optional native 3:4-packed B
    streaming, and the optional fused BF16->FP6 activation-quant prologue
    (B12X_DENSE_FUSED_QUANT). Only the unswapped single-slice TMA plan
    is wired; the constructor of DenseGemmKernel asserts the rest.
    """

    def __init__(
        self,
        *args,
        mxfp6_fmt: Optional[str] = None,
        mxfp6_fmt_a: Optional[str] = None,
        mxfp6_fmt_b: Optional[str] = None,
        b_packed: bool = False,
        a_preexpanded: bool = False,
        b_preexpanded: bool = False,
        row_scale: bool = False,
        fused_quant: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mxfp6_fmt_a is None and mxfp6_fmt_b is None:
            mxfp6_fmt_a = mxfp6_fmt
            mxfp6_fmt_b = mxfp6_fmt
        elif mxfp6_fmt_a is None or mxfp6_fmt_b is None:
            raise ValueError(
                "mxfp6_fmt_a and mxfp6_fmt_b must both be set or both None"
            )
        self._mxfp6_fmt_a = mxfp6_fmt_a
        self._mxfp6_fmt_b = mxfp6_fmt_b
        self._b_packed = bool(b_packed)
        self._packed_b_expand_ahead = self._b_packed and _PACKED_B_EXPAND_AHEAD
        # Key-only host-side flags (do not change codegen, kept for cache
        # hygiene) plus the import-time fused-quant knob, which DOES change
        # the generated kernel via DenseGemmKernel.a_bf16_fused.
        self._a_preexpanded = bool(a_preexpanded)
        self._b_preexpanded = bool(b_preexpanded)
        self._fused_quant_env = (
            _DENSE_FUSED_QUANT if fused_quant is None else bool(fused_quant)
        )
        self._row_scale = bool(row_scale)

    def compile_key(self) -> tuple[object, ...]:
        return (
            "mxfp6",
            self._mxfp6_fmt_a,
            self._mxfp6_fmt_b,
            self._b_packed,
            self._packed_b_expand_ahead,
            self._a_preexpanded,
            self._b_preexpanded,
            self._fused_quant_env,
            self._row_scale,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        x_bf16_ptr: cute.Pointer,
        w_gscale_ptr: cute.Pointer,
        row_scale_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        # Packed B carries 3 bytes per 4 codes: gmem extent is 3K/4.
        b_k_extent = self._k * 3 // 4 if self._b_packed else self._k
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, b_k_extent, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, self._n, self._c_l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        x_bf16_tensor = cute.make_tensor(
            x_bf16_ptr,
            layout=cute.make_ordered_layout((m, self._k), order=(0, 1)),
        )
        w_gscale_tensor = cute.make_tensor(
            w_gscale_ptr,
            layout=cute.make_ordered_layout((Int32(1),), order=(0,)),
        )
        if cutlass.const_expr(self._row_scale):
            # Row scaling and quantized-C are mutually exclusive, so the
            # existing quant_c_values operand carries the FP6 epilogue scale.
            row_scale_tensor = cute.make_tensor(
                row_scale_ptr,
                layout=cute.make_ordered_layout((m,), order=(0,)),
            )
        else:
            row_scale_tensor = alpha_tensor
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=policy.large_m_unroll,
            # The M=1 FP6 shape uses a 128-row MMA/TMA tile. In this lowering,
            # the narrow A/SFA/C tensor-map paths illegal-instruction instead
            # of relying on OOB handling, so keep them as explicit direct paths
            # (also required by the fused activation-quant prologue).
            use_m1_non_tma_a=policy.use_m1_non_tma,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=policy.use_m1_non_tma,
            load_path=self._load_path,
            swap_ab=self._swap_ab,
            sfb_k_reuse=self._sfb_k_reuse,
            alpha_is_one=self._alpha_is_one,
            mxfp6_fmt_a=self._mxfp6_fmt_a,
            mxfp6_fmt_b=self._mxfp6_fmt_b,
            b_packed=self._b_packed,
            target_occupancy=self._target_occupancy,
            row_scale=self._row_scale,
            fused_quant_bf16=self._fused_quant_env,
        )(
            a_tensor,
            a_tensor,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            row_scale_tensor,
            alpha_tensor,
            alpha_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
            x_bf16=x_bf16_tensor,
            w_gscale=w_gscale_tensor,
        )


@program_cache
def _get_compiled_dense_gemm_mxfp6(
    n: int,
    k: int,
    l: int,
    c_l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    policy: _DenseGemmPolicy,
    sm_count: int,
    sm_version: str,
    mxfp6_fmt_a: Optional[str],
    mxfp6_fmt_b: Optional[str],
    b_packed: bool,
    a_preexpanded: bool,
    b_preexpanded: bool,
    alpha_is_one: bool,
    row_scale: bool,
    fused_quant: bool,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
                x_bf16_data_ptr,
                w_gscale_data_ptr,
                row_scale_data_ptr,
            ) = [16 for _ in range(9)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
                x_bf16_tensor_gpu,
                w_gscale_tensor_gpu,
                row_scale_tensor_gpu,
            ) = input_tensors
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                alpha_tensor_gpu.data_ptr(),
            )
            x_bf16_data_ptr = (
                x_bf16_tensor_gpu.data_ptr() if x_bf16_tensor_gpu is not None else 16
            )
            w_gscale_data_ptr = (
                w_gscale_tensor_gpu.data_ptr()
                if w_gscale_tensor_gpu is not None
                else 16
            )
            row_scale_data_ptr = (
                row_scale_tensor_gpu.data_ptr()
                if row_scale_tensor_gpu is not None
                else 16
            )

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16,
                x_bf16_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                w_gscale_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                c_dtype,
                row_scale_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
        ]

    launch = _DenseGemmMxfp6Launch(
        n=n,
        k=k,
        l=l,
        c_l=c_l,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        policy=policy,
        sm_count=sm_count,
        sm_version=sm_version,
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=False,
        alpha_is_one=alpha_is_one,
        mxfp6_fmt_a=mxfp6_fmt_a,
        mxfp6_fmt_b=mxfp6_fmt_b,
        b_packed=b_packed,
        a_preexpanded=a_preexpanded,
        b_preexpanded=b_preexpanded,
        row_scale=row_scale,
        fused_quant=fused_quant,
        # MX-FP6 uses the same occupancy policy as the other dense kernels.
        target_occupancy=_dense_gemm_target_occupancy(
            n=n,
            k=k,
            l=l,
            ab_dtype=ab_dtype,
            c_dtype=c_dtype,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sm_count=sm_count,
            load_path="tma",
            swap_ab=False,
            b_tile_major=False,
            is_mxfp6=True,
        ),
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = b12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense.mxfp6", 1, compile_key),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        stream_int: Optional[int] = None,
        x_bf16_tensor_gpu: Optional[torch.Tensor] = None,
        w_gscale_tensor_gpu: Optional[torch.Tensor] = None,
        row_scale_tensor_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a_tensor_gpu.shape[0]
        if c_tensor_gpu is None:
            c_tensor_gpu = _empty_dense_gemm_output(
                int(m),
                n,
                c_l,
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = _cached_alpha_one(a_tensor_gpu.device)

        nonlocal compiled_kernel
        compiled_kernel(
            *_make_runtime_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    alpha_tensor_gpu,
                    x_bf16_tensor_gpu,
                    w_gscale_tensor_gpu,
                    row_scale_tensor_gpu,
                ]
            ),
            m,
            cuda_stream_from_int_or_current(stream_int),
        )
        return c_tensor_gpu

    return attach_programs(tensor_api, compiled_kernel)


class _DenseGemmFusedQuantALaunch(_DenseGemmLaunch):
    """Small-M MXFP8 launch that quantizes BF16 A into each CTA's stages."""

    def __init__(
        self,
        *args,
        fused_quant_a_inner_span: int = 0,
        fused_quant_a_wide: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fused_quant_a_inner_span = int(fused_quant_a_inner_span)
        self._fused_quant_a_wide = bool(fused_quant_a_wide)

    def compile_key(self) -> tuple[object, ...]:
        # Keep the fused entry point separate even if every ordinary launch
        # specialization matches. Deriving the rest from the parent key avoids
        # silently omitting a future codegen field here.
        return (
            "fused_quant_a",
            self._fused_quant_a_inner_span,
            self._fused_quant_a_wide,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_placeholder_ptr: cute.Pointer,
        a_source_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_placeholder_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_placeholder_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        a_source = cute.make_tensor(
            a_source_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        if cutlass.const_expr(self._b_tile_major):
            b_layout = cute.make_layout(
                (
                    (self._b_tile_n, self._n // self._b_tile_n),
                    (128, self._k // 128),
                    1,
                ),
                stride=(
                    (128, self._b_tile_n * self._k),
                    (1, self._b_tile_n * 128),
                    self._n * self._k,
                ),
            )
        else:
            b_layout = cute.make_ordered_layout((self._n, self._k, 1), order=(1, 0, 2))
        b_tensor = cute.make_tensor(b_ptr, layout=b_layout)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, self._n, self._c_l), order=(1, 0, 2)),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(
            sfa_placeholder_ptr, layout=cute.make_layout((1,))
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=False,
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=False,
            load_path="tma",
            swap_ab=False,
            sfb_k_reuse=self._sfb_k_reuse,
            fused_quant_a=True,
            fused_quant_a_inner_span=self._fused_quant_a_inner_span,
            fused_quant_a_wide=self._fused_quant_a_wide,
            atom_shape_24=self._atom_shape_24,
            b_tile_major=self._b_tile_major,
            target_occupancy=self._target_occupancy,
        )(
            a_tensor,
            a_source,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            a_tensor,
            sfa_tensor,
            sfa_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


@program_cache
def _get_compiled_dense_gemm_fused_quant_a(
    n: int,
    k: int,
    c_dtype: Type[cutlass.Numeric],
    policy: _DenseGemmPolicy,
    mma_tiler_mn: Tuple[int, int],
    sm_count: int,
    sfb_k_reuse: bool,
    b_tile_major: bool,
    a_inner_span: int = 0,
    kernel_c_l: int = 1,
    a_wide: bool = False,
) -> Callable:
    launch = _DenseGemmFusedQuantALaunch(
        n=n,
        k=k,
        l=1,
        c_l=kernel_c_l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=c_dtype,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=128,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        fused_quant_a_inner_span=a_inner_span,
        fused_quant_a_wide=a_wide,
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    placeholders = [16] * 7
    compiled = b12x_compile(
        launch,
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[0],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[1], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[2],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[3],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[4],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(c_dtype, placeholders[5], cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.Float32, placeholders[6], cute.AddressSpace.gmem, assumed_align=16
        ),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.dense_fused_quant_a", 4, compile_key
        ),
    )

    def tensor_api(
        source: torch.Tensor,
        b: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        alpha: torch.Tensor,
        stream_int: Optional[int],
    ) -> torch.Tensor:
        source_ptr = source.data_ptr()
        compiled(
            make_ptr(
                cutlass.Float8E4M3FN,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16, source_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                sfb.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(c_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.Float32,
                alpha.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(source.shape[0]),
            cuda_stream_from_int_or_current(stream_int),
        )
        return out

    return attach_programs(tensor_api, compiled)


class _DenseGemmFusedQuantAGroupedLaunch(_DenseGemmLaunch):
    """Grouped small-M MXFP8 launch quantizing a strided (optionally
    inverse-RoPE) BF16 A source into each CTA's stages (WO-A)."""

    def __init__(
        self,
        *args,
        fused_quant_a_row_stride: int,
        fused_quant_a_l_stride: int,
        fused_quant_a_inv_rope: bool,
        fused_quant_a_head_dim: int,
        fused_quant_a_nope_dim: int,
        fused_quant_a_rope_dim: int,
        fused_quant_a_wide: bool,
        positions_dtype: Type[cutlass.Numeric],
        cos_sin_dtype: Type[cutlass.Numeric],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fused_quant_a_row_stride = int(fused_quant_a_row_stride)
        self._fused_quant_a_l_stride = int(fused_quant_a_l_stride)
        self._fused_quant_a_inv_rope = bool(fused_quant_a_inv_rope)
        self._fused_quant_a_head_dim = int(fused_quant_a_head_dim)
        self._fused_quant_a_nope_dim = int(fused_quant_a_nope_dim)
        self._fused_quant_a_rope_dim = int(fused_quant_a_rope_dim)
        self._fused_quant_a_wide = bool(fused_quant_a_wide)
        self._positions_dtype = positions_dtype
        self._cos_sin_dtype = cos_sin_dtype

    def compile_key(self) -> tuple[object, ...]:
        return (
            "fused_quant_a_grouped",
            self._fused_quant_a_row_stride,
            self._fused_quant_a_l_stride,
            self._fused_quant_a_inv_rope,
            self._fused_quant_a_head_dim,
            self._fused_quant_a_nope_dim,
            self._fused_quant_a_rope_dim,
            self._fused_quant_a_wide,
            self._positions_dtype,
            self._cos_sin_dtype,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_placeholder_ptr: cute.Pointer,
        a_source_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_placeholder_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        cos_sin_len: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_placeholder_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        a_source = cute.make_tensor(
            a_source_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        positions_tensor = cute.make_tensor(
            positions_ptr, layout=cute.make_layout((m,))
        )
        cos_sin_tensor = cute.make_tensor(
            cos_sin_ptr, layout=cute.make_layout((cos_sin_len,))
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, self._k, self._l), order=(1, 0, 2)
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, self._n, self._c_l), order=(1, 0, 2)),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(
            sfa_placeholder_ptr, layout=cute.make_layout((1,))
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=False,
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=False,
            load_path="tma",
            swap_ab=False,
            sfb_k_reuse=self._sfb_k_reuse,
            fused_quant_a=True,
            fused_quant_a_row_stride=self._fused_quant_a_row_stride,
            fused_quant_a_l_stride=self._fused_quant_a_l_stride,
            fused_quant_a_inv_rope=self._fused_quant_a_inv_rope,
            fused_quant_a_head_dim=self._fused_quant_a_head_dim,
            fused_quant_a_nope_dim=self._fused_quant_a_nope_dim,
            fused_quant_a_rope_dim=self._fused_quant_a_rope_dim,
            fused_quant_a_wide=self._fused_quant_a_wide,
            atom_shape_24=self._atom_shape_24,
            target_occupancy=self._target_occupancy,
        )(
            a_tensor,
            a_source,
            positions_tensor,
            cos_sin_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            a_tensor,
            sfa_tensor,
            sfa_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


def _cutlass_positions_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    if dtype == torch.int64:
        return cutlass.Int64
    if dtype == torch.int32:
        return cutlass.Int32
    raise ValueError(f"fused inv-RoPE positions must be int32/int64, got {dtype}")


def _cutlass_cos_sin_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"fused inv-RoPE cos/sin cache must be bf16/fp32, got {dtype}")


@program_cache
def _get_compiled_dense_gemm_fused_quant_a_grouped(
    n: int,
    k: int,
    l: int,
    policy: _DenseGemmPolicy,
    mma_tiler_mn: Tuple[int, int],
    sm_count: int,
    sfb_k_reuse: bool,
    a_row_stride: int,
    a_l_stride: int,
    inv_rope: bool,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    a_wide: bool,
    positions_dtype: Type[cutlass.Numeric],
    cos_sin_dtype: Type[cutlass.Numeric],
) -> Callable:
    launch = _DenseGemmFusedQuantAGroupedLaunch(
        n=n,
        k=k,
        l=l,
        c_l=l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=cutlass.BFloat16,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=128,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        fused_quant_a_row_stride=a_row_stride,
        fused_quant_a_l_stride=a_l_stride,
        fused_quant_a_inv_rope=inv_rope,
        fused_quant_a_head_dim=head_dim,
        fused_quant_a_nope_dim=nope_dim,
        fused_quant_a_rope_dim=rope_dim,
        fused_quant_a_wide=a_wide,
        positions_dtype=positions_dtype,
        cos_sin_dtype=cos_sin_dtype,
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    placeholders = [16] * 9
    compiled = b12x_compile(
        launch,
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[0],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[1], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            positions_dtype, placeholders[2], cute.AddressSpace.gmem, assumed_align=8
        ),
        make_ptr(
            cos_sin_dtype, placeholders[3], cute.AddressSpace.gmem, assumed_align=4
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[4],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[5],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[6],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[7], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Float32, placeholders[8], cute.AddressSpace.gmem, assumed_align=16
        ),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.dense_fused_quant_a_grouped", 2, compile_key
        ),
    )

    def tensor_api(
        source: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        b: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        alpha: torch.Tensor,
        stream_int: Optional[int],
    ) -> torch.Tensor:
        source_ptr = source.data_ptr()
        compiled(
            make_ptr(
                cutlass.Float8E4M3FN,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16, source_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                positions_dtype,
                positions.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cos_sin_dtype,
                cos_sin.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                sfb.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16,
                out.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                alpha.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(source.shape[0]),
            int(cos_sin.numel()),
            cuda_stream_from_int_or_current(stream_int),
        )
        return out

    return attach_programs(tensor_api, compiled)


def dense_gemm_fused_quant_a_grouped(
    source: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    *,
    groups: int,
    out: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    head_dim: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
    expected_m: Optional[int] = None,
    sfb_k_replicated: bool = False,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    stream: object = None,
) -> torch.Tensor:
    """Small-M grouped BF16-A -> MXFP8 GEMM quantizing A in each CTA (WO-A).

    `source` is `[M, groups, K]` BF16 with contiguous trailing dims (rows may
    be strided); logical GEMM operands are per-group `[M, K] x [N, K]`. When
    `positions`/`cos_sin_cache` are given, the trailing `rope_dim` of every
    `head_dim` block is inverse-RoPE-rotated before quantization.
    """

    if source.dtype != torch.bfloat16 or source.ndim != 3:
        raise ValueError(
            "fused grouped MXFP8 quantization requires BF16 [M, groups, K]"
        )
    m = int(source.shape[0])
    k = int(source.shape[2])
    if int(source.shape[1]) != groups:
        raise ValueError(
            f"source groups {int(source.shape[1])} != weight groups {groups}"
        )
    if source.stride(2) != 1 or source.stride(1) != k:
        raise ValueError(
            f"fused grouped MXFP8 A needs contiguous [groups, K] rows, got strides {source.stride()}"
        )
    row_stride = int(source.stride(0))
    if m < 1 or m > 8 or k % 128 != 0 or row_stride % 8 != 0:
        raise ValueError(
            f"fused grouped MXFP8 quantization requires 1<=M<=8, K%128=0, row stride%8=0; "
            f"got M={m}, K={k}, row_stride={row_stride}"
        )
    if b.ndim != 3 or int(b.shape[1]) != k or int(b.shape[2]) != groups:
        raise ValueError(f"B must have shape [N,{k},{groups}], got {tuple(b.shape)}")
    n = int(b.shape[0])
    inv_rope = positions is not None or cos_sin_cache is not None
    if inv_rope:
        if positions is None or cos_sin_cache is None:
            raise ValueError("inverse-RoPE needs both positions and cos_sin_cache")
        if positions.shape != (m,):
            raise ValueError(
                f"positions must have shape {(m,)}, got {tuple(positions.shape)}"
            )
        if not positions.is_contiguous() or not cos_sin_cache.is_contiguous():
            raise ValueError("positions and cos_sin_cache must be contiguous")
        if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != rope_dim:
            raise ValueError(
                f"cos_sin_cache must have shape [max_pos, {rope_dim}], got {tuple(cos_sin_cache.shape)}"
            )
        if (
            head_dim <= 0
            or nope_dim + rope_dim != head_dim
            or head_dim % 32
            or nope_dim % 32
            or rope_dim % 32
            or k % head_dim
        ):
            raise ValueError(
                "fused inverse-RoPE requires 32-aligned head_dim = nope_dim + rope_dim "
                f"dividing K, got head={head_dim}, nope={nope_dim}, rope={rope_dim}, K={k}"
            )
        positions_dtype = _cutlass_positions_dtype(positions.dtype)
        cos_sin_dtype = _cutlass_cos_sin_dtype(cos_sin_cache.dtype)
    else:
        head_dim = nope_dim = rope_dim = 0
        positions = source
        cos_sin_cache = source
        positions_dtype = cutlass.Int64
        cos_sin_dtype = cutlass.BFloat16
    sm_count = get_num_sm(source.device)
    if mma_tiler_mn is None:
        plan = _select_default_dense_gemm_plan(
            m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m
        )
        if plan.swap_ab or plan.load_path != "tma":
            raise ValueError(
                "fused grouped MXFP8 quantization requires the unswapped TMA plan"
            )
        mma_tiler_mn = plan.mma_tiler_mn
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=groups,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        tile_k=128,
        expected_m=expected_m,
        generalize_mxfp8_split_k=True,
    )
    if policy.split_k_slices != 1:
        raise ValueError("fused grouped MXFP8 quantization does not support split-K")
    if out is None:
        out = _empty_dense_gemm_output(
            m, n, groups, dtype=torch.bfloat16, device=source.device
        )
    if out.shape != (m, n, groups) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(m, n, groups)}, got {out.dtype} {tuple(out.shape)}"
        )
    compiled = _get_compiled_dense_gemm_fused_quant_a_grouped(
        n,
        k,
        groups,
        policy,
        mma_tiler_mn,
        sm_count,
        bool(sfb_k_replicated),
        row_stride,
        k,
        bool(inv_rope),
        int(head_dim),
        int(nope_dim),
        int(rope_dim),
        (expected_m if expected_m is not None else m) == 1,
        positions_dtype,
        cos_sin_dtype,
    )
    return compiled(
        source,
        positions,
        cos_sin_cache,
        b,
        sfb,
        out,
        _cached_alpha_one(source.device),
        cuda_stream_to_int(stream),
    )


def _dense_gemm_target_occupancy(
    *,
    n: int,
    k: int,
    l: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    load_path: str,
    swap_ab: bool,
    b_tile_major: bool,
    is_mxfp6: bool = False,
) -> int:
    tile_m, tile_n = mma_tiler_mn
    n_tiles = ((n + tile_n - 1) // tile_n) * l
    if (
        # NOT is_mxfp6_ab_dtype(ab_dtype): the MX-FP6 path rewrites ab_dtype to
        # Float8E4M3FN byte-containers before compiling, so the operand dtype
        # cannot distinguish the two families here.
        is_mxfp6
        and c_dtype == cutlass.BFloat16
        and tile_k == 128
        and mma_tiler_mn == (16, 64)
        and cluster_shape_mn == (1, 1)
        and load_path == "tma"
        and not swap_ab
        and not b_tile_major
        and n_tiles > sm_count
    ):
        # Two resident CTAs avoid a partial tail wave when the decode grid is
        # larger than the SM count. Occupancy does not change MMA ordering.
        return 2
    return (
        2
        if ab_dtype == cutlass.Float8E4M3FN
        and c_dtype == cutlass.BFloat16
        and tile_k == 128
        and tile_m == 16
        and k <= 1024
        and cluster_shape_mn == (1, 1)
        and load_path == "tma"
        and not swap_ab
        and not b_tile_major
        and n_tiles >= 2 * sm_count
        else 1
    )


@program_cache
def _get_compiled_dense_gemm(
    n: int,
    k: int,
    l: int,
    c_l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    policy: _DenseGemmPolicy,
    sm_count: int,
    sm_version: str,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    b_tile_major: bool,
    quantize_c: bool = False,
    alpha_is_one: bool = False,
    direct_sfa_live16: bool = False,
    direct_m1_wo_a_inputs: bool = False,
    plain_fp8: bool = False,
    block_fp8: bool = False,
    target_occupancy_override: Optional[int] = None,
    weight_only: Optional[str] = None,
    mma_warps: Optional[int] = None,
    alpha_reciprocal: bool = False,
    input_k: Optional[int] = None,
    device_ordinal: Optional[int] = None,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
        quant_c_tensors: Optional[List[torch.Tensor]] = None,
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = [16 for _ in range(6)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
            ) = input_tensors
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                alpha_tensor_gpu.data_ptr(),
            )
        if quant_c_tensors is None:
            quant_c_values_ptr = 16
            quant_c_scale_rows_ptr = 16
            quant_c_scale_mma_ptr = 16
        else:
            (
                quant_c_values_gpu,
                quant_c_scale_rows_gpu,
                quant_c_scale_mma_gpu,
            ) = quant_c_tensors
            quant_c_values_ptr = quant_c_values_gpu.data_ptr()
            quant_c_scale_rows_ptr = quant_c_scale_rows_gpu.data_ptr()
            quant_c_scale_mma_ptr = quant_c_scale_mma_gpu.data_ptr()

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.Uint8 if weight_only else ab_dtype,
                b_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.Float8E4M3FN,
                quant_c_values_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                quant_c_scale_rows_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                quant_c_scale_mma_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
        ]

    launch = _DenseGemmLaunch(
        n=n,
        k=k,
        l=l,
        c_l=c_l,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        policy=policy,
        sm_count=sm_count,
        sm_version=sm_version,
        load_path=load_path,
        swap_ab=swap_ab,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        quantize_c=quantize_c,
        alpha_is_one=alpha_is_one,
        direct_sfa_live16=direct_sfa_live16,
        direct_m1_wo_a_inputs=direct_m1_wo_a_inputs,
        plain_fp8=plain_fp8,
        block_fp8=block_fp8,
        weight_only=weight_only,
        mma_warps=mma_warps,
        alpha_reciprocal=alpha_reciprocal,
        input_k=input_k,
        target_occupancy=(
            target_occupancy_override
            if target_occupancy_override is not None
            else _dense_gemm_target_occupancy(
                n=n,
                k=k,
                l=l,
                ab_dtype=ab_dtype,
                c_dtype=c_dtype,
                tile_k=tile_k,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sm_count=sm_count,
                load_path=load_path,
                swap_ab=swap_ab,
                b_tile_major=b_tile_major,
            )
        ),
    )
    if weight_only is not None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("A16 dense GEMM must be prewarmed before CUDA graph capture")
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = b12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense", 4, compile_key),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        stream_int: Optional[int] = None,
        quant_c_values_gpu: Optional[torch.Tensor] = None,
        quant_c_scale_rows_gpu: Optional[torch.Tensor] = None,
        quant_c_scale_mma_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a_tensor_gpu.shape[0]
        if c_tensor_gpu is None:
            c_tensor_gpu = torch.empty(
                (m, n, c_l),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = _cached_alpha_one(a_tensor_gpu.device)
        quant_c_tensors = None
        if quantize_c:
            if (
                quant_c_values_gpu is None
                or quant_c_scale_rows_gpu is None
                or quant_c_scale_mma_gpu is None
            ):
                raise ValueError("quantized C output tensors are required")
            quant_c_tensors = [
                quant_c_values_gpu,
                quant_c_scale_rows_gpu,
                quant_c_scale_mma_gpu,
            ]

        nonlocal compiled_kernel
        compiled_kernel(
            *_make_runtime_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    alpha_tensor_gpu,
                ],
                quant_c_tensors,
            ),
            m,
            cuda_stream_from_int_or_current(stream_int),
        )
        return c_tensor_gpu

    return attach_programs(tensor_api, compiled_kernel)
