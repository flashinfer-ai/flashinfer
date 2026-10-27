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

"""Dense GEMM tensor and launch preparation."""

from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from b12x._lib.utils.device import cuda_stream_to_int
import cutlass
from dataclasses import dataclass
from b12x._lib.utils.dtypes import get_cutlass_dtype
from b12x._lib.utils.device import get_num_sm
from b12x._lib.utils.fp6_geometry import mxfp6_logical_k_from_packed_bytes
from b12x._lib.utils.fp6_geometry import mxfp6_tile_k
from b12x._lib.program_cache import program_cache
import torch
import triton
from b12x._lib.dense_gemm.buffers import _cached_alpha_one, _empty_dense_gemm_output
from b12x._lib.dense_gemm.compile import (
    _cutlass_cos_sin_dtype,
    _cutlass_positions_dtype,
    _get_compiled_dense_gemm,
    _get_compiled_dense_gemm_fused_quant_a,
    _get_compiled_dense_gemm_fused_quant_a_grouped,
    _get_compiled_dense_gemm_mxfp6,
)
from b12x._lib.dense_gemm.config import (
    _B12X_DENSE_SPLITK_TURBO,
    _DENSE_FUSED_QUANT,
    _DENSE_LOAD_PATHS,
    _DenseGemmPolicy,
    _dense_gemm_policy_for,
    _use_direct_m1_wo_a_inputs,
    _use_direct_sfa_live16,
)
from b12x._lib.dense_gemm.execution import _dense_split_storage, _expand_packed_mxfp6_ab
from b12x._lib.dense_gemm.policy import (
    _select_default_dense_gemm_plan,
    _select_fp4_tile_k,
    _select_mxfp8_tile_k,
    _validate_mxfp8_bk64_plan,
)
from b12x._lib.dense_gemm.reductions import _reduce_split_k2_bf16_kernel


@dataclass(frozen=True)
class _DenseLowering:
    m: int
    n: int
    k: int
    l: int
    a_storage_k: int
    b_storage_k: int
    ab_dtype: str
    sf_dtype: str
    c_dtype: str
    alpha_dtype: str
    sf_vec_size: int
    sm_count: int
    mma_k: int
    tile_k: int
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    policy: _DenseGemmPolicy
    load_path: str
    swap_ab: bool
    expected_m: int | None
    kernel_c_l: int
    alpha_is_one: bool
    is_mxfp6: bool
    mxfp6_fmt_a: str | None
    mxfp6_fmt_b: str | None
    a_preexpanded: bool
    b_preexpanded: bool
    b_packed: bool
    plain_fp8: bool
    block_fp8: bool
    sfb_k_reuse: bool
    b_tile_major: bool
    quantize_c: bool
    fused_quant: bool
    row_scale: bool
    output_provided: bool
    target_occupancy_override: int | None
    direct_sfa_live16: bool
    direct_m1_wo_a_inputs: bool

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["policy"] = _DenseGemmPolicy(**values["policy"])
        for name in ("mma_tiler_mn", "cluster_shape_mn"):
            values[name] = tuple(values[name])
        return cls(**values)


def _lower_dense_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    sm_count: Optional[int] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    alpha: Optional[torch.Tensor] = None,
    alpha_dtype: Optional[str] = None,
    expected_m: Optional[int] = None,
    load_path: Optional[Literal["tma", "cpasync"]] = None,
    swap_ab: Optional[bool] = None,
    sfb_k_replicated: bool = False,
    rhs_values_tiled: Optional[torch.Tensor] = None,
    _quantized_c: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    stream: object = None,
    a_preexpanded: bool = False,
    b_preexpanded: bool = False,
    b_packed: bool = False,
    a_fmt: Optional[str] = None,
    b_fmt: Optional[str] = None,
    x_bf16: Optional[torch.Tensor] = None,
    w_gscale: Optional[torch.Tensor] = None,
    plain_fp8: bool = False,
    row_scale: Optional[torch.Tensor] = None,
    block_fp8: bool = False,
    _tile_k_override: Optional[int] = None,
    _split_k_slices_override: Optional[int] = None,
    _large_m_unroll_override: Optional[bool] = None,
    _target_occupancy_override: Optional[int] = None,
    _split_k_workspace: Optional[torch.Tensor] = None,
) -> _DenseLowering:
    """Lower the existing joint host rules without allocating or launching."""
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    if load_path is not None and load_path not in _DENSE_LOAD_PATHS:
        raise ValueError(
            f"dense_gemm load_path must be one of {_DENSE_LOAD_PATHS}, got {load_path!r}"
        )
    if b_packed and b_preexpanded:
        raise ValueError("b_packed and b_preexpanded are mutually exclusive")

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    mxfp6_fmt_a: Optional[str] = None
    mxfp6_fmt_b: Optional[str] = None
    if ab_dtype == "float4_e2m1fn":
        is_mxfp8 = False
        is_mxfp6 = False
        k *= 2
        mma_k = 64
        tile_k = 128
    elif ab_dtype == "float8_e4m3fn":
        is_mxfp8 = True
        is_mxfp6 = False
        mma_k = 32
        tile_k = (
            128 if block_fp8 else _select_mxfp8_tile_k(m, n, k, expected_m, sm_count)
        )
    elif ab_dtype in ("float6_e3m2fn", "float6_e2m3fn"):
        is_mxfp8 = False
        is_mxfp6 = True
        if sf_vec_size != 32:
            raise ValueError("MX-FP6 dense_gemm requires sf_vec_size=32")
        if sf_dtype != "float8_e8m0fnu":
            raise ValueError("MX-FP6 dense_gemm requires sf_dtype='float8_e8m0fnu'")
        if not a_preexpanded:
            k = mxfp6_logical_k_from_packed_bytes(k)
        mma_k = 32
        tile_k = mxfp6_tile_k(sf_vec_size)
        weight_fmt = "e3m2" if ab_dtype == "float6_e3m2fn" else "e2m3"
        mxfp6_fmt_b = b_fmt if b_fmt is not None else weight_fmt
        mxfp6_fmt_a = a_fmt if a_fmt is not None else weight_fmt
        for name, fmt in (("a_fmt", mxfp6_fmt_a), ("b_fmt", mxfp6_fmt_b)):
            if fmt not in ("e2m3", "e3m2", "e4m3"):
                raise ValueError(f"unsupported {name}={fmt!r}")
    else:
        raise TypeError(f"dense_gemm unsupported ab_dtype: {ab_dtype}")
    if _tile_k_override is not None:
        if ab_dtype == "float4_e2m1fn":
            valid_tile_k = (128, 256, 512)
            format_name = "NVFP4"
        elif (
            ab_dtype == "float8_e4m3fn"
            and not block_fp8
            and not plain_fp8
            and sf_dtype == "float8_e8m0fnu"
            and sf_vec_size == 32
        ):
            valid_tile_k = (64, 128)
            format_name = "MXFP8"
        else:
            raise ValueError(
                "_tile_k_override is restricted to NVFP4 or MXFP8 autotuning"
            )
        if _tile_k_override not in valid_tile_k or k % _tile_k_override:
            if format_name == "NVFP4":
                requirement = "128, 256, or 512"
            else:
                requirement = "one of (64, 128)"
            raise ValueError(
                f"{format_name} _tile_k_override must be {requirement} and divide "
                f"logical K={k}, got {_tile_k_override}"
            )
        tile_k = _tile_k_override
    if _target_occupancy_override is not None:
        if ab_dtype != "float4_e2m1fn":
            raise ValueError(
                "_target_occupancy_override is restricted to NVFP4 autotuning"
            )
        if _target_occupancy_override not in (1, 2, 3, 4):
            raise ValueError(
                "NVFP4 _target_occupancy_override must be 1, 2, 3, or 4, got "
                f"{_target_occupancy_override}"
            )
        if out is None:
            raise ValueError(
                "NVFP4 _target_occupancy_override requires a caller-owned output"
            )
    if block_fp8:
        plain_fp8 = True
        expected_sfa_shape = (m, k // 128)
        expected_sfb_shape = (n // 128, k // 128)
        if (
            sf_dtype != "float32"
            or sf_vec_size != 128
            or l != 1
            or n % 128 != 0
            or k % 128 != 0
            or sfa_torch.shape != expected_sfa_shape
            or sfb_torch.shape != expected_sfb_shape
            or sfa_torch.dtype != torch.float32
            or sfb_torch.dtype != torch.float32
            or not sfa_torch.is_contiguous()
            or not sfb_torch.is_contiguous()
        ):
            raise ValueError(
                "block_fp8 requires L=1, N/K divisible by 128, "
                "sf_dtype='float32', sf_vec_size=128, and contiguous FP32 "
                f"scales shaped {expected_sfa_shape} and {expected_sfb_shape}"
            )
        if (
            load_path not in (None, "tma")
            or swap_ab not in (None, False)
            or rhs_values_tiled is not None
            or _quantized_c is not None
            or sfb_k_replicated
        ):
            raise ValueError(
                "block_fp8 currently requires unswapped TMA operands without "
                "tiled RHS, quantized output, or replicated weight scales"
            )
        load_path = "tma"
        swap_ab = False
    if plain_fp8 and not is_mxfp8:
        raise ValueError("plain_fp8 requires ab_dtype='float8_e4m3fn'")
    if plain_fp8 and (
        rhs_values_tiled is not None or _quantized_c is not None or sfb_k_replicated
    ):
        raise ValueError(
            "plain_fp8 does not support tiled/quantized output or replicated scales"
        )
    if b_packed:
        if mxfp6_fmt_b is None:
            raise ValueError("b_packed requires an MX-FP6 ab_dtype")
        if b_torch.shape[1] * 4 != k * 3:
            raise ValueError(
                f"b_packed expects (N, 3K/4, L); got packed K bytes "
                f"{b_torch.shape[1]} for logical K {k}"
            )
    if not is_mxfp6 and (
        a_preexpanded
        or b_preexpanded
        or a_fmt is not None
        or b_fmt is not None
        or x_bf16 is not None
        or w_gscale is not None
    ):
        raise ValueError(
            "a_preexpanded/b_preexpanded/a_fmt/b_fmt/x_bf16/w_gscale are only "
            f"supported with an MX-FP6 ab_dtype, got ab_dtype={ab_dtype!r}"
        )
    if (x_bf16 is None) != (w_gscale is None):
        raise ValueError("x_bf16 and w_gscale must be provided together")
    if x_bf16 is not None and w_gscale is not None:
        if not is_mxfp6:
            raise ValueError("fused quantization inputs require an MX-FP6 ab_dtype")
        if (
            x_bf16.shape != (m, k)
            or x_bf16.dtype != torch.bfloat16
            or not x_bf16.is_contiguous()
            or x_bf16.device != a_torch.device
            or x_bf16.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "x_bf16 must be a contiguous, 16-byte-aligned BF16 tensor "
                f"with shape {(m, k)} on {a_torch.device}; got shape "
                f"{tuple(x_bf16.shape)}, dtype {x_bf16.dtype}, device "
                f"{x_bf16.device}, data_ptr alignment {x_bf16.data_ptr() % 16}"
            )
        if (
            w_gscale.shape != (1,)
            or w_gscale.dtype != torch.float32
            or not w_gscale.is_contiguous()
            or w_gscale.device != a_torch.device
            or w_gscale.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "w_gscale must be a contiguous, 16-byte-aligned float32 tensor "
                f"with shape (1,) on {a_torch.device}; got shape "
                f"{tuple(w_gscale.shape)}, dtype {w_gscale.dtype}, device "
                f"{w_gscale.device}, data_ptr alignment {w_gscale.data_ptr() % 16}"
            )

    ab_cutlass_dtype = get_cutlass_dtype(ab_dtype)
    if mxfp6_fmt_a is not None:
        # Stage 1: carry MX codes in Float8E4M3FN byte-containers so the kernel
        # uses cutlass's native 8-bit smem/TMA/ldmatrix path (cutlass cannot build
        # a 6-bit smem layout). Expand the 3:4-packed inputs to one code per byte
        # at this load boundary; the on-disk/wire format stays packed. E4M3
        # activations are already one-byte-per-code (a_preexpanded required).
        ab_cutlass_dtype = cutlass.Float8E4M3FN
        if not a_preexpanded:
            if mxfp6_fmt_a == "e4m3":
                raise ValueError("e4m3 activations require a_preexpanded=True")
    c_cutlass_dtype = get_cutlass_dtype(c_dtype)
    c_row_stride_bytes = n * c_cutlass_dtype.width // 8
    output_requires_swapped_store = (m > 1 or l > 1) and c_row_stride_bytes % 16 != 0
    use_default_mma_tiler = mma_tiler_mn is None
    use_default_output_storage = mma_tiler_mn is None and swap_ab is None
    if mma_tiler_mn is None or load_path is None or swap_ab is None:
        default_plan = _select_default_dense_gemm_plan(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=is_mxfp8,
            is_mxfp6=is_mxfp6,
            block_fp8=block_fp8,
            expected_m=expected_m,
            select_swapped_output_storage=(
                use_default_output_storage
                and l == 1
                and (n < 64 or output_requires_swapped_store)
            ),
        )
        if mma_tiler_mn is None:
            mma_tiler_mn = default_plan.mma_tiler_mn
        if load_path is None:
            load_path = default_plan.load_path
        if swap_ab is None:
            if use_default_mma_tiler:
                swap_ab = default_plan.swap_ab
            else:
                swap_ab = default_plan.swap_ab if mma_tiler_mn[1] < 64 else False
    assert load_path is not None
    assert swap_ab is not None
    if ab_dtype == "float4_e2m1fn" and _tile_k_override is None:
        tile_k = _select_fp4_tile_k(
            m,
            n,
            k,
            expected_m,
            sm_count,
            mma_tiler_mn,
        )
    if l > 1 and swap_ab:
        raise ValueError(
            "swapped dense_gemm output storage supports L=1 only; pad N for "
            f"grouped output, got L={l}, N={n}"
        )
    if output_requires_swapped_store and not swap_ab:
        remedy = (
            "pad N; swapped output storage is unsupported when L > 1"
            if l > 1
            else "use a supported swapped plan or pad N"
        )
        raise ValueError(
            "the unswapped dense_gemm TMA epilogue requires a 16-byte-aligned "
            f"C row stride, but N={n} and c_dtype={c_dtype!r} produce "
            f"{c_row_stride_bytes} bytes; {remedy}"
        )
    if is_mxfp8 and _tile_k_override is None and mma_tiler_mn[0] != 128:
        tile_k = 128
    if is_mxfp8 and swap_ab:
        # BK64 packed-scale staging requires the weight operand to remain in
        # the unswapped 128-row slot. Swapped storage therefore uses BK128.
        tile_k = 128
    if is_mxfp6:
        # Only the unswapped single-slice TMA mainloop is wired for the FP6
        # byte-container path; fail loudly instead of silently miscomputing.
        if swap_ab:
            raise ValueError("MX-FP6 dense_gemm does not support swap_ab")
        if load_path != "tma":
            raise ValueError(
                f"MX-FP6 dense_gemm only supports load_path='tma', got {load_path!r}"
            )
        if _quantized_c is not None:
            raise ValueError("MX-FP6 dense_gemm does not support quantized C output")
    b_launch_torch = b_torch
    if rhs_values_tiled is not None:
        tile_n = 0
        supported_plan = False
        if (n, k, l) == (1024, 4096, 4):
            tile_n = 64
            supported_plan = mma_tiler_mn in ((16, 64), (32, 64), (64, 64))
        elif (n, k, l) == (4096, 4096, 1):
            tile_n = 128
            supported_plan = mma_tiler_mn in (
                (16, 128),
                (32, 64),
                (32, 128),
            )
        if not is_mxfp8 or not supported_plan or swap_ab or load_path != "tma":
            raise ValueError(
                "tile-major MXFP8 RHS is restricted to production WO-A/WO-B "
                "Ntile/BK128 TMA plans"
            )
        expected_tiled_shape = (l, n // tile_n, k // 128, tile_n, 128)
        if (
            rhs_values_tiled.shape != expected_tiled_shape
            or rhs_values_tiled.dtype != b_torch.dtype
            or rhs_values_tiled.device != b_torch.device
            or not rhs_values_tiled.is_contiguous()
        ):
            raise ValueError(
                "tile-major MXFP8 RHS must be contiguous with shape "
                f"{expected_tiled_shape}, dtype {b_torch.dtype}, and device "
                f"{b_torch.device}; got shape={tuple(rhs_values_tiled.shape)}, "
                f"dtype={rhs_values_tiled.dtype}, device={rhs_values_tiled.device}"
            )
        b_launch_torch = rhs_values_tiled
    if is_mxfp8:
        _validate_mxfp8_bk64_plan(tile_k, mma_tiler_mn, swap_ab)
    # k-reuse relies on SFB being the 128x128-block weight operand; with
    # swap_ab the smem B slot holds activations, so force it off there.
    sfb_k_reuse = bool(sfb_k_replicated) and not swap_ab and is_mxfp8
    if alpha_dtype is None:
        alpha_dtype = "float32" if alpha is None else str(alpha.dtype).split(".")[-1]
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=l,
        ab_dtype=ab_cutlass_dtype,
        c_dtype=c_cutlass_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        tile_k=tile_k,
        expected_m=expected_m,
        generalize_mxfp8_split_k=(is_mxfp8 and not block_fp8 and not plain_fp8),
        generalize_block_fp8_split_k=block_fp8,
    )
    if _split_k_slices_override is not None:
        mxfp8_autotune = (
            is_mxfp8
            and not block_fp8
            and not plain_fp8
            and sf_vec_size == 32
            and sf_dtype == "float8_e8m0fnu"
        )
        block_fp8_autotune = (
            is_mxfp8 and block_fp8 and sf_vec_size == 128 and sf_dtype == "float32"
        )
        if not (mxfp8_autotune or block_fp8_autotune):
            raise ValueError(
                "_split_k_slices_override is restricted to MXFP8 or block-FP8 "
                "autotuning"
            )
        if _split_k_slices_override not in (1, 2, 4):
            raise ValueError(
                "FP8 _split_k_slices_override must be 1, 2, or 4, got "
                f"{_split_k_slices_override}"
            )
        if _split_k_slices_override > 1:
            if m > 8 or m > mma_tiler_mn[0] or l != 1 or swap_ab:
                raise ValueError(
                    "split-K FP8 autotuning requires M<=8 within one M tile, "
                    f"L=1, and an unswapped plan; got M={m}, L={l}, "
                    f"tile={mma_tiler_mn}, swap_ab={swap_ab}"
                )
            if k % (tile_k * _split_k_slices_override):
                raise ValueError(
                    "split-K FP8 autotuning requires the staged K-tile count "
                    f"to divide evenly across slices; got K={k}, BK={tile_k}, "
                    f"slices={_split_k_slices_override}"
                )
            if _split_k_slices_override > 2 and not _B12X_DENSE_SPLITK_TURBO:
                raise ValueError(
                    "four-way split-K requires the atomic-BF16 reduction path"
                )
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=_split_k_slices_override,
            split_k_atomic_bf16=(
                _split_k_slices_override > 1 and _B12X_DENSE_SPLITK_TURBO
            ),
            large_m_unroll=policy.large_m_unroll,
        )
    if _large_m_unroll_override is not None:
        if not is_mxfp8 or is_mxfp6 or l != 1:
            raise ValueError(
                "_large_m_unroll_override is restricted to FP8 autotuning with L=1"
            )
        if not isinstance(_large_m_unroll_override, bool):
            raise ValueError("_large_m_unroll_override must be a bool")
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=_large_m_unroll_override,
        )
    split_k_slices = policy.split_k_slices
    if swap_ab and split_k_slices != 1:
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=policy.large_m_unroll,
        )
        split_k_slices = 1
    if is_mxfp6 and (policy.split_k_slices != 1 or policy.large_m_unroll):
        # The policy helper sees the FP8 byte-container dtype and may pick
        # MXFP8 tactics that are not wired for MX-FP6.
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=False,
        )
        split_k_slices = 1
    split_k_output = split_k_slices > 1
    split_k_atomic_bf16 = split_k_output and policy.split_k_atomic_bf16
    if split_k_atomic_bf16:
        kernel_c_l = l
    elif split_k_output:
        kernel_c_l = split_k_slices
    else:
        kernel_c_l = l
    alpha_is_one = alpha is None

    common = dict(
        m=m,
        n=n,
        k=k,
        l=l,
        sf_vec_size=sf_vec_size,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        load_path=load_path,
        swap_ab=swap_ab,
        b_tile_major=rhs_values_tiled is not None,
        sfb_k_reuse=sfb_k_reuse,
        is_mxfp8=is_mxfp8,
    )
    return _DenseLowering(
        m=m,
        n=n,
        k=k,
        l=l,
        a_storage_k=lhs[0].shape[1],
        b_storage_k=rhs[0].shape[1],
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size,
        sm_count=sm_count,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=tuple(mma_tiler_mn),
        cluster_shape_mn=tuple(cluster_shape_mn),
        policy=policy,
        load_path=load_path,
        swap_ab=swap_ab,
        expected_m=expected_m,
        kernel_c_l=kernel_c_l,
        alpha_is_one=alpha_is_one,
        is_mxfp6=is_mxfp6,
        mxfp6_fmt_a=mxfp6_fmt_a,
        mxfp6_fmt_b=mxfp6_fmt_b,
        a_preexpanded=a_preexpanded,
        b_preexpanded=b_preexpanded,
        b_packed=b_packed,
        plain_fp8=plain_fp8,
        block_fp8=block_fp8,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=rhs_values_tiled is not None,
        quantize_c=_quantized_c is not None,
        fused_quant=_DENSE_FUSED_QUANT and x_bf16 is not None,
        row_scale=row_scale is not None,
        output_provided=out is not None,
        target_occupancy_override=_target_occupancy_override,
        direct_sfa_live16=_use_direct_sfa_live16(**common, alpha_is_one=alpha_is_one),
        direct_m1_wo_a_inputs=_use_direct_m1_wo_a_inputs(**common),
    )


@program_cache(scope="preparation")
def _compile_dense_lowering(payload, device_ordinal):
    """Compile exact production launchers from metadata, without tensor storage."""
    p = _DenseLowering.from_dict(payload)
    split = p.policy.split_k_slices > 1
    atomic = split and p.policy.split_k_atomic_bf16
    output_type = "float32" if split and not atomic else p.c_dtype
    common = dict(
        n=p.n,
        k=p.k,
        l=p.l,
        c_l=p.kernel_c_l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN if p.is_mxfp6 else get_cutlass_dtype(p.ab_dtype),
        sf_dtype=get_cutlass_dtype(p.sf_dtype),
        c_dtype=get_cutlass_dtype(output_type),
        alpha_dtype=get_cutlass_dtype(p.alpha_dtype),
        sf_vec_size=p.sf_vec_size,
        mma_k=p.mma_k,
        tile_k=p.tile_k,
        mma_tiler_mn=p.mma_tiler_mn,
        cluster_shape_mn=p.cluster_shape_mn,
        policy=p.policy,
        sm_count=p.sm_count,
        sm_version="sm_120",
        alpha_is_one=p.alpha_is_one,
    )
    with torch.cuda.device(device_ordinal):
        if p.is_mxfp6:
            gemm = _get_compiled_dense_gemm_mxfp6(
                **common,
                mxfp6_fmt_a=p.mxfp6_fmt_a,
                mxfp6_fmt_b=p.mxfp6_fmt_b,
                b_packed=p.b_packed,
                a_preexpanded=p.a_preexpanded,
                b_preexpanded=p.b_preexpanded,
                row_scale=p.row_scale,
                fused_quant=p.fused_quant,
            )
        else:
            gemm = _get_compiled_dense_gemm(
                **common,
                load_path=p.load_path,
                swap_ab=p.swap_ab,
                sfb_k_reuse=False if p.plain_fp8 else p.sfb_k_reuse,
                b_tile_major=False if p.plain_fp8 else p.b_tile_major,
                quantize_c=p.quantize_c,
                plain_fp8=p.plain_fp8,
                block_fp8=p.block_fp8,
                direct_sfa_live16=False if p.plain_fp8 else p.direct_sfa_live16,
                direct_m1_wo_a_inputs=False
                if p.plain_fp8 or p.quantize_c
                else p.direct_m1_wo_a_inputs,
                target_occupancy_override=p.target_occupancy_override,
            )
        programs = {"gemm": gemm}
        if split and not atomic:
            if p.policy.split_k_slices != 2:
                raise ValueError("non-atomic dense reduction requires two slices")
            programs["reduce"] = _reduce_split_k2_bf16_kernel.warmup(
                torch.float32,
                torch.bfloat16,
                p.m * p.n,
                BLOCK=1024,
                grid=(triton.cdiv(p.m * p.n, 1024),),
            )
        return programs


def _require_dense_value(tensor, shape, dtype, device, name):
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} differs from prepared dense shape, dtype, or device")
    # Pointer-based launchers use physical [L,M,K] / [L,M,N], ignoring strides
    # on singleton axes. Reject padded rows rather than silently misaddress them.
    strides = (shape[1], 1, shape[0] * shape[1])
    if any(
        size > 1 and actual != expected
        for size, actual, expected in zip(shape, tensor.stride(), strides)
    ):
        raise ValueError(f"{name} must retain the prepared group-major layout")
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


@dataclass(frozen=True)
class _DenseExecutionState:
    lowering: _DenseLowering
    device: torch.device
    gemm: Callable
    reduction: object | None
    alpha_one: torch.Tensor | None

    def run(
        self,
        lhs,
        rhs,
        out=None,
        *,
        alpha=None,
        stream=None,
        rhs_values_tiled=None,
        quantized_c=None,
        x_bf16=None,
        w_gscale=None,
        row_scale=None,
        split_k_workspace=None,
    ):
        p = self.lowering
        a, sfa = lhs
        b, sfb = rhs
        m = int(a.shape[0])
        if m > p.m:
            raise ValueError("dense execution exceeds its planned capacity")
        value_dtype = (
            torch.uint8 if p.ab_dtype != "float8_e4m3fn" else torch.float8_e4m3fn
        )
        a_dtype = torch.float8_e4m3fn if p.mxfp6_fmt_a == "e4m3" else value_dtype
        _require_dense_value(a, (m, p.a_storage_k, p.l), a_dtype, self.device, "A")
        _require_dense_value(
            b, (p.n, p.b_storage_k, p.l), value_dtype, self.device, "B"
        )
        for name, scale in (("SFA", sfa), ("SFB", sfb)):
            if scale.device != self.device or scale.dtype != getattr(torch, p.sf_dtype):
                raise ValueError(f"{name} differs from prepared dense dtype/device")
        if (alpha is None) != p.alpha_is_one:
            raise ValueError("dense scalar-alpha presence differs from preparation")
        if alpha is not None and (
            alpha.device != self.device
            or alpha.dtype != getattr(torch, p.alpha_dtype)
            or alpha.numel() != 1
            or not alpha.is_contiguous()
        ):
            raise ValueError("dense alpha must retain its declared scalar dtype/device")
        if (rhs_values_tiled is not None) != p.b_tile_major:
            raise ValueError("dense tiled weight representation changed")
        if (quantized_c is not None) != p.quantize_c or (
            row_scale is not None
        ) != p.row_scale:
            raise ValueError("dense auxiliary operand presence changed")
        if out is not None:
            _require_dense_value(
                out, (m, p.n, p.l), getattr(torch, p.c_dtype), self.device, "C"
            )
        if m == 0:
            return (
                out
                if out is not None
                else _empty_dense_gemm_output(
                    0,
                    p.n,
                    p.l,
                    dtype=getattr(torch, p.c_dtype),
                    device=self.device,
                )
            )
        if p.is_mxfp6:
            if not p.a_preexpanded:
                a = _expand_packed_mxfp6_ab(a, p.k)
            if not (p.b_preexpanded or p.b_packed):
                b = _expand_packed_mxfp6_ab(b, p.k)
        if out is None:
            out = _empty_dense_gemm_output(
                m, p.n, p.l, dtype=getattr(torch, p.c_dtype), device=self.device
            )
        alpha_value = self.alpha_one if alpha is None else alpha
        stream_int = cuda_stream_to_int(stream)
        if p.is_mxfp6:
            if row_scale is not None and (
                row_scale.shape != (m,)
                or row_scale.dtype != getattr(torch, p.c_dtype)
                or row_scale.device != self.device
                or not row_scale.is_contiguous()
                or row_scale.data_ptr() % 16
            ):
                raise ValueError("dense FP6 row scale differs from its planned layout")
            return self.gemm(
                a_tensor_gpu=a,
                b_tensor_gpu=b,
                sfa_tensor_gpu=sfa,
                sfb_tensor_gpu=sfb,
                c_tensor_gpu=out,
                alpha_tensor_gpu=alpha_value,
                stream_int=stream_int,
                x_bf16_tensor_gpu=x_bf16,
                w_gscale_tensor_gpu=w_gscale,
                row_scale_tensor_gpu=row_scale,
            )
        b_launch = b if rhs_values_tiled is None else rhs_values_tiled
        if p.quantize_c:
            values, rows, mma = quantized_c
            return self.gemm(
                a_tensor_gpu=a,
                b_tensor_gpu=b_launch,
                sfa_tensor_gpu=sfa,
                sfb_tensor_gpu=sfb,
                c_tensor_gpu=out,
                alpha_tensor_gpu=alpha_value,
                stream_int=stream_int,
                quant_c_values_gpu=values,
                quant_c_scale_rows_gpu=rows,
                quant_c_scale_mma_gpu=mma,
            )
        split = p.policy.split_k_slices > 1
        atomic = split and p.policy.split_k_atomic_bf16
        temporary = None
        if atomic:
            out.zero_()
        elif split:
            temporary = _dense_split_storage(
                split_k_workspace, p.policy.split_k_slices, m, p.n, self.device
            )
        self.gemm(
            a_tensor_gpu=a,
            b_tensor_gpu=b_launch,
            sfa_tensor_gpu=sfa,
            sfb_tensor_gpu=sfb,
            c_tensor_gpu=out if temporary is None else temporary.permute(1, 2, 0),
            alpha_tensor_gpu=alpha_value,
            stream_int=stream_int,
        )
        if temporary is not None:
            self.reduction[(triton.cdiv(m * p.n, 1024), 1, 1)](
                temporary, out, m * p.n, 1024
            )
        return out


def _materialize_dense(lowering, device):
    programs = _compile_dense_lowering(lowering.to_dict(), device.index)
    return _DenseExecutionState(
        lowering,
        device,
        programs["gemm"],
        programs.get("reduce"),
        _cached_alpha_one(device) if lowering.alpha_is_one else None,
    )


@dataclass(frozen=True)
class _DenseFusedQuantLowering:
    m: int
    n: int
    k: int
    groups: int
    grouped: bool
    source_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    b_dtype: str
    sfb_dtype: str
    expected_m: int | None
    sm_count: int
    mma_tiler_mn: tuple[int, int]
    policy: _DenseGemmPolicy
    sfb_k_replicated: bool
    b_tile_major: bool
    a_inner_span: int
    inv_rope: bool
    head_dim: int
    nope_dim: int
    rope_dim: int
    positions_dtype: str
    cos_sin_dtype: str
    output_provided: bool
    atomic_output_precleared: bool

    @property
    def workspace_nbytes(self):
        if self.policy.split_k_slices > 1 and not self.policy.split_k_atomic_bf16:
            return self.policy.split_k_slices * self.m * self.n * 4
        return 0

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["policy"] = _DenseGemmPolicy(**values["policy"])
        for name in ("source_shape", "source_strides", "mma_tiler_mn"):
            values[name] = tuple(values[name])
        return cls(**values)


def _lower_dense_gemm_fused_quant_a(
    source,
    b,
    sfb,
    *,
    sm_count,
    groups=None,
    out=None,
    positions=None,
    cos_sin_cache=None,
    head_dim=0,
    nope_dim=0,
    rope_dim=0,
    expected_m=None,
    sfb_k_replicated=False,
    rhs_values_tiled=None,
    a_inner_span=0,
    mma_tiler_mn=None,
    _atomic_output_precleared=False,
) -> _DenseFusedQuantLowering:
    """Lower the existing fused-quant branches without allocation or compilation."""
    grouped = groups is not None
    groups = 1 if groups is None else int(groups)
    a_inner_span = int(a_inner_span)
    if source.dtype != torch.bfloat16:
        raise ValueError("fused MXFP8 activation quantization requires BF16 A")
    if grouped:
        if a_inner_span or rhs_values_tiled is not None or _atomic_output_precleared:
            raise ValueError(
                "grouped fused quantization does not use WO-B storage options"
            )
        if source.ndim != 3 or int(source.shape[1]) != groups:
            raise ValueError("grouped fused quantization requires [M,groups,K]")
        m, k = int(source.shape[0]), int(source.shape[2])
        if source.stride(2) != 1 or source.stride(1) != k or source.stride(0) % 8:
            raise ValueError(
                "grouped fused A requires contiguous trailing dimensions and an aligned row stride"
            )
    elif not a_inner_span:
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError(
                "fused MXFP8 activation quantization requires contiguous BF16 [M,K]"
            )
        m, k = map(int, source.shape)
    else:
        if (
            a_inner_span % 32
            or source.ndim != 3
            or int(source.shape[1]) != a_inner_span
        ):
            raise ValueError(
                "L-blocked fused A requires an aligned [M,span,K/span] view"
            )
        m, k = int(source.shape[0]), a_inner_span * int(source.shape[2])
        if source.stride() != (a_inner_span, 1, m * a_inner_span):
            raise ValueError(
                "L-blocked fused A must retain the group-major source layout"
            )
    if not 1 <= m <= 8 or k % 128:
        raise ValueError(
            "fused MXFP8 activation quantization requires M1..8 and K divisible by128"
        )
    if b.ndim != 3 or tuple(b.shape[1:]) != (k, groups):
        raise ValueError(f"fused B must have shape [N,{k},{groups}]")
    n = int(b.shape[0])
    inv_rope = positions is not None or cos_sin_cache is not None
    if inv_rope:
        if not grouped or positions is None or cos_sin_cache is None:
            raise ValueError(
                "grouped inverse-RoPE requires positions and cos_sin_cache"
            )
        if (
            positions.shape != (m,)
            or not positions.is_contiguous()
            or not cos_sin_cache.is_contiguous()
        ):
            raise ValueError(
                "inverse-RoPE positions/cache must retain contiguous planned geometry"
            )
        if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != rope_dim:
            raise ValueError("inverse-RoPE cache width differs from rope_dim")
        if (
            head_dim <= 0
            or nope_dim + rope_dim != head_dim
            or head_dim % 32
            or nope_dim % 32
            or rope_dim % 32
            or k % head_dim
        ):
            raise ValueError(
                "inverse-RoPE head/nope/rope dimensions must preserve the aligned source contract"
            )
        _cutlass_positions_dtype(positions.dtype)
        _cutlass_cos_sin_dtype(cos_sin_cache.dtype)
        positions_dtype = str(positions.dtype).removeprefix("torch.")
        cos_sin_dtype = str(cos_sin_cache.dtype).removeprefix("torch.")
    else:
        head_dim = nope_dim = rope_dim = 0
        positions_dtype, cos_sin_dtype = "int64", "bfloat16"
    if mma_tiler_mn is None:
        selected = _select_default_dense_gemm_plan(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=True,
            expected_m=expected_m,
        )
        if selected.swap_ab or selected.load_path != "tma":
            raise ValueError("fused quantization requires the unswapped TMA plan")
        mma_tiler_mn = selected.mma_tiler_mn
    mma_tiler_mn = tuple(mma_tiler_mn)
    if rhs_values_tiled is not None:
        if (n, k) != (4096, 4096) or mma_tiler_mn not in ((16, 64), (16, 128)):
            raise ValueError("tile-major fused RHS requires the existing WO-B4096 plan")
        if (
            rhs_values_tiled.shape != (1, 32, 32, 128, 128)
            or rhs_values_tiled.dtype != b.dtype
            or rhs_values_tiled.device != b.device
            or not rhs_values_tiled.is_contiguous()
        ):
            raise ValueError(
                "tile-major fused RHS differs from its native storage contract"
            )
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
    if grouped and policy.split_k_slices != 1:
        raise ValueError("grouped fused quantization does not support split-K")
    if out is None and _atomic_output_precleared:
        raise ValueError("a precleared fused output must be caller-owned")
    if out is not None and (out.shape != (m, n, groups) or out.dtype != torch.bfloat16):
        raise ValueError("fused output must retain the planned BF16 shape")
    return _DenseFusedQuantLowering(
        m,
        n,
        k,
        groups,
        grouped,
        tuple(source.shape),
        tuple(source.stride()),
        str(b.dtype).removeprefix("torch."),
        str(sfb.dtype).removeprefix("torch."),
        expected_m,
        int(sm_count),
        mma_tiler_mn,
        policy,
        bool(sfb_k_replicated),
        rhs_values_tiled is not None,
        a_inner_span,
        inv_rope,
        int(head_dim),
        int(nope_dim),
        int(rope_dim),
        positions_dtype,
        cos_sin_dtype,
        out is not None,
        bool(_atomic_output_precleared),
    )


@program_cache(scope="preparation")
def _compile_dense_fused_quant_lowering(payload, device_ordinal):
    p = _DenseFusedQuantLowering.from_dict(payload)
    one_m = (p.expected_m if p.expected_m is not None else p.m) == 1
    split = p.policy.split_k_slices > 1
    partials = split and not p.policy.split_k_atomic_bf16
    with torch.cuda.device(device_ordinal):
        if p.grouped:
            gemm = _get_compiled_dense_gemm_fused_quant_a_grouped(
                p.n,
                p.k,
                p.groups,
                p.policy,
                p.mma_tiler_mn,
                p.sm_count,
                p.sfb_k_replicated,
                p.source_strides[0],
                p.k,
                p.inv_rope,
                p.head_dim,
                p.nope_dim,
                p.rope_dim,
                one_m,
                _cutlass_positions_dtype(getattr(torch, p.positions_dtype)),
                _cutlass_cos_sin_dtype(getattr(torch, p.cos_sin_dtype)),
            )
        else:
            gemm = _get_compiled_dense_gemm_fused_quant_a(
                p.n,
                p.k,
                cutlass.Float32 if partials else cutlass.BFloat16,
                p.policy,
                p.mma_tiler_mn,
                p.sm_count,
                p.sfb_k_replicated,
                p.b_tile_major,
                p.a_inner_span,
                p.policy.split_k_slices if partials else 1,
                one_m,
            )
        programs = {"gemm": gemm}
        if partials:
            if p.policy.split_k_slices != 2:
                raise ValueError("fused non-atomic split reduction requires two slices")
            programs["reduce"] = _reduce_split_k2_bf16_kernel.warmup(
                torch.float32,
                torch.bfloat16,
                p.m * p.n,
                BLOCK=1024,
                grid=(triton.cdiv(p.m * p.n, 1024),),
            )
        return programs


@dataclass(frozen=True)
class _DenseFusedQuantState:
    lowering: _DenseFusedQuantLowering
    device: torch.device
    gemm: Callable
    reduction: object | None
    alpha_one: torch.Tensor

    def run(
        self,
        source,
        b,
        sfb,
        out=None,
        *,
        positions=None,
        cos_sin_cache=None,
        rhs_values_tiled=None,
        split_k_workspace=None,
        stream=None,
    ):
        p = self.lowering
        m = int(source.shape[0])
        shape_matches = (
            tuple(source.shape[1:]) == p.source_shape[1:]
            and m <= p.m
            and (not p.grouped or m == p.m)
        )
        if (
            not shape_matches
            or source.stride() != p.source_strides
            or source.dtype != torch.bfloat16
            or source.device != self.device
        ):
            raise ValueError(
                "fused A exceeds its prepared capacity or differs in layout/device"
            )
        _require_dense_value(
            b, (p.n, p.k, p.groups), getattr(torch, p.b_dtype), self.device, "fused B"
        )
        if sfb.device != self.device or sfb.dtype != getattr(torch, p.sfb_dtype):
            raise ValueError("fused SFB differs from its prepared dtype/device")
        if (out is not None) != p.output_provided:
            raise ValueError("fused output allocation form changed")
        if (rhs_values_tiled is not None) != p.b_tile_major:
            raise ValueError("fused tiled RHS presence changed")
        if p.inv_rope:
            if (
                positions is None
                or positions.shape != (m,)
                or positions.dtype != getattr(torch, p.positions_dtype)
                or positions.device != self.device
                or not positions.is_contiguous()
                or cos_sin_cache is None
                or cos_sin_cache.ndim != 2
                or int(cos_sin_cache.shape[1]) != p.rope_dim
                or cos_sin_cache.dtype != getattr(torch, p.cos_sin_dtype)
                or cos_sin_cache.device != self.device
                or not cos_sin_cache.is_contiguous()
            ):
                raise ValueError("fused inverse-RoPE operands differ from preparation")
        elif positions is not None or cos_sin_cache is not None:
            raise ValueError("inverse-RoPE was not declared")
        if out is None:
            out = _empty_dense_gemm_output(
                m, p.n, p.groups, dtype=torch.bfloat16, device=self.device
            )
        else:
            _require_dense_value(
                out, (m, p.n, p.groups), torch.bfloat16, self.device, "fused C"
            )
        if m == 0:
            return out
        stream_int = cuda_stream_to_int(stream)
        if p.grouped:
            self.gemm(
                source,
                positions if p.inv_rope else source,
                cos_sin_cache if p.inv_rope else source,
                b,
                sfb,
                out,
                self.alpha_one,
                stream_int,
            )
            return out
        split = p.policy.split_k_slices > 1
        atomic = split and p.policy.split_k_atomic_bf16
        temporary = None
        if atomic:
            if not p.atomic_output_precleared:
                out.zero_()
        elif split:
            temporary = _dense_split_storage(
                split_k_workspace, p.policy.split_k_slices, m, p.n, self.device
            )
        b_launch = b
        if rhs_values_tiled is not None:
            if (
                rhs_values_tiled.shape != (1, 32, 32, 128, 128)
                or rhs_values_tiled.dtype != b.dtype
                or rhs_values_tiled.device != self.device
                or not rhs_values_tiled.is_contiguous()
            ):
                raise ValueError("fused tile-major RHS storage changed")
            b_launch = rhs_values_tiled
        self.gemm(
            source,
            b_launch,
            sfb,
            out if temporary is None else temporary,
            self.alpha_one,
            stream_int,
        )
        if temporary is not None:
            self.reduction[(triton.cdiv(m * p.n, 1024), 1, 1)](
                temporary, out, m * p.n, 1024
            )
        return out


def _materialize_dense_fused_quant(lowering, device):
    programs = _compile_dense_fused_quant_lowering(lowering.to_dict(), device.index)
    return _DenseFusedQuantState(
        lowering,
        device,
        programs["gemm"],
        programs.get("reduce"),
        _cached_alpha_one(device),
    )
