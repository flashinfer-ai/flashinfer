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

"""Dense GEMM execution entrypoints."""

# Import the launch module to register the Torch custom operators.
from b12x._lib.dense_gemm import launch

from typing import Literal
from typing import Optional
from typing import Tuple
from b12x._lib.utils.device import cuda_stream_to_int
import cutlass
from b12x._lib.utils.dtypes import cutlass_to_torch_dtype
from b12x._lib.fp6 import expand_mxfp6_packed_to_bytes
from b12x._lib.utils.dtypes import get_cutlass_dtype
from b12x._lib.utils.device import get_num_sm
from b12x._lib.utils.fp6_geometry import mxfp6_logical_k_from_packed_bytes
from b12x._lib.utils.fp6_geometry import mxfp6_tile_k
import time
import torch
from b12x._lib.dense_gemm.buffers import _cached_alpha_one, _empty_dense_gemm_output
from b12x._lib.dense_gemm.compile import (
    _get_compiled_dense_gemm,
    _get_compiled_dense_gemm_fused_quant_a,
    _get_compiled_dense_gemm_mxfp6,
)
from b12x._lib.dense_gemm.config import (
    _B12X_DENSE_SPLITK_TURBO,
    _B12X_TIMING,
    _B12X_TIMING_THRESHOLD_MS,
    _DENSE_FUSED_QUANT,
    _DENSE_LOAD_PATHS,
    _DenseGemmPolicy,
    _dense_gemm_policy_for,
    _use_direct_sfa_live16,
    logger,
)
from b12x._lib.dense_gemm.policy import (
    _select_default_dense_gemm_plan,
    _select_fp4_tile_k,
    _select_mxfp8_tile_k,
    _validate_mxfp8_bk64_plan,
)
from b12x._lib.dense_gemm.reductions import _reduce_split_k2_bf16


def dense_gemm_fused_quant_a(
    source: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    sfb_k_replicated: bool = False,
    rhs_values_tiled: Optional[torch.Tensor] = None,
    a_inner_span: int = 0,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    _atomic_output_precleared: bool = False,
    _split_k_workspace: Optional[torch.Tensor] = None,
    stream: object = None,
) -> torch.Tensor:
    """Small-M BF16-A -> MXFP8 GEMM with activation quantization in each CTA.

    a_inner_span > 0 reads A from an L-blocked source instead of contiguous
    rows: `source` is the `[M, span, K/span]` dense-GEMM mnl view over physical
    `[K/span, M, span]` storage (the WO tmp group-major layout). Follows the
    dense_gemm split-K policy (FP32 partials + fused reduce) instead of forcing
    a single un-split kernel, which loses ~2x at M=1 for N,K >= 4096.
    """

    a_inner_span = int(a_inner_span)
    if source.dtype != torch.bfloat16:
        raise ValueError("fused MXFP8 activation quantization requires BF16 A")
    if a_inner_span == 0:
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError(
                "fused MXFP8 activation quantization requires contiguous BF16 [M,K]"
            )
        m, k = map(int, source.shape)
    else:
        if a_inner_span % 32 != 0:
            raise ValueError(
                f"fused MXFP8 a_inner_span must be a multiple of 32, got {a_inner_span}"
            )
        if source.ndim != 3 or int(source.shape[1]) != a_inner_span:
            raise ValueError(
                "L-blocked fused MXFP8 A requires an [M, span, K/span] view, "
                f"got {tuple(source.shape)} for span={a_inner_span}"
            )
        m = int(source.shape[0])
        k = a_inner_span * int(source.shape[2])
        if source.stride() != (a_inner_span, 1, m * a_inner_span):
            raise ValueError(
                "L-blocked fused MXFP8 A must be a dense-GEMM mnl view over "
                f"physical [K/span, M, span] storage, got strides {source.stride()}"
            )
    if m < 1 or m > 8 or k % 128 != 0:
        raise ValueError(
            f"fused MXFP8 activation quantization requires 1<=M<=8 and K%128=0, got M={m}, K={k}"
        )
    if b.ndim != 3 or int(b.shape[1]) != k or int(b.shape[2]) != 1:
        raise ValueError(f"B must have shape [N,{k},1], got {tuple(b.shape)}")
    n = int(b.shape[0])
    sm_count = get_num_sm(source.device)
    if mma_tiler_mn is None:
        plan = _select_default_dense_gemm_plan(
            m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m
        )
        if plan.swap_ab or plan.load_path != "tma":
            raise ValueError(
                "fused MXFP8 activation quantization requires the unswapped TMA plan"
            )
        mma_tiler_mn = plan.mma_tiler_mn
    b_launch = b
    if rhs_values_tiled is not None:
        expected_tiled_shape = (1, 32, 32, 128, 128)
        if (n, k) != (4096, 4096) or mma_tiler_mn not in (
            (16, 64),
            (16, 128),
        ):
            raise ValueError(
                "tile-major fused-quant RHS is restricted to the production "
                "WO-B 16xN/BK128 plans"
            )
        if (
            rhs_values_tiled.shape != expected_tiled_shape
            or rhs_values_tiled.dtype != b.dtype
            or rhs_values_tiled.device != b.device
            or not rhs_values_tiled.is_contiguous()
        ):
            raise ValueError(
                "tile-major fused-quant RHS must be contiguous with shape "
                f"{expected_tiled_shape}, dtype {b.dtype}, and device {b.device}; "
                f"got shape={tuple(rhs_values_tiled.shape)}, "
                f"dtype={rhs_values_tiled.dtype}, device={rhs_values_tiled.device}"
            )
        b_launch = rhs_values_tiled
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        tile_k=128,
        expected_m=expected_m,
        generalize_mxfp8_split_k=True,
    )
    split_k_slices = policy.split_k_slices
    split_k_output = split_k_slices > 1
    split_k_atomic_bf16 = split_k_output and policy.split_k_atomic_bf16
    if out is None:
        if _atomic_output_precleared:
            raise ValueError("a precleared fused-quant output must be caller-owned")
        out = torch.empty((m, n, 1), dtype=torch.bfloat16, device=source.device)
    if out.shape != (m, n, 1) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(m, n, 1)}, got {out.dtype} {tuple(out.shape)}"
        )
    split_storage = None
    if split_k_atomic_bf16:
        if not _atomic_output_precleared:
            out.zero_()
        kernel_c_l = 1
        kernel_c_dtype = cutlass.BFloat16
        c_tensor_gpu = out
    elif split_k_output:
        split_storage = _dense_split_storage(
            _split_k_workspace, split_k_slices, m, n, source.device
        )
        kernel_c_l = split_k_slices
        kernel_c_dtype = cutlass.Float32
        c_tensor_gpu = split_storage
    else:
        kernel_c_l = 1
        kernel_c_dtype = cutlass.BFloat16
        c_tensor_gpu = out
    compiled = _get_compiled_dense_gemm_fused_quant_a(
        n,
        k,
        kernel_c_dtype,
        policy,
        mma_tiler_mn,
        sm_count,
        bool(sfb_k_replicated),
        rhs_values_tiled is not None,
        a_inner_span,
        kernel_c_l,
        (expected_m if expected_m is not None else m) == 1,
    )
    compiled(
        source,
        b_launch,
        sfb,
        c_tensor_gpu,
        _cached_alpha_one(source.device),
        cuda_stream_to_int(stream),
    )
    if split_storage is not None:
        _reduce_split_k2_bf16(split_storage.permute(1, 2, 0), out, m=m, n=n)
    return out


def _expand_packed_mxfp6_ab(t: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Expand a 3:4-packed FP6 operand ``(X, packed_k, L)`` to byte-containers.

    Returns an ``(X, num_codes, L)`` uint8 view whose underlying memory is laid out
    K-major (matching ``a_major``/``b_major == "k"``), so the compiled kernel reads
    each FP6 code from one byte. The launch only uses ``data_ptr`` plus the compiled
    ``(X, K, L)`` layout, so the returned view's strides need not be contiguous; the
    contiguous backing buffer is kept alive by the returned view.
    """
    t_lxk = t.permute(2, 0, 1).contiguous()  # (L, X, packed_k), packed_k fastest
    e_lxk = expand_mxfp6_packed_to_bytes(t_lxk, num_codes)  # (L, X, num_codes)
    return e_lxk.permute(1, 2, 0)  # (X, num_codes, L), K stride 1


def _dense_split_storage(workspace, slices, m, n, device):
    if workspace is None:
        return torch.empty((slices, m, n), dtype=torch.float32, device=device)
    if (
        workspace.dtype != torch.float32
        or workspace.device != device
        or not workspace.is_contiguous()
        or workspace.data_ptr() % 16
        or workspace.numel() < slices * m * n
    ):
        raise ValueError(
            "split-K workspace must be aligned contiguous FP32 storage with sufficient capacity"
        )
    return workspace.view(-1)[: slices * m * n].view(slices, m, n)


def dense_gemm(
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
) -> torch.Tensor:
    """Execute dense block-scaled GEMM for one expert-major batch stack.

    expected_m: optional regime hint (DeepGEMM-style). When set, the default tile
    is chosen for that representative M instead of being M-independent, giving a
    per-regime-optimal kernel that is still reused across all live M in the regime
    (e.g. expected_m<=128 selects a decode-tuned tile). Ignored when mma_tiler_mn
    is given. Live M stays a runtime arg; only the tile (a compile key) changes.

    MX-FP6 (``ab_dtype`` in ``float6_e3m2fn``/``float6_e2m3fn``) parameters:

    ``b_preexpanded``: when True the RHS is already in 1-byte-per-code FP8
    container layout (the output of ``_expand_packed_mxfp6_ab``) and the
    per-call weight expansion is skipped. Callers hoist the expansion to load
    time so the static weight is unpacked once instead of every token.

    ``a_preexpanded``: same for the LHS — its shape is then ``(M, K, L)`` with
    one code per byte (e.g. the quantizer's ``emit="bytes"`` output), so the
    logical K is read directly from the shape and no expansion runs.

    ``b_packed``: native packed-FP6 streaming. The RHS stays in the 3:4-packed
    wire format ``(N, 3K/4, L)``; the kernel TMA-streams the packed bytes and
    expands to byte-containers in smem. 25% less B HBM traffic than the
    byte-container layout and no expanded copy resident in VRAM. MX-FP6 only;
    mutually exclusive with ``b_preexpanded``.

    ``a_fmt`` / ``b_fmt``: optional per-operand MX sub-formats (``e2m3`` /
    ``e3m2`` / ``e4m3``). When omitted, both operands use the format implied by
    ``ab_dtype``. W6A8 dense passes ``a_fmt="e4m3"`` (activations) and
    ``b_fmt="e2m3"`` (weights) with ``ab_dtype="float6_e2m3fn"`` so the weight
    packing / expansion path stays on the MX-FP6 branch.

    ``x_bf16`` / ``w_gscale``: fused BF16 activation-quant inputs, only used
    when B12X_DENSE_FUSED_QUANT is enabled on an m=1 MX-FP6 launch.

    ``plain_fp8``: emit non-block-scaled E4M3 warp MMA while reusing the
    SM12x dense pipeline. The scalar ``alpha`` carries the combined activation
    and weight dequantization scale.

    ``row_scale``: optional contiguous ``(M,)`` tensor in the C dtype, applied
    per output row in the epilogue. MX-FP6 only.

    ``block_fp8``: accumulate ordinary E4M3 MMA over each K128 block, then
    apply compact FP32 activation ``[M,K/128]`` and weight
    ``[N/128,K/128]`` scales before adding it to the final accumulator.
    """
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
            a_torch = _expand_packed_mxfp6_ab(a_torch, k)
        if not (b_preexpanded or b_packed):
            b_torch = _expand_packed_mxfp6_ab(b_torch, k)
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
        # A caller's explicit M tile can differ from the default tile that
        # justified automatic BK64 selection. Keep its native BK128 path;
        # an explicitly forced incompatible BK64 still fails validation below.
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
                    "four-way split-K requires atomic-BF16 or an explicit "
                    "caller-owned FP32 reduction"
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
    if alpha is None:
        alpha = _cached_alpha_one(a_torch.device)
    stream_int = cuda_stream_to_int(stream)
    kernel_c_dtype_name = (
        "float32" if split_k_output and not split_k_atomic_bf16 else c_dtype
    )
    if row_scale is not None:
        if not is_mxfp6:
            raise ValueError("row_scale is only wired for the MX-FP6 path")
        if (
            row_scale.dim() != 1
            or row_scale.shape[0] != m
            or not row_scale.is_contiguous()
            or row_scale.dtype != cutlass_to_torch_dtype(c_cutlass_dtype)
            # The epilogue reads it through a raw device pointer, so a
            # host or wrong-device tensor is an illegal access, not an error.
            or row_scale.device != a_torch.device
            or row_scale.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "row_scale must be a contiguous 1-D tensor of shape (M,) in the "
                f"C dtype on the operand device; got shape "
                f"{tuple(row_scale.shape)} dtype {row_scale.dtype} device "
                f"{row_scale.device} for M={m}, C dtype "
                f"{cutlass_to_torch_dtype(c_cutlass_dtype)}, operand device "
                f"{a_torch.device}"
            )
    if is_mxfp6:
        # Dedicated MX-FP6 launch path (bypasses the torch custom ops, like
        # the quantized-C path): the byte-container operands plus the
        # x_bf16/w_gscale fused-quant tensors flow straight into the compiled
        # launch, and the FP6 fmt/packed axes key the compile cache.
        compiled_mxfp6 = _get_compiled_dense_gemm_mxfp6(
            n=n,
            k=k,
            l=l,
            c_l=l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=ab_cutlass_dtype,
            sf_dtype=get_cutlass_dtype(sf_dtype),
            c_dtype=c_cutlass_dtype,
            alpha_dtype=get_cutlass_dtype(alpha_dtype),
            sf_vec_size=sf_vec_size,
            mma_k=mma_k,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            policy=policy,
            sm_count=sm_count,
            sm_version="sm_120",
            mxfp6_fmt_a=mxfp6_fmt_a,
            mxfp6_fmt_b=mxfp6_fmt_b,
            b_packed=b_packed,
            a_preexpanded=a_preexpanded,
            b_preexpanded=b_preexpanded,
            alpha_is_one=alpha_is_one,
            row_scale=row_scale is not None,
            # Fused quantization changes code generation and is part of the
            # compiled-kernel key.
            fused_quant=_DENSE_FUSED_QUANT and x_bf16 is not None,
        )
        if out is None:
            out = _empty_dense_gemm_output(
                m,
                n,
                l,
                dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
                device=a_torch.device,
            )
        return compiled_mxfp6(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=out,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
            x_bf16_tensor_gpu=x_bf16,
            w_gscale_tensor_gpu=w_gscale,
            row_scale_tensor_gpu=row_scale,
        )
    if _quantized_c is not None:
        quant_c_values, quant_c_scale_rows, quant_c_scale_mma = _quantized_c
        quant_c_width = n * l
        expected_scale_mma_shape = (
            32,
            4,
            1,
            4,
            quant_c_width // 128,
            1,
        )
        if (
            split_k_output
            or swap_ab
            or c_dtype != "bfloat16"
            or m < 1
            or m > 16
            or n % 32 != 0
            or quant_c_width % 128 != 0
            or mma_tiler_mn[1] != 64
            or quant_c_values.shape != (m, quant_c_width)
            or not quant_c_values.is_contiguous()
            or quant_c_values.dtype != torch.float8_e4m3fn
            or quant_c_scale_rows.shape != (m, quant_c_width // 32)
            or not quant_c_scale_rows.is_contiguous()
            or quant_c_scale_rows.dtype != torch.float8_e8m0fnu
            or quant_c_scale_mma.shape != expected_scale_mma_shape
            or quant_c_scale_mma.dtype != torch.float8_e8m0fnu
        ):
            raise ValueError("quantized C is restricted to the BF16 WO-A decode layout")
        if out is None:
            out = _empty_dense_gemm_output(
                m,
                n,
                l,
                dtype=torch.bfloat16,
                device=a_torch.device,
            )
        compiled_quant_c = _get_compiled_dense_gemm(
            n=n,
            k=k,
            l=l,
            c_l=kernel_c_l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=ab_cutlass_dtype,
            sf_dtype=get_cutlass_dtype(sf_dtype),
            c_dtype=c_cutlass_dtype,
            alpha_dtype=get_cutlass_dtype(alpha_dtype),
            sf_vec_size=sf_vec_size,
            mma_k=mma_k,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            policy=policy,
            sm_count=sm_count,
            sm_version="sm_120",
            load_path=load_path,
            swap_ab=swap_ab,
            sfb_k_reuse=sfb_k_reuse,
            b_tile_major=rhs_values_tiled is not None,
            quantize_c=True,
            alpha_is_one=alpha_is_one,
            direct_sfa_live16=_use_direct_sfa_live16(
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
                alpha_is_one=alpha_is_one,
                is_mxfp8=is_mxfp8,
            ),
        )
        return compiled_quant_c(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_launch_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=out,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
            quant_c_values_gpu=quant_c_values,
            quant_c_scale_rows_gpu=quant_c_scale_rows,
            quant_c_scale_mma_gpu=quant_c_scale_mma,
        )
    if out is None and not plain_fp8:
        # No caller-owned output buffer: functional launch (allocate + return
        # inside the opaque op). The compile graph then carries no
        # auto_functionalized dense node mutating a (possibly strided) caller
        # view -- which inductor's decompose pass cannot remove. No is_compiling;
        # purely caller-intent, behaviorally identical to the eager out=None path.
        return torch.ops.b12x.dense_gemm_launch_functional(
            a_torch,
            b_launch_torch,
            sfa_torch,
            sfb_torch,
            alpha,
            n,
            k,
            l,
            kernel_c_l,
            ab_dtype,
            sf_dtype,
            c_dtype,
            kernel_c_dtype_name,
            alpha_dtype,
            sf_vec_size,
            mma_k,
            tile_k,
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            sm_count,
            policy.single_work_tile_per_cta,
            policy.direct_one_m_tile_scheduler,
            policy.use_m1_non_tma,
            policy.split_k_slices,
            policy.split_k_atomic_bf16,
            policy.large_m_unroll,
            load_path,
            swap_ab,
            sfb_k_reuse,
            alpha_is_one,
            stream_int,
        )
    split_storage = None
    split_scratch = None
    if split_k_output:
        if out is None:
            out = torch.empty(
                (m, n, l),
                dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
                device=a_torch.device,
            )
        if split_k_atomic_bf16:
            out.zero_()
        else:
            split_storage = _dense_split_storage(
                _split_k_workspace, split_k_slices, m, n, a_torch.device
            )
            split_scratch = split_storage.permute(1, 2, 0)
    elif out is None:
        out = torch.empty(
            (m, n, l),
            dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
            device=a_torch.device,
        )
    if alpha is None:
        alpha = _cached_alpha_one(a_torch.device)

    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    cache_before = _get_compiled_dense_gemm.cache_info() if _B12X_TIMING else None
    t_compiled = t0
    kernel_c_dtype_name = (
        "float32" if split_k_output and not split_k_atomic_bf16 else c_dtype
    )
    c_tensor_gpu = (
        out if split_k_atomic_bf16 else split_scratch if split_k_output else out
    )
    assert c_tensor_gpu is not None
    if plain_fp8:
        compiled_plain_fp8 = _get_compiled_dense_gemm(
            n=n,
            k=k,
            l=l,
            c_l=kernel_c_l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=ab_cutlass_dtype,
            sf_dtype=get_cutlass_dtype(sf_dtype),
            c_dtype=get_cutlass_dtype(kernel_c_dtype_name),
            alpha_dtype=get_cutlass_dtype(alpha_dtype),
            sf_vec_size=sf_vec_size,
            mma_k=mma_k,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            policy=policy,
            sm_count=sm_count,
            sm_version="sm_120",
            load_path=load_path,
            swap_ab=swap_ab,
            sfb_k_reuse=False,
            b_tile_major=False,
            alpha_is_one=alpha_is_one,
            plain_fp8=True,
            block_fp8=block_fp8,
        )
        compiled_plain_fp8(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_launch_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=c_tensor_gpu,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
        )
    else:
        torch.ops.b12x.dense_gemm_launch(
            a_torch,
            b_launch_torch,
            sfa_torch,
            sfb_torch,
            c_tensor_gpu,
            alpha,
            n,
            k,
            l,
            kernel_c_l,
            ab_dtype,
            sf_dtype,
            kernel_c_dtype_name,
            alpha_dtype,
            sf_vec_size,
            mma_k,
            tile_k,
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            sm_count,
            policy.single_work_tile_per_cta,
            policy.direct_one_m_tile_scheduler,
            policy.use_m1_non_tma,
            policy.split_k_slices,
            policy.split_k_atomic_bf16,
            policy.large_m_unroll,
            load_path,
            swap_ab,
            sfb_k_reuse,
            alpha_is_one,
            _target_occupancy_override,
            stream_int,
        )
    result = out
    if split_k_output and not split_k_atomic_bf16:
        assert split_scratch is not None
        assert out is not None
        _reduce_split_k2_bf16(split_scratch, out, m=m, n=n)
        result = out
    if _B12X_TIMING:
        t_launch = time.perf_counter()
        cache_after = _get_compiled_dense_gemm.cache_info()
        assert cache_before is not None
        compile_ms = (t_compiled - t0) * 1000.0
        launch_ms = (t_launch - t_compiled) * 1000.0
        total_ms = (t_launch - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_dense_gemm timing m=%d n=%d k=%d l=%d ab=%s sf=%s c=%s "
                "tile=%s load=%s swap_ab=%s cache_hit=%s compile_or_lookup=%.3fms "
                "launch_enqueue=%.3fms total=%.3fms cache=%s",
                m,
                n,
                k,
                l,
                ab_dtype,
                sf_dtype,
                c_dtype,
                mma_tiler_mn,
                load_path,
                swap_ab,
                cache_after.hits > cache_before.hits,
                compile_ms,
                launch_ms,
                total_ms,
                cache_after,
            )
    return result
