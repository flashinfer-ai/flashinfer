# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Masked Grouped GEMM Wrapper for CuTe-DSL Kernels
===================================================

This module provides the unified entry point for masked grouped GEMM
with block-scaled inputs, supporting both Blackwell (SM100/SM103) and
Rubin (SM107) architectures.

It handles:
- Architecture detection via compute capability
- Parameter translation between Blackwell (2-tuple mma_tiler_mn) and
  Rubin (3-tuple mma_tiler + mma_inst_shape) conventions
- Routing to the appropriate arch-specific implementation

Architecture-specific implementations:
- SM100/SM103 (Blackwell): grouped_gemm_masked_blackwell.py
- SM107 (Rubin):           grouped_gemm_masked_rubin.py
"""

from typing import Optional, Tuple, Union

import torch

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from flashinfer.utils import get_compute_capability
from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.gemm import grouped_gemm_nt_masked_trace
from flashinfer.cute_dsl.utils import get_num_sm


@flashinfer_api(trace=grouped_gemm_nt_masked_trace)
def grouped_gemm_nt_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    topk_weights: Optional[torch.Tensor] = None,
    idx_src_info: Optional[torch.Tensor] = None,
    rank_src_info: Optional[torch.Tensor] = None,
    out_ptrs: Optional[torch.Tensor] = None,
    num_ranks: int = 0,
    dst_signals: Optional[torch.Tensor] = None,
    sm_count: Optional[int] = None,
    barrier_flag_local: Optional[torch.Tensor] = None,
    barrier_flag_multicast: Optional[torch.Tensor] = None,
    is_combine_fusion: bool = False,
    is_swap_ab: bool = False,
    activation_type: Optional[Union[ActivationType, int]] = None,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    alpha_up: Optional[torch.Tensor] = None,
    a_per_token_scale: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    **kwargs,
):
    r"""Masked, batched, block-scaled GEMM on Blackwell (SM100/SM103) and Rubin (SM107).

    Routes to the architecture-specific kernel for the device's compute
    capability: SM100/SM103 use the Blackwell kernel, SM107 the Rubin one.

    Executes a masked, batched matrix multiplication with scale factors and
    optional per-batch alpha scaling on the output.  ``alpha`` is currently
    applied internally by the kernel; see Notes for the canonical tensor
    layouts.

    Parameters
    ----------
    lhs : Tuple[torch.Tensor, torch.Tensor]
        ``(A, SFA)`` — left-hand-side input tensor and its scale-factor tensor.
        ``A`` has logical shape ``(m, k, l)`` (physically ``(l, m, k)``);
        for FP4 with 8-bit storage the physical shape is ``(m, k/2, l)``.
        ``SFA`` has logical shape ``(m32, m4, rm, k4, rk, l)``
        (physically ``(l, rm, rk, m32, m4, k4)``).
    rhs : Tuple[torch.Tensor, torch.Tensor]
        ``(B, SFB)`` — right-hand-side input tensor and its scale-factor
        tensor.  ``B`` has logical shape ``(n, k, l)``
        (physically ``(l, n, k)``; FP4 with 8-bit storage is
        ``(n, k/2, l)``).  ``SFB`` has logical shape
        ``(n32, n4, rn, k4, rk, l)``
        (physically ``(l, rn, rk, n32, n4, k4)``).
    out : torch.Tensor
        Output tensor of shape ``(l, m, n)``.  Mutated in place.
    masked_m : torch.Tensor
        1-D ``int32`` tensor of shape ``(l,)`` giving the valid row count of
        each batch.  Rows above ``masked_m[batch]`` are ignored.
    ab_dtype : str
        Data type for ``A`` and ``B``.  One of ``"float4_e2m1fn"``,
        ``"float8_e4m3fn"``, ``"float8_e5m2"``.
    sf_dtype : str
        Data type for the scale factors.  One of ``"float8_e8m0fnu"`` or
        ``"float8_e4m3fn"``.
    c_dtype : str
        Data type for output matrix ``C``.  One of ``"float16"``,
        ``"bfloat16"``, ``"float32"``, ``"float8_e4m3fn"``, ``"float8_e5m2"``,
        or ``"float4_e2m1fn"`` (SM100/SM103 only) to quantize the output to
        NVFP4; see ``out_scale``.
    sf_vec_size : int
        Vector size for scale factors (typically 16 or 32).
    topk_weights : Optional[torch.Tensor]
        2-D ``float32`` tensor of shape ``(l, m)`` containing top-k routing
        weights.  Defaults to ``None``.
    idx_src_info : Optional[torch.Tensor]
        2-D ``int32`` tensor of shape ``(l, m)`` carrying source-index metadata
        for the combine fusion path.  Defaults to ``None``.
    rank_src_info : Optional[torch.Tensor]
        2-D ``int32`` tensor of shape ``(l, m)`` carrying rank-source metadata
        for the combine fusion path.  Defaults to ``None``.
    out_ptrs : Optional[torch.Tensor]
        1-D ``int64`` tensor of shape ``(num_ranks,)`` containing remote output
        pointers for multi-rank combine.  Defaults to ``None``.
    num_ranks : int
        Number of ranks participating in the combine path.  Defaults to ``0``.
    dst_signals : Optional[torch.Tensor]
        Optional 1-D signal tensor used by the combine-fusion path to notify
        consumers.  Defaults to ``None``.
    sm_count : Optional[int]
        Number of SMs to use.  If ``None``, the runtime picks the max available
        under the CTA configuration.
    barrier_flag_local : Optional[torch.Tensor]
        1-D ``int32`` tensor of shape ``(sm_count,)`` containing flags for
        local barrier synchronization (spin-lock wait in multi-rank ops).
        Defaults to ``None``.
    barrier_flag_multicast : Optional[torch.Tensor]
        1-D ``int32`` tensor of shape ``(sm_count,)`` containing flags for
        multicast barrier synchronization (release across ranks).  Defaults to
        ``None``.
    is_combine_fusion : bool
        If ``True``, enable the fused GEMM + combine operation mode.  Defaults
        to ``False``.
    is_swap_ab : bool
        If ``True``, swap the ``lhs``/``rhs`` input tensors.  Defaults to
        ``False``.
    activation_type : Optional[Union[ActivationType, int]]
        Gated activation fused into the epilogue (SM100/SM103 only).  Only
        ``ActivationType.Swiglu`` is supported.  ``None`` (the default) keeps
        the plain GEMM epilogue.  When set, ``B`` and ``SFB`` rows must be
        interleaved in 64-row blocks with the up projection first,
        ``[up_0:64, gate_0:64, up_64:128, gate_64:128, ...]``, ``n`` must be a
        multiple of ``mma_tiler_mn[1]``, and ``out`` has ``n / 2`` columns,
        shape ``(m, n / 2, l)`` (``(m, n / 4, l)`` bytes for NVFP4 output).  Each output element is
        ``gate * sigmoid(swiglu_alpha * gate) * (up + swiglu_beta)`` with
        ``gate = min(alpha * s * acc_gate, swiglu_limit)``,
        ``up = clamp(alpha_up * s * acc_up, -swiglu_limit, swiglu_limit)``
        and ``s`` the optional per-row ``a_per_token_scale``.  Requires
        ``mma_tiler_mn`` ``(128, 128)`` or ``(128, 256)`` and does not support
        combine fusion, ``is_swap_ab`` or ``dst_signals``.
    swiglu_alpha : float
        Sigmoid multiplier of the gated activation.  Defaults to ``1.0``.
    swiglu_beta : float
        Bias added to the up projection.  Defaults to ``0.0``.
    swiglu_limit : float
        Clamp limit of the gated activation.  Defaults to the float32 maximum.
    alpha_up : Optional[torch.Tensor]
        Optional contiguous ``float32`` tensor of shape ``(l,)`` scaling the up
        projection when the up and gate weights have different global scales.
        Defaults to ``alpha``.
    a_per_token_scale : Optional[torch.Tensor]
        Optional contiguous ``float32`` tensor of shape ``(l, m)`` holding a
        per-row scale of ``A``, applied before the gated activation.
    out_scale : Optional[torch.Tensor]
        Required when ``c_dtype="float4_e2m1fn"``.  Receives the ``float8_e4m3fn``
        block scales (``sf_vec_size`` 16) of the output in the same swizzled
        layout as ``SFA``, so ``out`` and ``out_scale`` can feed the next masked
        GEMM directly: logical shape ``(32, 4, rm, 4, rk, l)``, physically
        ``(l, rm, rk, 32, 4, 4)``, with ``rk = n_out / 64``.  ``out`` is then a
        ``uint8`` tensor of shape ``(m, n_out / 2, l)`` holding two E2M1 values
        per byte, where ``n_out`` is the number of output columns.  The
        epilogue rounds the output to BF16 and quantizes it with the default
        recipe of ``scaled_fp4_grouped_quantize``, giving the same bits as
        quantizing a BF16 output with that function.  Requires
        ``mma_tiler_mn[0] == 128``, ``n`` a multiple of ``mma_tiler_mn[1]``
        and ``n_out`` a multiple of 64.  Works with or without
        ``activation_type``.  Raises ``ValueError`` when
        ``FLASHINFER_NVFP4_4OVER6`` or ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH``
        is set, since the epilogue does not implement those recipes.
    global_scale : Optional[torch.Tensor]
        Required when ``c_dtype="float4_e2m1fn"``.  Contiguous ``float32``
        tensor of shape ``(l,)`` with the per-batch NVFP4 global scale, as the
        ``a_global_sf`` argument of ``scaled_fp4_grouped_quantize``.
    **kwargs
        Additional keyword arguments.  Currently recognized:

        * ``mma_tiler_mn`` (``Tuple[int, int]``): shape of the MMA tiler
          ``(M, N)``.  Defaults to ``(128, 128)``.  ``mma_tiler_mn[0] == 256``
          enables the 2-CTA MMA path.  Must be ``(128, 128)`` when
          ``is_combine_fusion=True``.
        * ``cluster_shape_mn`` (``Tuple[int, int]``): shape of the CTA
          cluster ``(ClusterM, ClusterN)``.  Defaults to ``(1, 1)``.
        * ``alpha`` (``Optional[torch.Tensor]``): optional 1-D tensor of
          shape ``(l,)`` containing per-batch scaling factors.  When
          provided, each batch output is multiplied by its corresponding
          alpha value: ``out = alpha * (A @ B)``.
        * ``alpha_dtype`` (``str``): elemental dtype string for the
          ``alpha`` tensor (e.g. ``"float32"``).  Required when ``alpha``
          is provided.

        Other entries are reserved for forward-compatible kernel options.

    Notes
    -----
    Tensor-layout conventions:

    * ``l`` is the batch size, ``m``/``n`` are the row/column counts, and
      ``k`` is the contraction dimension.
    * ``m32`` and ``n32`` are the constants ``32``; ``m4`` and ``n4`` are the
      constants ``4``; and ``k4`` is the constant ``4``.
    * ``m32 * m4 * rm`` equals ``M``, where ``M`` is ``m`` padded up to the
      nearest multiple of 128.
    * ``n32 * n4 * rn`` equals ``N``, where ``N`` is ``n`` padded up to the
      nearest multiple of 128.
    * ``k4 * rk`` equals ``K``, where ``K`` is ``k / sf_vec_size`` padded up to
      the nearest multiple of 4.

    Masking is applied per batch via ``masked_m``.  When ``alpha`` is
    provided (see ``**kwargs``), each batch output is multiplied by its
    corresponding alpha value: ``out = alpha * (A @ B)``.
    """
    a_torch = lhs[0]

    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)

    # Detect architecture
    major, minor = get_compute_capability(a_torch.device)

    if major == 11 and minor == 0:
        raise ValueError("SM110 is not supported for cute-dsl backend.")

    if major == 10 and minor == 7:
        # ----------------------------------------------------------------
        # SM107 (Rubin)
        # ----------------------------------------------------------------
        from .grouped_gemm_masked_rubin import _grouped_gemm_nt_masked_sm107

        # Extract Blackwell-style kwargs and translate to Rubin conventions
        mma_tiler_mn = kwargs.pop("mma_tiler_mn", (128, 128))
        cluster_shape_mn = kwargs.pop("cluster_shape_mn", (1, 1))
        alpha = kwargs.pop("alpha", None)
        alpha_dtype = kwargs.pop("alpha_dtype", None)

        # Translate 2-tuple mma_tiler_mn to 3-tuple mma_tiler + mma_inst_shape
        # K-mode depends on data type:
        #   FP8:  mma_inst_shape_k=64,  mma_tiler_k=128  (2x B-reuse)
        #   FP4:  mma_inst_shape_k=128, mma_tiler_k=256  (2x B-reuse)
        if ab_dtype == "float4_e2m1fn":
            inst_k, tiler_k = 128, 256
        else:
            inst_k, tiler_k = 64, 128
        mma_tiler = (mma_tiler_mn[0], mma_tiler_mn[1], tiler_k)
        mma_inst_shape = (mma_tiler_mn[0], mma_tiler_mn[1], inst_k)

        # The combine-fusion / multi-rank parameters are Blackwell-only; the
        # Rubin kernel has no equivalent, so reject them explicitly instead of
        # silently ignoring them.
        unsupported = [
            name
            for name, value, default in (
                ("topk_weights", topk_weights, None),
                ("idx_src_info", idx_src_info, None),
                ("rank_src_info", rank_src_info, None),
                ("out_ptrs", out_ptrs, None),
                ("num_ranks", num_ranks, 0),
                ("barrier_flag_local", barrier_flag_local, None),
                ("barrier_flag_multicast", barrier_flag_multicast, None),
                ("is_combine_fusion", is_combine_fusion, False),
                ("is_swap_ab", is_swap_ab, False),
                ("activation_type", activation_type, None),
                ("alpha_up", alpha_up, None),
                ("a_per_token_scale", a_per_token_scale, None),
                ("out_scale", out_scale, None),
                ("global_scale", global_scale, None),
            )
            if value is not default
        ]
        if unsupported:
            raise NotImplementedError(
                "The Rubin (SM107) masked grouped GEMM does not support: "
                + ", ".join(unsupported)
            )
        if kwargs:
            raise ValueError(f"Unsupported kwargs: {kwargs}")

        return _grouped_gemm_nt_masked_sm107(
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            dst_signals=dst_signals,
            sm_count=sm_count,
            mma_tiler=mma_tiler,
            mma_inst_shape=mma_inst_shape,
            cluster_shape_mn=cluster_shape_mn,
            alpha=alpha,
            alpha_dtype=alpha_dtype,
        )

    elif major >= 10:
        # ----------------------------------------------------------------
        # SM100/SM103 (Blackwell)
        # ----------------------------------------------------------------
        from .grouped_gemm_masked_blackwell import _grouped_gemm_nt_masked_sm100

        return _grouped_gemm_nt_masked_sm100(
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            topk_weights=topk_weights,
            idx_src_info=idx_src_info,
            rank_src_info=rank_src_info,
            out_ptrs=out_ptrs,
            num_ranks=num_ranks,
            dst_signals=dst_signals,
            sm_count=sm_count,
            barrier_flag_local=barrier_flag_local,
            barrier_flag_multicast=barrier_flag_multicast,
            is_combine_fusion=is_combine_fusion,
            is_swap_ab=is_swap_ab,
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            alpha_up=alpha_up,
            a_per_token_scale=a_per_token_scale,
            out_scale=out_scale,
            global_scale=global_scale,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Masked grouped GEMM is only supported on SM100+ "
            f"(Blackwell/Rubin), got SM{major}{minor}"
        )
