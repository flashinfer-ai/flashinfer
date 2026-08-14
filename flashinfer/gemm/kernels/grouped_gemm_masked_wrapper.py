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

Location: flashinfer/gemm/kernels/grouped_gemm_masked_wrapper.py

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

from typing import Optional, Tuple

import torch

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
    dst_signals: Optional[torch.Tensor] = None,
    sm_count: Optional[int] = None,
    **kwargs,
):
    """
    Executes a masked, batched matrix multiplication (GEMM) with scale factors
    and optional alpha scaling at output.

    Automatically routes to the appropriate architecture-specific kernel based on
    the device's compute capability:
    - SM100/SM103 (Blackwell) -> grouped_gemm_masked_blackwell
    - SM107 (Rubin)           -> grouped_gemm_masked_rubin

    Args:
        lhs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing (A, SFA).
            - A should be in (m, k, l) order, but physically (l, m, k).
              For fp4 tensor with 8-bit storage, shape is (m, k/2, l).
            - SFA should be in (m32, m4, rm, k4, rk, l) order,
              but physically (l, rm, rk, m32, m4, k4).
        rhs (Tuple[torch.Tensor, torch.Tensor]): Tuple containing (B, SFB).
            - B should be in (n, k, l) order, but physically (l, n, k).
              For fp4 tensor with 8-bit storage, shape is (n, k/2, l).
            - SFB should be in (n32, n4, rn, k4, rk, l) order,
              but physically (l, rn, rk, n32, n4, k4).
        out (torch.Tensor): Output tensor with shape (l, m, n).
        masked_m (torch.Tensor): 1D tensor of shape (l,) specifying the valid
            row count for each batch (used for masking).
        ab_dtype (str): Data type for A and B matrices.
            Supported: "float4_e2m1fn", "float8_e4m3fn", "float8_e5m2".
        sf_dtype (str): Data type for scale factors.
            Supported: "float8_e8m0fnu", "float8_e4m3fn".
        c_dtype (str): Data type for output matrix C.
            Supported: "float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2".
        sf_vec_size (int): Vector size for scale factors. Typically 16 or 32.
        dst_signals (torch.Tensor, optional): Destination signals tensor for DSM.
        sm_count (int, optional): Number of SMs to use. Default: max available.
        mma_tiler_mn (Tuple[int, int], optional): Shape of the MMA tiler (M, N).
            Default: (128, 128). Used for SM100/SM103. For SM107, this is automatically
            translated to a 3-tuple with dtype-dependent K-mode.
        cluster_shape_mn (Tuple[int, int], optional): Shape of the CTA cluster
            (ClusterM, ClusterN). Default: (1, 1).
        alpha_dtype (str, optional): Data type for alpha scaling factors.
        alpha (torch.Tensor, optional): Optional 1D tensor of shape (l,) containing
            per-batch scaling factors. Performs per-batch scaling: out = alpha * (A @ B).

    Notes:
        - Legends of the input tensors:
            * `l` is the batch size, `m/n` is the number of rows, and `k` is the
              number of columns.
            * `m/n32`, `m/n4`, `k4` are constant values 32, 4, 4 respectively.
            * `m32 * m4 * rm` should be same as `M`, which is `m` padded up to
              the nearest multiple of 128.
            * `n32 * n4 * rn` should be same as `N`, which is `n` padded up to
              the nearest multiple of 128.
            * `k4 * rk` should be same as `K`, which is `k / sf_vec_size` padded
              up to the nearest multiple of 4.
        - The function applies masking per batch using masked_m.
        - If alpha is provided, each batch output is multiplied by its corresponding
          alpha value. out = alpha * (A @ B).
        - The result is written to out.
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

        assert len(kwargs) == 0, f"Unsupported kwargs: {kwargs}"

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
            dst_signals=dst_signals,
            sm_count=sm_count,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Masked grouped GEMM is only supported on SM100+ "
            f"(Blackwell/Rubin), got SM{major}{minor}"
        )
