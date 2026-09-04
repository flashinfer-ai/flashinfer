# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BF16 x per-tensor-scaled FP8 GEMM public API."""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import mm_bf16_fp8_trace
from ..utils import backend_requirement, supported_compute_capability


@supported_compute_capability([120, 121])
def _check_mm_bf16_fp8_problem_size(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
):
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"mm_bf16_fp8 expects A [M,K] and B [K,N], got "
            f"{tuple(A.shape)} and {tuple(B.shape)}."
        )
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"mm_bf16_fp8 K mismatch: {tuple(A.shape)} and {tuple(B.shape)}."
        )
    if A.dtype != torch.bfloat16:
        raise TypeError(f"A must be bfloat16, got {A.dtype}.")
    if B.dtype != torch.float8_e4m3fn:
        raise TypeError(f"B must be float8_e4m3fn, got {B.dtype}.")
    if B_scale.dtype != torch.float32 or B_scale.numel() != 1:
        raise ValueError("B_scale must be a float32 scalar dequantization tensor.")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported output dtype {dtype}.")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous.")
    if not B.T.is_contiguous():
        raise ValueError(
            "B must be column-major [K,N], backed by contiguous [N,K] storage."
        )
    if not B_scale.is_contiguous():
        raise ValueError("B_scale must be contiguous.")
    expected_out = (A.shape[0], B.shape[1])
    if out is not None and (
        out.shape != expected_out
        or out.dtype != dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous {expected_out} with dtype={dtype} on {A.device}."
        )
    if not (A.device == B.device == B_scale.device):
        raise ValueError("A, B, and B_scale must be on the same device.")
    return True


@backend_requirement(
    {},
    common_check=_check_mm_bf16_fp8_problem_size,
)
@flashinfer_api(trace=mm_bf16_fp8_trace)
def mm_bf16_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A16 GEMM using an unbatched :func:`bmm_fp8` weight.

    Computes ``A @ dequant(B)`` where ``A`` is BF16 and ``B`` is a
    per-tensor-scaled E4M3 weight. ``B`` must be a column-major ``[K, N]``
    view of the storage accepted by :func:`bmm_fp8`; adding a size-one batch
    dimension requires no copy or preprocessing.

    Parameters
    ----------
    A : torch.Tensor
        Contiguous BF16 activation matrix with shape ``(M, K)``.
    B : torch.Tensor
        E4M3 weight matrix with shape ``(K, N)``. It must be a column-major
        view backed by contiguous ``(N, K)`` storage.
    B_scale : torch.Tensor
        Contiguous scalar FP32 tensor containing the per-tensor weight
        dequantization scale.
    dtype : torch.dtype
        Output data type. Supported values are ``torch.bfloat16`` and
        ``torch.float16``. Defaults to ``torch.bfloat16``.
    out : Optional[torch.Tensor]
        Optional contiguous output tensor with shape ``(M, N)`` and the data
        type specified by ``dtype``.

    Returns
    -------
    torch.Tensor
        The output matrix with shape ``(M, N)``.

    This API currently contains a PyTorch reference implementation. The
    implementation in ``kernels/dense_bf16_fp8_gemm_sm12x.py`` is the
    insertion point for an optimized kernel.
    """
    if out is None:
        out = torch.empty((A.shape[0], B.shape[1]), dtype=dtype, device=A.device)
    from .kernels.dense_bf16_fp8_gemm_sm12x import mm_bf16_fp8_sm12x

    return mm_bf16_fp8_sm12x(A, B, B_scale, out)


__all__ = ["mm_bf16_fp8"]
