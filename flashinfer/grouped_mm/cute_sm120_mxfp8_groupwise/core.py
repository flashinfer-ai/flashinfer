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

SM120 cute backend for the MXFP8 grouped GEMM family. Public ``group_gemm_*``
entries are re-exported via the parent :mod:`flashinfer.grouped_mm` package.
"""

import functools
from typing import Literal, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...jit.cute_sm120_mxfp8_groupwise import gen_gemm_sm120_module_cute_mxfp8
from ...utils import supported_compute_capability


@functools.cache
def get_gemm_sm120_module_cute_mxfp8():
    """MXFP8 grouped MM module accessor for SM120 cute backend."""
    return gen_gemm_sm120_module_cute_mxfp8().build_and_load()


def _check_m_indptr(m_indptr: torch.Tensor, num_experts: int) -> None:
    """Validate ``m_indptr`` metadata (no GPU→CPU sync).

    Caller contract: ``m_indptr[0] == 0``, monotonic non-decreasing, last element
    equals the packed A row count. Value invariants are not enforced here to avoid
    GPU→CPU sync on a low-latency kernel path.
    """
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32; got {m_indptr.dtype}")
    if m_indptr.dim() != 1:
        raise ValueError(f"m_indptr must be 1-D; got {m_indptr.dim()}D")
    if m_indptr.numel() != num_experts + 1:
        raise ValueError(
            f"m_indptr.numel() must equal num_experts + 1 = {num_experts + 1}; "
            f"got {m_indptr.numel()}"
        )


def _check_scale_granularity_mnk(scale_granularity_mnk: Tuple[int, int, int]) -> None:
    """Validate the per-token UE8M0 ``scale_granularity_mnk`` contract shared by all
    MXFP8 cute SM120 GEMM entries. Accepts ``(1, 1, 32)`` or ``(1, 1, 128)``.
    """
    if len(scale_granularity_mnk) != 3:
        raise ValueError(
            f"scale_granularity_mnk must be a 3-tuple (m_gran, n_gran, k_gran); "
            f"got length {len(scale_granularity_mnk)}"
        )
    if scale_granularity_mnk[0] != 1:
        raise ValueError(
            f"scale_granularity_mnk[0] (m_gran) must be 1; got {scale_granularity_mnk[0]}"
        )
    if scale_granularity_mnk[1] != 1:
        raise ValueError(
            f"scale_granularity_mnk[1] (n_gran) must be 1 (kernel only exposes granK "
            f"along K; 2D block B-scale must be broadcast to per-token caller-side); "
            f"got {scale_granularity_mnk[1]}"
        )
    if scale_granularity_mnk[2] not in (32, 128):
        raise ValueError(
            f"scale_granularity_mnk[2] (k_gran) must be 32 or 128; "
            f"got {scale_granularity_mnk[2]}"
        )


def _check_scale_major_mode_mxfp8(scale_major_mode: str) -> None:
    """Validate ``scale_major_mode`` for the MXFP8 cute SM120 GEMM entries.

    The kernel only consumes per-token INT32-packed UE8M0 scales in MN-major
    TMA-aligned layout. Future K-major support would require kernel changes;
    until then any value other than ``"MN"`` is rejected.
    """
    if scale_major_mode != "MN":
        raise ValueError(
            f'scale_major_mode must be "MN" (kernel currently only supports the '
            f'per-token MN-major TMA-aligned UE8M0 scale layout); got "{scale_major_mode}"'
        )


@supported_compute_capability([120, 121])
@flashinfer_api
def moe_gemm_mxfp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 1, 128),
    scale_major_mode: Literal["MN"] = "MN",
    backend: Literal["cute"] = "cute",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform grouped GEMM with MXFP8 inputs in zero-padding mode using groupwise UE8M0
    scaling. Currently only supported on NVIDIA RTX PRO 6000 Blackwell (SM120) architecture.

    Zero-padding mode accepts token-packed input ``a`` (no per-expert pre-padding along M)
    with 4-row per-expert padding on the scale tensor ``a_scale``. The group descriptor is
    a CSR cumsum ``m_indptr``. This mode is optimized for decoding with small per-expert M
    (down to ``m_per_expert = 1``) where DeepGEMM-style contiguous padding would waste
    memory and compute.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn``.
        Token-packed across experts; ``cum_m`` is the cumulative sum of segment lengths.

    b: torch.Tensor
        Column-major input tensor shape ``(num_experts, n, k)``, data type is
        ``torch.float8_e4m3fn``.

    a_scale: torch.Tensor
        Int32-packed UE8M0 scale tensor for ``a`` (4 UE8M0 scales packed per int32), shape
        ``(m_padded, k_align)`` where ``m_padded = (cum_m + num_experts * 3) // 4 * 4`` and
        ``k_align = (k + 4 * k_granularity - 1) // (4 * k_granularity)``. Data type is
        ``torch.int32``.

    b_scale: torch.Tensor
        Int32-packed UE8M0 scale tensor for ``b`` in per-token layout, shape
        ``(num_experts, n, k_align)``. Data type is ``torch.int32``. See Notes for the
        per-token layout requirement.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(num_experts + 1,)``, data type is
        ``torch.int32``. ``m_indptr[0] = 0``, ``m_indptr[num_experts] = cum_m``.

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, ``(m_granularity, n_granularity, k_granularity)``.
        Accepted values: ``(1, 1, 128)`` (DeepGEMM-style production, default) or
        ``(1, 1, 32)`` (OCP MXFP8). ``m_granularity`` and ``n_granularity`` must both be
        ``1`` (per-token scaling along M and N); ``k_granularity`` must be ``32`` or ``128``.
        Anything else raises ``ValueError``.

    backend: Literal["cute"]
        Backend selector. Currently only ``"cute"`` is implemented.

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, an output tensor will be
        created.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor. Currently only ``torch.bfloat16`` is supported.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    - MXFP8 uses UE8M0 scales over K-axis blocks of size 32 (OCP spec) or 128
      (DeepGEMM convention). Both ``a_scale`` and ``b_scale`` must be provided in per-token
      layout: one UE8M0 scale per row along M (for ``a``) or N (for ``b``), packed 4 scales
      per int32 along the K-axis blocks.
    - If a caller starts from a 2D ``(k_granularity, k_granularity)`` block-quantized
      ``b_scale``, it must be broadcast to per-token shape ``(num_experts, n, k_align)``
      before invoking this function (one scale per N-row).
    """
    _check_scale_granularity_mnk(scale_granularity_mnk)
    _check_scale_major_mode_mxfp8(scale_major_mode)
    _check_m_indptr(m_indptr, num_experts=b.shape[0])
    if backend != "cute":
        raise NotImplementedError(
            f'Only backend="cute" is currently implemented; got backend="{backend}"'
        )

    if out_dtype is None:
        out_dtype = out.dtype if out is not None else torch.bfloat16
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"Only out_dtype=torch.bfloat16 is supported; got {out_dtype}"
        )

    n = b.shape[1]
    if out is None:
        out = torch.empty((a.shape[0], n), dtype=out_dtype, device=a.device)

    get_gemm_sm120_module_cute_mxfp8().moe_gemm_mxfp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        out,
        scale_major_mode,
        scale_granularity_mnk[0],
        scale_granularity_mnk[1],
        scale_granularity_mnk[2],
    )
    return out
