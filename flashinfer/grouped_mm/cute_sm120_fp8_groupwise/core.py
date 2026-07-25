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

SM120 cute backend for the FP8 float-scale grouped GEMM family. Public entries
are re-exported via the parent :mod:`flashinfer.grouped_mm` package.
"""

import functools
from typing import Literal, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...jit.cute_sm120_mxfp8_groupwise import gen_gemm_sm120_module_cute_mxfp8
from ...utils import supported_compute_capability

from ..cute_sm120_mxfp8_groupwise import _check_m_indptr


@functools.cache
def get_gemm_sm120_module_cute_fp8():
    """FP8 grouped MM module accessor for SM120 cute backend.

    FP8 shares the single-``.so`` JIT module with the MXFP8 entries; the accessor
    is separate so callers depend only on the FP8 surface.
    """
    return gen_gemm_sm120_module_cute_mxfp8().build_and_load()


def _check_scale_granularity_mnk_fp8(
    scale_granularity_mnk: Tuple[int, int, int],
) -> None:
    """Validate ``scale_granularity_mnk`` for the FP8 float-scale cute SM120 entries.

    The FP8 kernel contract is fixed 1d2d ``(1, 128, 128)``: per-token A-scale along M,
    ``(128, 128)``-block B-scale.
    """
    if len(scale_granularity_mnk) != 3:
        raise ValueError(
            f"scale_granularity_mnk must be a 3-tuple (m_gran, n_gran, k_gran); "
            f"got length {len(scale_granularity_mnk)}"
        )
    if tuple(scale_granularity_mnk) != (1, 128, 128):
        raise ValueError(
            f"scale_granularity_mnk must be (1, 128, 128) for the FP8 float-scale "
            f"kernel; got {tuple(scale_granularity_mnk)}"
        )


def _check_scale_major_mode_fp8(scale_major_mode: str) -> None:
    """Validate ``scale_major_mode`` for the FP8 float-scale cute SM120 entries.

    The kernel consumes MN-major float32 scales (zero-padding SFA ``[Kb, MpE]`` and
    compact SFB ``[num_experts, Kb, Nb]``); any value other than ``"MN"`` is rejected.
    """
    if scale_major_mode != "MN":
        raise ValueError(
            f'scale_major_mode must be "MN" (kernel currently only supports the '
            f'MN-major float32 scale layout); got "{scale_major_mode}"'
        )


@supported_compute_capability([120, 121])
@flashinfer_api
def moe_gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal["MN"] = "MN",
    backend: Literal["cute"] = "cute",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform grouped GEMM with FP8 inputs in zero-padding mode using groupwise
    float32 scaling. Currently only supported on NVIDIA RTX PRO 6000 Blackwell (SM120)
    architecture.

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
        Float32 scale tensor for ``a`` in zero-padding MN-major layout: contiguous shape
        ``(k_blocks, m_padded)`` where ``k_blocks = ceil(k / 128)`` and
        ``m_padded = (cum_m + num_experts * 3) // 4 * 4``. Expert ``i``'s scales start at
        column ``(m_indptr[i] + 3 * i) // 4 * 4``; padding columns are ignored by the
        kernel. The data pointer must be 16-byte aligned.

    b_scale: torch.Tensor
        Float32 scale tensor for ``b``, contiguous shape
        ``(num_experts, k_blocks, n_blocks)`` with ``n_blocks = ceil(n / 128)``
        (one scale per ``(128, 128)`` block).

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(num_experts + 1,)``, data type is
        ``torch.int32``. ``m_indptr[0] = 0``, ``m_indptr[num_experts] = cum_m``.

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensors. The FP8 float-scale kernel contract is
        fixed to ``(1, 128, 128)`` (per-token A-scale, ``(128, 128)``-block B-scale);
        anything else raises ``ValueError``.

    scale_major_mode: Literal["MN"]
        The layout mode of scale tensors. Currently only ``"MN"`` is supported.
        Defaults to ``"MN"``.

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
    - Unlike :func:`flashinfer.grouped_mm.moe_gemm_mxfp8_nt_groupwise` (int32-packed UE8M0
      scales, per-token along both M and N), the FP8 entry consumes plain float32 scales:
      per-token along M for ``a`` and one scale per ``(128, 128)`` block for ``b``.
    - A row-major ``(cum_m, k_blocks)`` per-token A-scale must be transposed and
      re-packed into the padded ``(k_blocks, m_padded)`` layout described above
      (zero-fill the padding columns; copy expert ``i``'s transposed scales to its
      4-row-aligned start column).
    """
    _check_scale_granularity_mnk_fp8(scale_granularity_mnk)
    _check_scale_major_mode_fp8(scale_major_mode)
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

    get_gemm_sm120_module_cute_fp8().moe_gemm_fp8_nt_groupwise(
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
