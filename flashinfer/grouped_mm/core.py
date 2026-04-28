"""Grouped Matrix Multiplication for Mixture-of-Experts (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each expert
in a Mixture-of-Experts layer has its own weight matrix and tokens are routed
to experts via ``m_indptr``.

This module is a backend-agnostic facade: it owns the public ``grouped_mm_*``
entry points, argument validation, and ``backend`` dispatch.  Backend-specific
implementations live in sibling subpackages (e.g. :mod:`.cudnn`).  Adding a
new backend means adding a new subpackage and wiring an extra ``elif`` branch
in the dispatch sections below.

Public APIs: ``grouped_mm_bf16``, ``grouped_mm_fp8``, ``grouped_mm_mxfp8``,
``grouped_mm_fp4``.
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..utils import backend_requirement, supported_compute_capability
from .cudnn import (
    _CUDNN_MOE_BLOCK_SCALE_MIN_VERSION,
    _CUDNN_MOE_MIN_VERSION,
    _check_cudnn_version,
    _run_cudnn_moe_block_scale_grouped_gemm_fp4,
    _run_cudnn_moe_block_scale_grouped_gemm_mxfp8,
    _run_cudnn_moe_grouped_gemm,
)

# =========================================================================
# grouped_mm_bf16
# =========================================================================


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 110, 120, 121])
def _check_grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_bf16)
@flashinfer_api
def grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with BF16/FP16 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_bf16` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, bf16 or fp16.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Examples
    --------
    >>> import torch, flashinfer
    >>> E, tpe, k, n = 8, 128, 4096, 2048
    >>> a = torch.randn(E * tpe, k, dtype=torch.bfloat16, device="cuda")
    >>> b = torch.randn(E, n, k, dtype=torch.bfloat16, device="cuda")
    >>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
    >>> out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr)
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, "grouped_mm_bf16")
        return _run_cudnn_moe_grouped_gemm(
            a, b, m_indptr, out_dtype=out_dtype, out=out, tactic=tactic
        )
    else:
        raise ValueError("backend does not support grouped_mm_bf16")


# =========================================================================
# grouped_mm_fp8
# =========================================================================


@supported_compute_capability([89, 90, 100, 103, 110, 120, 121])
def _check_grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"a must be float8_e4m3fn or float8_e5m2, got {a.dtype}")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"b must be float8_e4m3fn or float8_e5m2, got {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if alpha is not None:
        if alpha.dtype != torch.float32:
            raise ValueError(f"alpha must be float32, got {alpha.dtype}")
        if alpha.shape != (1,):
            raise ValueError(f"alpha must be a scalar, got {alpha.shape}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_fp8)
@flashinfer_api
def grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with FP8 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_fp8` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, e4m3 or e5m2.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    alpha : Optional[torch.Tensor]
        Scaling factor for the output, shape ``(1,)``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.
    tactic : int
        cuDNN execution plan index.  ``-1`` (default) uses the heuristic-best
        plan.  Non-negative values select a specific plan.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Examples
    --------
    >>> import torch, flashinfer
    >>> E, tpe, k, n = 8, 128, 4096, 2048
    >>> a = torch.randn(E * tpe, k, dtype=torch.float8_e4m3, device="cuda")
    >>> b = torch.randn(E, n, k, dtype=torch.float8_e4m3, device="cuda")
    >>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
    >>> out = flashinfer.grouped_mm.grouped_mm_fp8(a, b, m_indptr, alpha=torch.tensor(1.0, device="cuda"))
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, "grouped_mm_fp8")
        return _run_cudnn_moe_grouped_gemm(
            a,
            b,
            m_indptr,
            alpha=alpha,
            out_dtype=out_dtype,
            out=out,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp8")


# =========================================================================
# grouped_mm_mxfp8
# =========================================================================


@supported_compute_capability([100, 103, 110, 120, 121])
def _check_grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"a must be float8_e4m3fn or float8_e5m2, got {a.dtype}")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"b must be float8_e4m3fn or float8_e5m2, got {b.dtype}")
    if a_descale.dtype != torch.uint8:
        raise ValueError(f"a_descale must be uint8, got {a_descale.dtype}")
    if b_descale.dtype != torch.uint8:
        raise ValueError(f"b_descale must be uint8, got {b_descale.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if a_descale.ndim != 2:
        raise ValueError(
            f"a_descale must be 2D (cum_m, k // 32), got {a_descale.shape}"
        )
    if b_descale.ndim != 3:
        raise ValueError(
            f"b_descale must be 3D (num_experts, n, k // 32), got {b_descale.shape}"
        )
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_mxfp8)
@flashinfer_api
def grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with MXFP8 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_mxfp8` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, e4m3 or e5m2.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``, e4m3 or e5m2.
    a_descale : torch.Tensor
        Block scale tensor for A. Can be:
        - 2D swizzled 128x4: shape (cum_m, k // 32)
        dtype: uint8.
    b_descale : torch.Tensor
        Block scale tensor for B. Can be:
        - 3D swizzled 128x4: shape (batch_size, n, k // 32)
        dtype: uint8.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_BLOCK_SCALE_MIN_VERSION, "grouped_mm_mxfp8")
        return _run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
            a,
            b,
            a_descale,
            b_descale,
            m_indptr,
            out_dtype=out_dtype,
            out=out,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_mxfp8")


# =========================================================================
# grouped_mm_fp4
# =========================================================================


@supported_compute_capability([100, 103, 110, 120, 121])
def _check_grouped_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise ValueError(f"a must be float4_e2m1fn_x2 or uint8, got {a.dtype}")
    if b.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise ValueError(f"b must be float4_e2m1fn_x2 or uint8, got {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if a_descale.ndim != 2:
        raise ValueError(
            f"a_descale must be 2D (cum_m, k // 32), got {a_descale.shape}"
        )
    if b_descale.ndim != 3:
        raise ValueError(
            f"b_descale must be 3D (num_experts, n, k // 32), got {b_descale.shape}"
        )
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if alpha is not None:
        if alpha.dtype != torch.float32:
            raise ValueError(f"alpha must be float32, got {alpha.dtype}")
        if alpha.shape != (1,):
            raise ValueError(f"alpha must be a scalar, got {alpha.shape}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    if block_size == 16:  # nvfp4
        if (
            a_descale.dtype != torch.float8_e4m3fn
            or b_descale.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(
                f"a_descale and b_descale must be float8_e4m3fn, got {a_descale.dtype} and {b_descale.dtype}"
            )
    elif block_size == 32:  # mxfp4
        if a_descale.dtype != torch.uint8 or b_descale.dtype != torch.uint8:
            raise ValueError(
                f"a_descale and b_descale must be uint8, got {a_descale.dtype} and {b_descale.dtype}"
            )
    else:
        raise ValueError(f"block_size must be 16 or 32, got {block_size}")

    return True


@backend_requirement({}, common_check=_check_grouped_mm_fp4)
@flashinfer_api
def grouped_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with FP4 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_fp4` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, fp4 e2m1fn_x2 or uint8.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``, fp4 e2m1fn_x2 or uint8.
    a_descale : torch.Tensor
        Block scale tensor for A. Can be:
        - 2D swizzled 128x4: shape (cum_m, k // block_size)
        dtype: float8_e4m3fn or uint8.
    b_descale : torch.Tensor
        Block scale tensor for B. Can be:
        - 3D swizzled 128x4: shape (batch_size, n, k // block_size)
        dtype: float8_e4m3fn or uint8.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    alpha : Optional[torch.Tensor]
        Scaling factor for the output, shape ``(1,)``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_BLOCK_SCALE_MIN_VERSION, "grouped_mm_fp4")
        return _run_cudnn_moe_block_scale_grouped_gemm_fp4(
            a,
            b,
            a_descale,
            b_descale,
            m_indptr,
            alpha=alpha,
            out_dtype=out_dtype,
            out=out,
            block_size=block_size,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp4")
