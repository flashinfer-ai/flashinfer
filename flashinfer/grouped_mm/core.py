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

import functools
from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.grouped_mm import gen_grouped_mm_sm120_module_cute_mxfp8
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
    *,
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
    *,
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
    *,
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
    *,
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
    *,
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
    *,
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
    *,
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
    *,
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


# =========================================================================
# grouped_mm_mxfp8_nt_groupwise_zero_padding (SM120 cute backend)
# =========================================================================


@functools.cache
def get_grouped_mm_sm120_module_cute_mxfp8():
    """MXFP8 grouped MM module accessor for SM120 cute backend."""
    return gen_grouped_mm_sm120_module_cute_mxfp8().build_and_load()


def _check_m_indptr(m_indptr: torch.Tensor, token_num: int) -> None:
    """Validate ``m_indptr`` semantic consistency before launching the kernel.

    Checks: 1-D int32, first element 0, monotonic non-decreasing, last element
    equals ``token_num`` (the row count of the packed A tensor).
    """
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32; got {m_indptr.dtype}")
    if m_indptr.dim() != 1:
        raise ValueError(f"m_indptr must be 1-D; got {m_indptr.dim()}D")
    if m_indptr.numel() < 2:
        raise ValueError(
            f"m_indptr must have at least 2 elements (num_experts >= 1); "
            f"got numel={m_indptr.numel()}"
        )
    first = int(m_indptr[0].item())
    last = int(m_indptr[-1].item())
    if first != 0:
        raise ValueError(f"m_indptr[0] must be 0; got {first}")
    if last != token_num:
        raise ValueError(f"m_indptr[-1] must equal token_num={token_num}; got {last}")
    if not bool((m_indptr[1:] >= m_indptr[:-1]).all().item()):
        raise ValueError("m_indptr must be non-decreasing")


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
            f"scale_granularity_mnk[2] (k_gran) must be 32 or 128; got {scale_granularity_mnk[2]}"
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
def quantize_mxfp8_for_zero_padding(
    input: torch.Tensor,
    m_indptr: torch.Tensor,
    gran_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-token MXFP8 quantization for ``grouped_mm_mxfp8_nt_groupwise_zero_padding``.

    Produces a token-packed FP8 output and a 4-row per-expert padded int32-packed UE8M0
    scale tensor matching the input layout expected by
    ``grouped_mm_mxfp8_nt_groupwise_zero_padding``.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor shape ``(token_num, k)``, data type is ``torch.bfloat16``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(num_experts + 1,)``, data type is
        ``torch.int32``. ``m_indptr[0] = 0``, ``m_indptr[num_experts] = token_num``.

    gran_k: int
        UE8M0 K-axis block granularity. Must be ``32`` or ``128``.

    Returns
    -------
    out_fp8: torch.Tensor
        Token-packed FP8 output, shape ``(token_num, k)``, data type is
        ``torch.float8_e4m3fn``.

    out_scale: torch.Tensor
        Int32-packed UE8M0 scale tensor (4 UE8M0 scales packed per int32), shape
        ``(m_padded, k_align)`` where ``m_padded = (token_num + num_experts * 3) // 4 * 4``
        and ``k_align = (k + 4 * gran_k - 1) // (4 * gran_k)``. Data type is ``torch.int32``.
        Underlying storage is ``(k_align, m_padded)`` contiguous; the returned view is a
        ``.transpose(0, 1)`` that aligns with the
        ``grouped_mm_mxfp8_nt_groupwise_zero_padding`` caller convention.
    """
    if gran_k not in (32, 128):
        raise ValueError(f"gran_k must be 32 or 128; got {gran_k}")
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be bfloat16; got {input.dtype}")
    if input.dim() != 2:
        raise ValueError(f"input must be 2D; got {input.dim()}D")

    token_num, k = input.shape
    _check_m_indptr(m_indptr, token_num=token_num)
    num_experts = m_indptr.shape[0] - 1
    if k % 16 != 0:
        raise ValueError(f"k must be multiple of 16; got k={k}")

    pack_nk = gran_k * 4
    m_padded = (token_num + num_experts * 3) // 4 * 4
    k_align = (k + pack_nk - 1) // pack_nk

    out_fp8 = torch.empty(
        (token_num, k), dtype=torch.float8_e4m3fn, device=input.device
    )
    out_scale_raw = torch.zeros(
        (k_align, m_padded), dtype=torch.int32, device=input.device
    )

    get_grouped_mm_sm120_module_cute_mxfp8().quantize_mxfp8_for_zero_padding(
        input, m_indptr, out_fp8, out_scale_raw, gran_k
    )

    out_scale = out_scale_raw.transpose(0, 1)
    return out_fp8, out_scale


@supported_compute_capability([120, 121])
@flashinfer_api
def grouped_mm_mxfp8_nt_groupwise_zero_padding(
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
    _check_m_indptr(m_indptr, token_num=a.shape[0])
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

    get_grouped_mm_sm120_module_cute_mxfp8().group_gemm_mxfp8_nt_groupwise_zero_padding(
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
