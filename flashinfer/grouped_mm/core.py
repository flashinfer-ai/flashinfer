"""
Grouped Matrix Multiplication for Mixture-of-Experts (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each expert
in a Mixture-of-Experts layer has its own weight matrix and tokens are routed
to experts via ``m_indptr``.

Public APIs: ``grouped_mm_bf16``, ``grouped_mm_fp8``, ``grouped_mm_mxfp8``,
``grouped_mm_fp4``.
"""

import functools
from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..gemm.gemm_base import _get_real_fp4_shape_from_packed_uint8
from ..utils import (
    backend_requirement,
    supported_compute_capability,
)

# ---------------------------------------------------------------------------
# Optional cuDNN import
# ---------------------------------------------------------------------------

_CUDNN_AVAILABLE = False
try:
    import cudnn

    _CUDNN_AVAILABLE = True
except ImportError:
    pass
except OSError as e:
    if any(ext in str(e).lower() for ext in [".so", ".dll"]):
        pass
    else:
        raise


# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------

_CUDNN_MOE_MIN_VERSION = 91800  # 9.18.0
_CUDNN_MOE_BLOCK_SCALE_MIN_VERSION = 92100  # 9.21.0


class _CUDNN_UIDs(Enum):
    """Tensor UIDs for cuDNN MOE graphs (unique namespace)."""

    TOKEN = 20
    WEIGHT = 21
    FIRST_TOKEN_OFFSET = 22
    TOKEN_SCALE_FACTOR = 25
    WEIGHT_SCALE_FACTOR = 26
    ALPHA = 27
    OUTPUT = 100


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_cudnn_handles: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cudnn():
    if not _CUDNN_AVAILABLE:
        raise RuntimeError(
            "cuDNN is not available.  Install with: "
            "pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )


def _check_cudnn_version(min_ver: int, feature: str):
    _check_cudnn()
    ver = cudnn.backend_version()
    if ver < min_ver:
        raise RuntimeError(
            f"cuDNN {feature} requires backend version >= {min_ver}, found {ver}."
        )


def _get_handle(device: torch.device):
    device = torch.device(device)
    key = (device.type, device.index)
    if key not in _cudnn_handles:
        _check_cudnn()
        _cudnn_handles[key] = cudnn.create_handle()
    stream = torch.cuda.current_stream(device).cuda_stream
    cudnn.set_stream(handle=_cudnn_handles[key], stream=stream)
    return _cudnn_handles[key]


_TORCH_TO_CUDNN = None


def _to_cudnn_dtype(dtype: torch.dtype):
    global _TORCH_TO_CUDNN
    if _TORCH_TO_CUDNN is None:
        _TORCH_TO_CUDNN = {
            torch.float16: cudnn.data_type.HALF,
            torch.bfloat16: cudnn.data_type.BFLOAT16,
            torch.float32: cudnn.data_type.FLOAT,
            torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
            torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
            torch.float8_e8m0fnu: cudnn.data_type.FP8_E8M0,
        }
    return _TORCH_TO_CUDNN[dtype]


# ---------------------------------------------------------------------------
# Generic cuDNN MOE graph builder & executor
# ---------------------------------------------------------------------------


@functools.cache
def _build_cudnn_moe_grouped_gemm_graph(
    handle,
    token_shape,
    token_stride,
    token_cudnn_dtype,
    weight_shape,
    weight_stride,
    weight_cudnn_dtype,
    fto_shape,
    fto_stride,
    alpha_cudnn_dtype,
    output_cudnn_dtype,
    policy=None,
):
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    token = graph.tensor(
        name="token",
        dim=list(token_shape),
        stride=list(token_stride),
        data_type=token_cudnn_dtype,
        uid=_CUDNN_UIDs.TOKEN.value,
    )
    weight = graph.tensor(
        name="weight",
        dim=list(weight_shape),
        stride=list(weight_stride),
        data_type=weight_cudnn_dtype,
        uid=_CUDNN_UIDs.WEIGHT.value,
    )
    fto = graph.tensor(
        name="first_token_offset",
        dim=list(fto_shape),
        stride=list(fto_stride),
        data_type=cudnn.data_type.INT32,
        uid=_CUDNN_UIDs.FIRST_TOKEN_OFFSET.value,
    )

    accum = graph.moe_grouped_matmul(
        name="cudnn_moe_grouped_gemm",
        token=token,
        weight=weight,
        first_token_offset=fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    if alpha_cudnn_dtype is not None:
        alpha = graph.tensor(
            name="alpha",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=alpha_cudnn_dtype,
            uid=_CUDNN_UIDs.ALPHA.value,
        )
        output = graph.mul(
            name="scale",
            a=accum,
            b=alpha,
            compute_data_type=cudnn.data_type.FLOAT,
        )
    else:
        output = accum

    output.set_uid(_CUDNN_UIDs.OUTPUT.value).set_output(True).set_data_type(
        output_cudnn_dtype
    )

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(policy)

    return graph, graph.get_workspace_size()


def _run_cudnn_moe_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
):
    """Build/cache graph → allocate output → execute."""
    token_cudnn_dtype = _to_cudnn_dtype(a.dtype)
    weight_cudnn_dtype = _to_cudnn_dtype(b.dtype)
    alpha_cudnn_dtype = _to_cudnn_dtype(alpha.dtype) if alpha is not None else None
    out_cudnn_dtype = _to_cudnn_dtype(out_dtype)

    cum_m = a.shape[0]
    _, n, _ = b.shape

    token_3d = a.unsqueeze(0)
    weight_3d = b.transpose(1, 2)
    fto = m_indptr[:-1].reshape(-1, 1, 1).contiguous()
    alpha_3d = alpha.unsqueeze(0).unsqueeze(0) if alpha is not None else None

    handle = _get_handle(a.device)

    graph, ws = _build_cudnn_moe_grouped_gemm_graph(
        handle,
        token_3d.shape,
        token_3d.stride(),
        token_cudnn_dtype,
        weight_3d.shape,
        weight_3d.stride(),
        weight_cudnn_dtype,
        fto.shape,
        fto.stride(),
        alpha_cudnn_dtype,
        out_cudnn_dtype,
    )

    if out is None:
        out = torch.empty(cum_m, n, dtype=out_dtype, device=a.device)
    out_3d = out.view(1, cum_m, n)

    variant_pack = {
        _CUDNN_UIDs.TOKEN.value: token_3d,
        _CUDNN_UIDs.WEIGHT.value: weight_3d,
        _CUDNN_UIDs.FIRST_TOKEN_OFFSET.value: fto,
        _CUDNN_UIDs.OUTPUT.value: out_3d,
    }
    if alpha is not None:
        variant_pack[_CUDNN_UIDs.ALPHA.value] = alpha_3d

    workspace = torch.empty(ws, device=a.device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=handle)
    return out


@functools.cache
def _build_cudnn_moe_block_scale_grouped_gemm_graph(
    handle,
    token_shape,
    token_stride,
    token_cudnn_dtype,
    token_descale_shape,
    token_descale_stride,
    token_descale_cudnn_dtype,
    weight_shape,
    weight_stride,
    weight_cudnn_dtype,
    weight_descale_shape,
    weight_descale_stride,
    weight_descale_cudnn_dtype,
    fto_shape,
    fto_stride,
    alpha_cudnn_dtype,
    output_cudnn_dtype,
    block_size,
    policy=None,
):
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    token = graph.tensor(
        name="token",
        dim=list(token_shape),
        stride=list(token_stride),
        data_type=token_cudnn_dtype,
        uid=_CUDNN_UIDs.TOKEN.value,
    )
    token_descale = graph.tensor(
        name="token_descale",
        dim=list(token_descale_shape),
        stride=list(token_descale_stride),
        data_type=token_descale_cudnn_dtype,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
        uid=_CUDNN_UIDs.TOKEN_SCALE_FACTOR.value,
    )
    token_dequant = graph.block_scale_dequantize(
        token,
        token_descale,
        block_size=[1, block_size],
        name="token_dequant",
    )

    weight = graph.tensor(
        name="weight",
        dim=list(weight_shape),
        stride=list(weight_stride),
        data_type=weight_cudnn_dtype,
        uid=_CUDNN_UIDs.WEIGHT.value,
    )
    weight_descale = graph.tensor(
        name="weight_descale",
        dim=list(weight_descale_shape),
        stride=list(weight_descale_stride),
        data_type=weight_descale_cudnn_dtype,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
        uid=_CUDNN_UIDs.WEIGHT_SCALE_FACTOR.value,
    )
    weight_dequant = graph.block_scale_dequantize(
        weight,
        weight_descale,
        block_size=[block_size, 1],
        name="weight_dequant",
    )

    fto = graph.tensor(
        name="first_token_offset",
        dim=list(fto_shape),
        stride=list(fto_stride),
        data_type=cudnn.data_type.INT32,
        uid=_CUDNN_UIDs.FIRST_TOKEN_OFFSET.value,
    )

    accum = graph.moe_grouped_matmul(
        name="cudnn_moe_grouped_gemm",
        token=token_dequant,
        weight=weight_dequant,
        first_token_offset=fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    if alpha_cudnn_dtype is not None:
        alpha = graph.tensor(
            name="alpha",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=alpha_cudnn_dtype,
            uid=_CUDNN_UIDs.ALPHA.value,
        )
        output = graph.mul(
            name="scale",
            a=accum,
            b=alpha,
            compute_data_type=cudnn.data_type.FLOAT,
        )
    else:
        output = accum

    output.set_uid(_CUDNN_UIDs.OUTPUT.value).set_output(True).set_data_type(
        output_cudnn_dtype
    )

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(policy)

    return graph, graph.get_workspace_size()


def _run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
):
    """Build/cache graph → allocate output → execute."""
    token_cudnn_dtype = _to_cudnn_dtype(a.dtype)
    weight_cudnn_dtype = _to_cudnn_dtype(b.dtype)
    alpha_cudnn_dtype = _to_cudnn_dtype(alpha.dtype) if alpha is not None else None
    out_cudnn_dtype = _to_cudnn_dtype(out_dtype)

    cum_m = a.shape[0]
    _, n, _ = b.shape

    token_3d = a.unsqueeze(0)
    token_descale_3d = a_descale.unsqueeze(0)
    weight_3d = b.transpose(1, 2)
    weight_descale_3d = b_descale.transpose(1, 2)
    fto = m_indptr[:-1].reshape(-1, 1, 1).contiguous()
    alpha_3d = alpha.unsqueeze(0).unsqueeze(0) if alpha is not None else None

    handle = _get_handle(a.device)

    graph, ws = _build_cudnn_moe_block_scale_grouped_gemm_graph(
        handle,
        token_3d.shape,
        token_3d.stride(),
        token_cudnn_dtype,
        token_descale_3d.shape,
        token_descale_3d.stride(),
        cudnn.data_type.FP8_E8M0,
        weight_3d.shape,
        weight_3d.stride(),
        weight_cudnn_dtype,
        weight_descale_3d.shape,
        weight_descale_3d.stride(),
        cudnn.data_type.FP8_E8M0,
        fto.shape,
        fto.stride(),
        alpha_cudnn_dtype,
        out_cudnn_dtype,
        block_size=32,
    )

    if out is None:
        out = torch.empty(cum_m, n, dtype=out_dtype, device=a.device)
    out_3d = out.view(1, cum_m, n)

    variant_pack = {
        _CUDNN_UIDs.TOKEN.value: token_3d,
        _CUDNN_UIDs.TOKEN_SCALE_FACTOR.value: token_descale_3d,
        _CUDNN_UIDs.WEIGHT.value: weight_3d,
        _CUDNN_UIDs.WEIGHT_SCALE_FACTOR.value: weight_descale_3d,
        _CUDNN_UIDs.FIRST_TOKEN_OFFSET.value: fto,
        _CUDNN_UIDs.OUTPUT.value: out_3d,
    }
    if alpha is not None:
        variant_pack[_CUDNN_UIDs.ALPHA.value] = alpha_3d

    workspace = torch.empty(ws, device=a.device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=handle)
    return out


def _run_cudnn_moe_block_scale_grouped_gemm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
):
    """Build/cache graph → allocate output → execute."""
    a_descale_cudnn_type = (
        cudnn.data_type.FP8_E8M0
        if a_descale.dtype == torch.uint8
        else _to_cudnn_dtype(a_descale.dtype)
    )
    b_descale_cudnn_type = (
        cudnn.data_type.FP8_E8M0
        if b_descale.dtype == torch.uint8
        else _to_cudnn_dtype(b_descale.dtype)
    )
    alpha_cudnn_dtype = _to_cudnn_dtype(alpha.dtype) if alpha is not None else None
    out_cudnn_dtype = _to_cudnn_dtype(out_dtype)

    cum_m = a.shape[0]
    _, n, _ = b.shape

    token_3d = a.unsqueeze(0)
    token_descale_3d = a_descale.unsqueeze(0)
    weight_3d = b.transpose(1, 2)
    weight_descale_3d = b_descale.transpose(1, 2)
    fto = m_indptr[:-1].reshape(-1, 1, 1).contiguous()
    alpha_3d = alpha.unsqueeze(0).unsqueeze(0) if alpha is not None else None

    token_logical_shape, token_logical_stride = _get_real_fp4_shape_from_packed_uint8(
        token_3d
    )
    weight_logical_shape, weight_logical_stride = _get_real_fp4_shape_from_packed_uint8(
        weight_3d
    )

    handle = _get_handle(a.device)

    graph, ws = _build_cudnn_moe_block_scale_grouped_gemm_graph(
        handle,
        token_logical_shape,
        token_logical_stride,
        cudnn.data_type.FP4_E2M1,
        token_descale_3d.shape,
        token_descale_3d.stride(),
        a_descale_cudnn_type,
        weight_logical_shape,
        weight_logical_stride,
        cudnn.data_type.FP4_E2M1,
        weight_descale_3d.shape,
        weight_descale_3d.stride(),
        b_descale_cudnn_type,
        fto.shape,
        fto.stride(),
        alpha_cudnn_dtype,
        out_cudnn_dtype,
        block_size=block_size,
    )

    if out is None:
        out = torch.empty(cum_m, n, dtype=out_dtype, device=a.device)
    out_3d = out.view(1, cum_m, n)

    variant_pack = {
        _CUDNN_UIDs.TOKEN.value: token_3d,
        _CUDNN_UIDs.TOKEN_SCALE_FACTOR.value: token_descale_3d,
        _CUDNN_UIDs.WEIGHT.value: weight_3d,
        _CUDNN_UIDs.WEIGHT_SCALE_FACTOR.value: weight_descale_3d,
        _CUDNN_UIDs.FIRST_TOKEN_OFFSET.value: fto,
        _CUDNN_UIDs.OUTPUT.value: out_3d,
    }
    if alpha is not None:
        variant_pack[_CUDNN_UIDs.ALPHA.value] = alpha_3d

    workspace = torch.empty(ws, device=a.device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=handle)
    return out


# =========================================================================
# grouped_mm_bf16
# =========================================================================


@supported_compute_capability([80, 86, 89, 90, 100, 103, 120, 121])
def _check_grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
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
        return _run_cudnn_moe_grouped_gemm(a, b, m_indptr, out_dtype=out_dtype, out=out)
    else:
        raise ValueError("backend does not support grouped_mm_bf16")


# =========================================================================
# grouped_mm_fp8
# =========================================================================


@supported_compute_capability([89, 90, 100, 103, 120, 121])
def _check_grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
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
            a, b, m_indptr, alpha=alpha, out_dtype=out_dtype, out=out
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp8")


# =========================================================================
# grouped_mm_mxfp8
# =========================================================================


@supported_compute_capability([100, 103, 120, 121])
def _check_grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cudnn",
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
            a, b, a_descale, b_descale, m_indptr, out_dtype=out_dtype, out=out
        )
    else:
        raise ValueError("backend does not support grouped_mm_mxfp8")


# =========================================================================
# grouped_mm_fp4
# =========================================================================


@supported_compute_capability([100, 103, 120, 121])
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
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp4")
