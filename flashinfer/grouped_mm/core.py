"""
cuDNN-backed Grouped Matrix Multiplication (MoE Grouped GEMM).

Provides ``grouped_mm_bf16`` as a public API, mirroring the ``mm_bf16`` dense
GEMM API but operating on expert-partitioned (grouped) inputs.

All implementations use cuDNN's ``moe_grouped_matmul`` graph node under the
hood with automatic graph caching.
"""

import functools
from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
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
_CUDNN_MOE_FP4_MIN_VERSION = 92100  # 9.21.0


class _UIDs(Enum):
    """Tensor UIDs for cuDNN MOE graphs (unique namespace)."""

    TOKEN = 20
    WEIGHT = 21
    FIRST_TOKEN_OFFSET = 22
    A_DESCALE = 25
    B_DESCALE = 26
    OUTPUT = 27


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
        uid=_UIDs.TOKEN.value,
    )
    weight = graph.tensor(
        name="weight",
        dim=list(weight_shape),
        stride=list(weight_stride),
        data_type=weight_cudnn_dtype,
        uid=_UIDs.WEIGHT.value,
    )
    fto = graph.tensor(
        name="first_token_offset",
        dim=list(fto_shape),
        stride=list(fto_stride),
        data_type=cudnn.data_type.INT32,
        uid=_UIDs.FIRST_TOKEN_OFFSET.value,
    )

    output = graph.moe_grouped_matmul(
        token=token,
        weight=weight,
        first_token_offset=fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="cudnn_moe_grouped_gemm",
    )

    output.set_uid(_UIDs.OUTPUT.value).set_output(True).set_data_type(
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
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
):
    """Build/cache graph → allocate output → execute."""
    token_cudnn_dtype = _to_cudnn_dtype(a.dtype)
    weight_cudnn_dtype = _to_cudnn_dtype(b.dtype)
    out_cudnn_dtype = _to_cudnn_dtype(out_dtype)

    cum_m = a.shape[0]
    _, n, _ = b.shape

    token_3d = a.unsqueeze(0)
    weight_3d = b.transpose(1, 2)
    fto = m_indptr[:-1].reshape(-1, 1, 1).contiguous()

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
        out_cudnn_dtype,
    )

    if out is None:
        out = torch.empty(cum_m, n, dtype=out_dtype, device=a.device)
    out_3d = out.view(1, cum_m, n)

    variant_pack = {
        _UIDs.TOKEN.value: token_3d,
        _UIDs.WEIGHT.value: weight_3d,
        _UIDs.FIRST_TOKEN_OFFSET.value: fto,
        _UIDs.OUTPUT.value: out_3d,
    }

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
    _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, "grouped_mm_bf16")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"a must be float16 or bfloat16, got {a.dtype}")
    if b.dtype != a.dtype:
        raise ValueError(f"b dtype {b.dtype} must match a dtype {a.dtype}")
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
        Expert weights, shape ``(num_experts, n, k)``.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(num_experts + 1,)``, ``int32``.
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
        return _run_cudnn_moe_grouped_gemm(a, b, m_indptr, out_dtype, out)
    else:
        raise ValueError("backend does not support grouped_mm_bf16")
