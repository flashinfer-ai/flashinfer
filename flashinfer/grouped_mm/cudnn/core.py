"""cuDNN backend for Grouped Matrix Multiplication (MoE Grouped GEMM).

This module owns every cuDNN-specific concern for the grouped GEMM family:

* Optional ``cudnn`` import + capability/version probes.
* cuDNN handle / dtype-mapping caches.
* Graph builders (cached with :func:`functools.lru_cache`) for the plain and
  block-scaled MOE grouped matmul variants.
* Runners that wire PyTorch tensors into the cuDNN graphs and execute them.

The public ``grouped_mm_*`` entry points in :mod:`..core` dispatch into the
``_run_*`` helpers exported here.  Anything cuDNN-specific should live in this
file (or sibling files in this subpackage); ``..core`` must remain
backend-agnostic.
"""

import functools
from enum import Enum
from typing import Optional

import torch

from ...gemm.gemm_base import _get_real_fp4_shape_from_packed_uint8
from ...utils import _get_cache_buf

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
    # The cuDNN Python frontend package version can lag the backend lib, so
    # `backend_version() >= min_ver` is necessary but not sufficient. Probe the
    # actual symbol used by the MOE graph builders to fail fast with a clear
    # message instead of an opaque AttributeError deep in the graph code.
    if not hasattr(cudnn, "moe_grouped_matmul_mode"):
        raise RuntimeError(
            f"cuDNN {feature} requires a frontend exposing "
            "`cudnn.moe_grouped_matmul_mode`, but it is missing from the "
            "installed `cudnn` Python package. Upgrade with: "
            "pip install -U nvidia-cudnn-frontend"
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


@functools.lru_cache(maxsize=1024)
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
    tactic: int = -1,
):
    """Build/cache graph → allocate output → execute.

    Args:
        tactic: Execution plan index. -1 (default) uses the heuristic-best
            plan.  Non-negative values select a specific plan built with
            ``build_plan_policy.ALL``.
    """
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

    policy = (
        cudnn.build_plan_policy.HEURISTICS_CHOICE
        if tactic == -1
        else cudnn.build_plan_policy.ALL
    )

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
        policy=policy,
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

    workspace = _get_cache_buf("grouped_mm_workspace", ws, a.device)
    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=handle)
    else:
        plan_count = graph.get_execution_plan_count()
        if tactic >= plan_count:
            return None
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=handle)
    return out


@functools.lru_cache(maxsize=1024)
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
    tactic: int = -1,
):
    """Build/cache graph → allocate output → execute.

    Args:
        tactic: Execution plan index. -1 (default) uses the heuristic-best
            plan.  Non-negative values select a specific plan built with
            ``build_plan_policy.ALL``.
    """
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

    policy = (
        cudnn.build_plan_policy.HEURISTICS_CHOICE
        if tactic == -1
        else cudnn.build_plan_policy.ALL
    )

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
        policy=policy,
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

    workspace = _get_cache_buf("grouped_mm_mxfp8_workspace", ws, a.device)
    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=handle)
    else:
        plan_count = graph.get_execution_plan_count()
        if tactic >= plan_count:
            return None
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=handle)
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
    tactic: int = -1,
):
    """Build/cache graph → allocate output → execute.

    Args:
        tactic: Execution plan index. -1 (default) uses the heuristic-best
            plan.  Non-negative values select a specific plan built with
            ``build_plan_policy.ALL``.
    """
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

    policy = (
        cudnn.build_plan_policy.HEURISTICS_CHOICE
        if tactic == -1
        else cudnn.build_plan_policy.ALL
    )

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
        policy=policy,
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

    workspace = _get_cache_buf("grouped_mm_fp4_workspace", ws, a.device)
    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=handle)
    else:
        plan_count = graph.get_execution_plan_count()
        if tactic >= plan_count:
            return None
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=handle)
    return out
