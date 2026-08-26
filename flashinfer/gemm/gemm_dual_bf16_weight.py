"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.gemm import gen_dual_bf16_weight_gemm_sm100_module
from ..trace.templates.gemm import mm_bf16_dual_weight_trace
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    get_device_index,
    supported_compute_capability,
)


@functools.cache
def _get_dual_bf16_weight_gemm_module():
    return gen_dual_bf16_weight_gemm_sm100_module().build_and_load()


def _normalize_cuda_device(
    device: Optional[Union[torch.device, str, int]],
) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    normalized = torch.device(device)
    if normalized.type != "cuda":
        raise ValueError(f"device must be CUDA; got {normalized}")
    if normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def _require_exact_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise ValueError(
            "dual BF16 weight GEMM requires an exact SM100 "
            f"(compute capability 10.0) device; got {capability[0]}.{capability[1]}"
        )


@functools.cache
def _workspace_size_cached(device_index: int, m: int, n: int, k: int) -> int:
    module = _get_dual_bf16_weight_gemm_module()
    return int(module.workspace_size(m, n, k, device_index))


def dual_bf16_weight_gemm_workspace_size(
    m: int,
    n: int,
    k: int,
    device: Optional[Union[torch.device, str, int]] = None,
) -> int:
    """Return the caller-owned workspace size required by the GEMM, in bytes.

    The split-K path uses this memory for FP32 partial outputs and int32 tile
    counters. Persistent 1SM and cluster 2SM paths return zero.
    """

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M, N, and K must be positive; got M={m}, N={n}, K={k}")
    if k % 128 != 0:
        raise ValueError(f"K must be a multiple of 128; got K={k}")
    normalized = _normalize_cuda_device(device)
    _require_exact_sm100(normalized)
    return _workspace_size_cached(get_device_index(normalized), int(m), int(n), int(k))


def _dual_bf16_weight_gemm_kernel_kind(
    m: int,
    n: int,
    k: int,
    device: Optional[Union[torch.device, str, int]] = None,
) -> int:
    """Return internal dispatch kind: 0=split-K, 1=1SM, 2=2SM."""

    normalized = _normalize_cuda_device(device)
    _require_exact_sm100(normalized)
    module = _get_dual_bf16_weight_gemm_module()
    return int(module.kernel_kind(m, n, k, get_device_index(normalized)))


@flashinfer_api
@torch.no_grad()
def prepare_dual_bf16_weights(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an FP32 [N, K] weight into high and residual BF16 tensors.

    The compute kernel reconstructs the runtime weight as
    weight_high.float() + weight_low.float() / 256. Call this helper once
    when loading weights, then reuse both returned contiguous tensors.
    """

    if weight.dim() != 2:
        raise ValueError(
            f"weight must be 2-D with shape [N, K]; got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.float32:
        raise TypeError(f"weight must be float32; got {weight.dtype}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError(
            f"weight dimensions must be positive; got {tuple(weight.shape)}"
        )
    if weight.shape[1] % 128 != 0:
        raise ValueError(f"weight K must be a multiple of 128; got K={weight.shape[1]}")

    weight_contiguous = weight.contiguous()
    weight_high = weight_contiguous.to(torch.bfloat16)
    weight_low = ((weight_contiguous - weight_high.to(torch.float32)) * 256.0).to(
        torch.bfloat16
    )
    return weight_high.contiguous(), weight_low.contiguous()


@supported_compute_capability([100])
def _check_mm_bf16_dual_weight(
    a: torch.Tensor,
    weight_high: torch.Tensor,
    weight_low: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
) -> bool:
    if a.dim() != 2:
        raise ValueError(f"a must be 2-D with shape [M, K]; got {tuple(a.shape)}")
    if weight_high.dim() != 2 or weight_low.dim() != 2:
        raise ValueError(
            "weight_high and weight_low must both be 2-D with shape [N, K]"
        )
    if a.dtype != torch.bfloat16:
        raise TypeError(f"a must be bfloat16; got {a.dtype}")
    if weight_high.dtype != torch.bfloat16 or weight_low.dtype != torch.bfloat16:
        raise TypeError(
            "weight_high and weight_low must both have dtype torch.bfloat16"
        )
    if not a.is_cuda or not weight_high.is_cuda or not weight_low.is_cuda:
        raise ValueError("a, weight_high, and weight_low must be CUDA tensors")
    if a.device != weight_high.device or a.device != weight_low.device:
        raise ValueError("a, weight_high, and weight_low must be on the same device")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous")
    if not weight_high.is_contiguous() or not weight_low.is_contiguous():
        raise ValueError("weight_high and weight_low must be contiguous")
    if weight_high.shape != weight_low.shape:
        raise ValueError("weight_high and weight_low must have identical [N, K] shapes")

    m, k = a.shape
    n, weight_k = weight_high.shape
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M, N, and K must be positive; got M={m}, N={n}, K={k}")
    if weight_k != k:
        raise ValueError(f"activation K={k} does not match weight K={weight_k}")
    if k % 128 != 0:
        raise ValueError(f"K must be a multiple of 128; got K={k}")

    resolved_out_dtype = (
        out_dtype
        if out_dtype is not None
        else (out.dtype if out is not None else torch.bfloat16)
    )
    if resolved_out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"out_dtype must be torch.bfloat16 or torch.float32; got {resolved_out_dtype}"
        )
    if out is not None:
        if not out.is_cuda or out.device != a.device:
            raise ValueError("out must be a CUDA tensor on the same device as a")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.shape != (m, n):
            raise ValueError(f"out must have shape {(m, n)}; got {tuple(out.shape)}")
        if out.dtype != resolved_out_dtype:
            raise TypeError(
                f"out dtype {out.dtype} does not match out_dtype {resolved_out_dtype}"
            )

    if workspace_buffer is not None:
        if not workspace_buffer.is_cuda or workspace_buffer.device != a.device:
            raise ValueError(
                "workspace_buffer must be a CUDA tensor on the same device as a"
            )
        if workspace_buffer.dtype != torch.uint8:
            raise TypeError(
                f"workspace_buffer must have dtype torch.uint8; got {workspace_buffer.dtype}"
            )
        if not workspace_buffer.is_contiguous():
            raise ValueError("workspace_buffer must be contiguous")

    return True


@backend_requirement(backend_checks={}, common_check=_check_mm_bf16_dual_weight)
@flashinfer_api(trace=mm_bf16_dual_weight_trace)
def mm_bf16_dual_weight(
    a: torch.Tensor,
    weight_high: torch.Tensor,
    weight_low: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a @ (weight_high + weight_low / 256).T on SM100.

    Args:
        a: Contiguous BF16 activation with shape [M, K].
        weight_high: Contiguous BF16 high component with shape [N, K].
        weight_low: Contiguous BF16 residual component with shape [N, K].
        out_dtype: torch.bfloat16 (default) or torch.float32.
        out: Optional contiguous preallocated output with shape [M, N].
        workspace_buffer: Optional contiguous CUDA uint8 workspace. Use
            dual_bf16_weight_gemm_workspace_size to size it. Supplying
            independent buffers is recommended for concurrent work. Its base
            address must be 16-byte aligned; ordinary torch allocations are.

    Returns:
        The [M, N] output tensor.

    Note:
        When workspace_buffer is omitted, FlashInfer caches one workspace per
        CUDA stream and device. CUDA Graph callers should pass an explicit
        workspace allocated before capture.
    """

    resolved_out_dtype = (
        out_dtype
        if out_dtype is not None
        else (out.dtype if out is not None else torch.bfloat16)
    )
    m, k = a.shape
    n = weight_high.shape[0]

    if out is None:
        out = torch.empty((m, n), dtype=resolved_out_dtype, device=a.device)

    required_workspace_size = dual_bf16_weight_gemm_workspace_size(
        int(m), int(n), int(k), a.device
    )
    if workspace_buffer is None:
        stream_handle = int(torch.cuda.current_stream(a.device).cuda_stream)
        workspace_buffer = _get_cache_buf(
            f"mm_bf16_dual_weight_workspace_{stream_handle}",
            max(required_workspace_size, 1),
            a.device,
        )
    elif workspace_buffer.numel() < required_workspace_size:
        raise ValueError(
            "workspace_buffer is too small: "
            f"need {required_workspace_size} bytes, got {workspace_buffer.numel()}"
        )

    module = _get_dual_bf16_weight_gemm_module()
    module.run(a, weight_high, weight_low, out, workspace_buffer)
    return out
