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
import logging
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import bf16_gemv_trace
from ..utils import get_compute_capability, supported_compute_capability
from .kernels.bf16_gemv_sm12x import SMALL_M_MAX, get_bf16_gemv_kernel

logger = logging.getLogger(__name__)

#: Largest m the GEMV kernel path covers. Larger inputs use cuBLAS.
BF16_GEMV_SMALL_M_MAX = SMALL_M_MAX

# Thresholds for bf16_gemv_window.
BF16_GEMV_MAX_N = 128
BF16_GEMV_MIN_K = 1024
BF16_GEMV_FULL_M_MAX_N = 64


def bf16_gemv_window(m: int, n: int, k: int) -> bool:
    """Whether ``(m, n, k)`` falls in the measured win region of the GEMV.

    Route a bf16 linear through :func:`bf16_gemv` when this returns True.
    The op stays correct outside the window, but wins nothing there.
    """
    if m < 1 or m > SMALL_M_MAX or n > BF16_GEMV_MAX_N or k < BF16_GEMV_MIN_K:
        return False
    return m <= 4 or n <= BF16_GEMV_FULL_M_MAX_N


# (device_index, m, n, k) specializations already compiled and launched
# once. Only these are safe to launch during CUDA-graph capture.
_COMPILED_KEYS: set = set()


@functools.cache
def _check_sm12x(device_index: int) -> None:
    major, minor = get_compute_capability(torch.device("cuda", device_index))
    if major != 12:
        raise ValueError(
            f"bf16_gemv requires an SM120/SM121 GPU, got sm_{major}{minor}"
        )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _cublas_fallback(
    x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor]
) -> torch.Tensor:
    if out is None:
        return torch.nn.functional.linear(x, weight)
    return torch.matmul(x, weight.t(), out=out)


def _use_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    m, k = x.shape
    return (
        1 <= m <= SMALL_M_MAX
        and weight.shape[0] >= 1
        and k >= 8
        and k % 8 == 0
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@supported_compute_capability([120, 121])
@flashinfer_api(trace=bf16_gemv_trace)
def bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""``y = x @ weight.T`` for bf16 ``x (m, K)`` / ``weight (N, K)``.

    A latency-optimized GEMV for decode-size ``m`` on narrow unquantized
    projections, where cuBLAS tile kernels are mostly launch and tile
    overhead. Runs one CTA per output column with f32 accumulation.

    Shapes the kernel does not cover fall back to cuBLAS inside the op, so
    callers can route any bf16 linear through it unconditionally. The
    fallback is also taken for a not-yet-compiled shape while a CUDA graph
    is capturing. Call :func:`precompile_bf16_gemv` at weight-load time so
    capture never hits an uncompiled shape.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(m, K)``, bfloat16.
    weight : torch.Tensor
        Weight of shape ``(N, K)``, bfloat16, row-major contiguous (the
        layout ``F.linear`` takes).
    out : Optional[torch.Tensor]
        Preallocated output of shape ``(m, N)``, bfloat16, contiguous.
        Allocated when ``None``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(m, N)``, bfloat16.

    Notes
    -----
    Requires SM120/SM121 and nvidia-cutlass-dsl. Kernels are compiled per
    ``(m, N, K)`` and cached on disk.
    """
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(
            f"bf16_gemv expects 2D x and weight, got {x.dim()}D and {weight.dim()}D"
        )
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"bf16_gemv K mismatch: x {tuple(x.shape)} vs weight {tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(
            f"bf16_gemv expects bfloat16 inputs, got {x.dtype} and {weight.dtype}"
        )
    m, k = x.shape
    n = weight.shape[0]
    if out is not None:
        if out.shape != (m, n) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"bf16_gemv expects a bfloat16 out of shape {(m, n)}, got "
                f"{out.dtype} {tuple(out.shape)}"
            )
        if not out.is_contiguous():
            raise ValueError("bf16_gemv expects a contiguous out")
    _check_sm12x(_device_index(x.device))

    if not _use_kernel(x, weight) or (out is not None and out.data_ptr() % 16 != 0):
        return _cublas_fallback(x, weight, out)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return _cublas_fallback(x, weight, out)

    # Never JIT or first-launch mid-capture: cute.compile and the lazy module
    # load both break an in-flight CUDA-graph capture.
    key = (_device_index(x.device), m, n, k)
    if key not in _COMPILED_KEYS and torch.cuda.is_current_stream_capturing():
        return _cublas_fallback(x, weight, out)
    kernel = get_bf16_gemv_kernel(m, n, k, key[0])
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    kernel(x, weight, out.view(-1))
    _COMPILED_KEYS.add(key)
    return out


@flashinfer_api
def precompile_bf16_gemv(weight: torch.Tensor) -> None:
    """Compile and warm-run every decode-m variant for ``weight``'s shape.

    Call at weight-load time. Neither compilation nor the lazy first-launch
    module load can happen while a CUDA graph is capturing, so both are
    forced here on a normal stream. Compilation is disk-cached.
    """
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (
        n < 1
        or k < 8
        or k % 8 != 0
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
    ):
        logger.info(
            "bf16_gemv precompile: weight n=%d k=%d unsupported, skipping", n, k
        )
        return
    device_index = _device_index(weight.device)
    _check_sm12x(device_index)
    for m in range(1, SMALL_M_MAX + 1):
        kernel = get_bf16_gemv_kernel(m, n, k, device_index)
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
        y = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
        kernel(x, weight, y.view(-1))
        _COMPILED_KEYS.add((device_index, m, n, k))
    torch.cuda.synchronize()
