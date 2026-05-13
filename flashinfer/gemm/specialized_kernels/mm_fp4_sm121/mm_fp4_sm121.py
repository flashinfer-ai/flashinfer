"""
Runtime router for the mm_fp4 SM121 specialized kernel.

The LUT intentionally routes only exact shapes and static arguments listed in
``workloads.json``.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from typing import Optional

import torch

from ....env import is_specialized_kernel_disabled
from .._utils import is_cuda_13_or_newer

_BLOCK_SIZE = 16


def _dtype_from_workload(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported mm_fp4 specialized workload out dtype: {name}")


def _load_workload_lut() -> dict[tuple, str]:
    lut = {}
    workload_path = Path(__file__).with_name("workloads.json")
    workloads = json.loads(workload_path.read_text())
    for item in workloads:
        key = (
            int(item["m"]),
            int(item["k"]),
            int(item["n"]),
            int(item["block_size"]),
            _dtype_from_workload(item["out_dtype"]),
            bool(item["use_8x4_sf_layout"]),
            str(item["backend"]),
            bool(item["use_nvfp4"]),
        )
        lut[key] = str(item["impl"])
    return lut


_WORKLOAD_LUT = _load_workload_lut()

_ALPHA_ONE_CACHE: dict[torch.device, torch.Tensor] = {}


def _device_key(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return torch.device("cuda", device_index)


def _is_column_major_view(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and tensor.stride(0) == 1
        and tensor.stride(1) == tensor.shape[0]
    )


def _normalize_backend(backend: str) -> str:
    return "b12x" if backend == "auto" else backend


@functools.cache
def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _impl_available(impl: str) -> bool:
    if impl == "cute_dsl":
        return _module_available("cutlass") and _module_available(
            "cuda.bindings.driver"
        )
    return False


def _select_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    use_8x4_sf_layout: bool,
    backend: str,
    use_nvfp4: bool,
) -> Optional[str]:
    if is_specialized_kernel_disabled():
        return None
    if not is_cuda_13_or_newer():
        return None
    if _normalize_backend(backend) != "b12x":
        return None
    if out_dtype != torch.bfloat16 or block_size != _BLOCK_SIZE:
        return None
    if use_8x4_sf_layout or not use_nvfp4:
        return None
    if not (a.is_cuda and b.is_cuda and a_descale.is_cuda and b_descale.is_cuda):
        return None
    if torch.cuda.get_device_capability(a.device) != (12, 1):
        return None
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        return None
    if a_descale.dtype != torch.uint8 or b_descale.dtype != torch.uint8:
        return None
    if a.ndim != 2 or b.ndim != 2 or a_descale.ndim != 2 or b_descale.ndim != 2:
        return None

    m = int(a.shape[0])
    k = int(a.shape[1] * 2)
    n = int(b.shape[1])
    key = (
        m,
        k,
        n,
        block_size,
        out_dtype,
        use_8x4_sf_layout,
        _normalize_backend(backend),
        use_nvfp4,
    )
    impl = _WORKLOAD_LUT.get(key)
    if impl is None or not _impl_available(impl):
        return None
    if tuple(b.shape) != (k // 2, n):
        return None

    sf_m = ((m + 127) // 128) * 128
    sf_n = ((n + 127) // 128) * 128
    if tuple(a_descale.shape) != (sf_m, k // block_size):
        return None
    if tuple(b_descale.shape) != (k // block_size, sf_n):
        return None
    if not a.is_contiguous() or not a_descale.is_contiguous():
        return None
    if not _is_column_major_view(b) or not _is_column_major_view(b_descale):
        return None

    if out is not None:
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n):
            return None
        if not out.is_cuda or not out.is_contiguous():
            return None

    return impl


def is_mm_fp4_sm121_specialized_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: str = "auto",
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> bool:
    del alpha, enable_pdl

    return (
        _select_impl(
            a,
            b,
            a_descale,
            b_descale,
            out_dtype,
            out,
            block_size,
            use_8x4_sf_layout,
            backend,
            use_nvfp4,
        )
        is not None
    )


@functools.cache
def _get_cute_dsl_kernel():
    from .cute_dsl import kernel as cute_dsl_kernel

    return cute_dsl_kernel


def _prepare_alpha(alpha: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    device = _device_key(device)
    if alpha is None:
        cached = _ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha.dim() == 0:
        return alpha.unsqueeze(0)
    return alpha.reshape(1)


def run_mm_fp4_sm121_specialized(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    impl = _select_impl(
        a,
        b,
        a_descale,
        b_descale,
        out.dtype,
        out,
        _BLOCK_SIZE,
        False,
        "b12x",
        True,
    )
    if impl is None:
        raise ValueError("Unsupported mm_fp4 SM121 specialized problem")

    alpha_tensor = _prepare_alpha(alpha, a.device)
    if impl == "cute_dsl":
        _get_cute_dsl_kernel().run(a, b, a_descale, b_descale, alpha_tensor, out)
    else:
        raise ValueError(f"Unsupported mm_fp4 SM121 specialized implementation: {impl}")
    return out
