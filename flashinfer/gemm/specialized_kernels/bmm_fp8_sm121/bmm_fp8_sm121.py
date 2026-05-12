from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from typing import Optional

import torch

from ....env import is_specialized_kernel_disabled
from ....jit.core import gen_jit_spec, sm121a_nvcc_flags
from .._utils import is_cuda_13_2_or_newer


def _dtype_from_workload(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported bmm_fp8 specialized workload out dtype: {name}")


def _load_workload_lut() -> dict[tuple, str]:
    lut = {}
    workload_path = Path(__file__).with_name("workloads.json")
    for item in json.loads(workload_path.read_text()):
        key = (
            int(item["b"]),
            int(item["m"]),
            int(item["k"]),
            int(item["n"]),
            _dtype_from_workload(item["out_dtype"]),
            str(item["backend"]),
        )
        lut[key] = str(item["impl"])
    return lut


_WORKLOAD_LUT = _load_workload_lut()
_EMPTY_WORKSPACE_CACHE: dict[torch.device, torch.Tensor] = {}
_WORKSPACE_CACHE: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}


@functools.cache
def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _impl_available(impl: str) -> bool:
    if impl == "cuda":
        return True
    if impl == "cute_dsl":
        return _module_available("cutlass") and _module_available(
            "cuda.bindings.driver"
        )
    if impl == "cutile":
        return _module_available("cuda.tile")
    return False


def _device_key(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return torch.device("cuda", device_index)


def _compute_splits(m: int, n: int, k: int) -> int:
    if m <= 16 and n <= 64 and k <= 2048 and k >= 512 and k % 16 == 0:
        return 1

    small_m = m <= 16
    if small_m:
        bm, bn = 16, 64
    elif m <= 32:
        bm, bn = 32, 128
    elif m <= 64:
        bm, bn = 64, 64 if n <= 1024 else 128
    elif m <= 128:
        bm, bn = 64 if n <= 4096 else 128, 64 if n <= 1024 else 128
    elif m >= 1024:
        bm, bn = 192, 128
    else:
        bm, bn = 64 if n <= 1024 else 128, 64 if n <= 1024 else 128
    blocks = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)

    if blocks < 8:
        split_ok = k >= 1024
    elif blocks < 16:
        split_ok = k >= 2048
    elif blocks < 32:
        split_ok = k >= 2048 if small_m else k >= 4096
    elif blocks < 48:
        split_ok = k >= 4096
    elif blocks < 96:
        split_ok = small_m and k >= 4096
    elif blocks < 128:
        split_ok = small_m and k >= 8192
    else:
        split_ok = False
    if not split_ok:
        return 1

    if small_m and blocks <= 16:
        min_k_per_split = 256
    elif small_m and blocks <= 64:
        min_k_per_split = 512
    else:
        min_k_per_split = 1024
    max_useful_splits = k // min_k_per_split
    if max_useful_splits < 2:
        return 1

    target = 256 if small_m else 96
    splits = min((target + blocks - 1) // blocks, max_useful_splits)
    if small_m and k >= 8192 and blocks >= 32 and splits > 3:
        splits = 3
    if small_m and k <= 4096 and splits > 4:
        splits = 4
    splits = min(splits, 8)
    return 1 if splits < 2 else splits


def _cuda_workspace(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    device = _device_key(A.device)
    m = int(A.shape[1])
    k = int(A.shape[2])
    n = int(out.shape[2])
    splits = _compute_splits(m, n, k)
    required_numel = splits * m * n if splits > 1 else 0
    if required_numel == 0:
        cached = _EMPTY_WORKSPACE_CACHE.get(device)
        if cached is None:
            cached = torch.empty((0,), dtype=torch.float32, device=device)
            _EMPTY_WORKSPACE_CACHE[device] = cached
        return cached

    key = (device, m, n, splits)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is None or cached.numel() < required_numel:
        cached = torch.empty((required_numel,), dtype=torch.float32, device=device)
        _WORKSPACE_CACHE[key] = cached
    return cached


def _select_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
    backend: str,
) -> Optional[str]:
    if is_specialized_kernel_disabled():
        return None
    if not is_cuda_13_2_or_newer():
        return None
    if backend != "cublas":
        return None
    if dtype != torch.bfloat16:
        return None
    if not (
        A.is_cuda and B.is_cuda and A_scale.is_cuda and B_scale.is_cuda and out.is_cuda
    ):
        return None
    if torch.cuda.get_device_capability(A.device) != (12, 1):
        return None
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        return None
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        return None
    if A.ndim != 3 or B.ndim != 3 or out.ndim != 3:
        return None
    if A_scale.numel() != 1 or B_scale.numel() != 1:
        return None
    if not A.is_contiguous() or not out.is_contiguous():
        return None
    if B.stride(1) != 1 or B.stride(2) != B.shape[1]:
        return None

    batch = int(A.shape[0])
    m = int(A.shape[1])
    k = int(A.shape[2])
    n = int(B.shape[2])
    if tuple(B.shape) != (batch, k, n):
        return None
    if tuple(out.shape) != (batch, m, n):
        return None

    impl = _WORKLOAD_LUT.get((batch, m, k, n, dtype, backend))
    if impl is None or not _impl_available(impl):
        return None
    return impl


def is_bmm_fp8_sm121_specialized_problem(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
    backend: str = "cublas",
) -> bool:
    return _select_impl(A, B, A_scale, B_scale, dtype, out, backend) is not None


def gen_bmm_fp8_sm121_specialized_cuda_module():
    source_dir = Path(__file__).resolve().parent
    return gen_jit_spec(
        "bmm_fp8_sm121_specialized_cuda",
        [
            source_dir / "cuda" / "kernel.cu",
            source_dir / "cuda" / "binding.cu",
        ],
        extra_cuda_cflags=sm121a_nvcc_flags,
    )


@functools.cache
def _get_cuda_module():
    return gen_bmm_fp8_sm121_specialized_cuda_module().build_and_load()


@functools.cache
def _get_cute_dsl_kernel():
    from .cute_dsl import kernel as cute_dsl_kernel

    return cute_dsl_kernel


@functools.cache
def _get_cutile_kernel():
    from .cutile import kernel as cutile_kernel

    return cutile_kernel


def run_bmm_fp8_sm121_specialized(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
    backend: str = "cublas",
) -> torch.Tensor:
    impl = _select_impl(A, B, A_scale, B_scale, out.dtype, out, backend)
    if impl is None:
        raise ValueError("Unsupported bmm_fp8 SM121 specialized problem")
    if impl == "cuda":
        _get_cuda_module().run(A, B, A_scale, B_scale, out, _cuda_workspace(A, out))
    elif impl == "cute_dsl":
        _get_cute_dsl_kernel().run(A, B, A_scale, B_scale, out)
    elif impl == "cutile":
        _get_cutile_kernel().run(A, B, A_scale, B_scale, out)
    else:
        raise ValueError(
            f"Unsupported bmm_fp8 SM121 specialized implementation: {impl}"
        )
    return out
