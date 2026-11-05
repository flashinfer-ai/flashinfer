# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-neutral tensor helpers.

The frontend-only CuTeDSL APIs only need tensor *metadata* (shape/stride/dtype/device)
plus a CUDA stream at launch time -- the data handoff to the compiled kernels is DLPack.
This module reads that metadata from torch tensors, JAX arrays, or numpy arrays without
eagerly importing any of those frameworks: a tensor of framework X can only exist if X is
already in sys.modules, so probing sys.modules never triggers an import.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cuda.bindings.driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type


@dataclass(frozen=True)
class Device:
    """Minimal stand-in for torch.device with the same .type/.index/str() surface."""

    type: str
    index: Optional[int] = None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


_torch_tensor_cls: Any = None


def is_torch_tensor(tensor: Any) -> bool:
    # Called ~40x per grouped-GEMM launch (once per operand per metadata read), so the
    # sys.modules probe and the .Tensor attribute lookup are worth caching. torch cannot
    # be un-imported, so the class is stable once resolved; until then this re-probes.
    global _torch_tensor_cls
    cls = _torch_tensor_cls
    if cls is None:
        torch = sys.modules.get("torch")
        if torch is None:
            return False
        cls = _torch_tensor_cls = torch.Tensor
    return isinstance(tensor, cls)


def is_jax_array(tensor: Any) -> bool:
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(tensor, getattr(jax, "Array", ())):
        return True
    return type(tensor).__module__.startswith(("jax", "jaxlib"))


def is_numpy_array(tensor: Any) -> bool:
    numpy = sys.modules.get("numpy")
    return numpy is not None and isinstance(tensor, numpy.ndarray)


def detect_framework(tensor: Any) -> str:
    """Return "torch", "jax", "numpy", or "unknown" for the given tensor."""
    if is_torch_tensor(tensor):
        return "torch"
    if is_jax_array(tensor):
        return "jax"
    if is_numpy_array(tensor):
        return "numpy"
    return "unknown"


def get_shape(tensor: Any) -> Tuple[int, ...]:
    return tuple(tensor.shape)


def _c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def get_strides(tensor: Any) -> Tuple[int, ...]:
    """Strides in elements. JAX arrays don't expose strides; their default layout is C-contiguous."""
    if is_torch_tensor(tensor):
        return tuple(tensor.stride())
    if is_jax_array(tensor):
        return _c_contiguous_strides(get_shape(tensor))
    if is_numpy_array(tensor):
        return tuple(s // tensor.itemsize for s in tensor.strides)
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        return tuple(stride())
    strides = getattr(tensor, "strides", None)
    if strides is not None:
        return tuple(strides)
    return _c_contiguous_strides(get_shape(tensor))


def get_device(tensor: Any) -> Any:
    """Device of the tensor: torch tensors keep their native torch.device; others map to Device."""
    if is_torch_tensor(tensor):
        return tensor.device
    if is_jax_array(tensor):
        jax_device = getattr(tensor, "device", None)
        if jax_device is None or callable(jax_device):
            jax_device = next(iter(tensor.devices()))
        platform = jax_device.platform
        return Device("cuda" if platform == "gpu" else platform, jax_device.id)
    if is_numpy_array(tensor):
        return Device("cpu")
    device = getattr(tensor, "device", None)
    if device is not None:
        return device
    return Device("unknown")


def default_stream(framework: str) -> cuda.CUstream:
    """Stream to launch on when the caller did not pass one.

    torch has a per-thread "current stream" notion; other frameworks don't expose one,
    so they get the CUDA legacy default stream (callers doing stream overlap should
    pass an explicit cuda.CUstream).
    """
    if framework == "torch":
        import torch

        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cuda.CUstream(0)


def cuda_is_available() -> bool:
    torch = sys.modules.get("torch")
    if torch is not None:
        return torch.cuda.is_available()
    from cuda.bindings import runtime as cudart

    err, count = cudart.cudaGetDeviceCount()
    return err == cudart.cudaError_t.cudaSuccess and count > 0


def get_compute_capability(device_index: Optional[int] = None) -> Tuple[int, int]:
    """(major, minor) of the selected CUDA device (current if omitted), without requiring torch."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_capability(torch.cuda.current_device() if device_index is None else device_index)
    from cuda.bindings import runtime as cudart

    def _check(result):
        err, *values = result
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error: {err}")
        return values[0] if len(values) == 1 else values

    device = _check(cudart.cudaGetDevice()) if device_index is None else device_index
    major = _check(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device))
    minor = _check(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device))
    return major, minor


# Memoized: pure function of (shape, stride), called per input tensor on every
# CuTeDSL launch to build the cache key; bounded since keys are shape-derived.
@functools.lru_cache(maxsize=4096)
def canonicalize_unit_dim_strides(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> Tuple[int, ...]:
    """Give extent-1 dims the dense "outermost" stride (numel) so that layouts that differ
    only in unit-dim strides -- which the kernels cannot observe -- compare and compile equal."""
    numel = 1
    for dim in shape:
        numel *= dim
    return tuple(numel if dim == 1 else s for dim, s in zip(shape, stride))


def get_data_ptr(tensor: Any) -> int:
    """Device data pointer of the tensor, in the caller's framework.

    JAX note: the pointer is only valid while the array is alive and not donated;
    callers must hold a reference for the duration of any kernel that uses it.
    """
    if is_torch_tensor(tensor):
        return tensor.data_ptr()
    if is_jax_array(tensor):
        return tensor.unsafe_buffer_pointer()
    data_ptr = getattr(tensor, "data_ptr", None)
    if callable(data_ptr):
        return data_ptr()
    raise ValueError(f"Cannot extract a device pointer from {type(tensor)!r}")


def get_version(tensor: Any) -> int:
    """Mutation counter for validation caching: torch's ._version, 0 for immutable arrays (JAX)."""
    return int(getattr(tensor, "_version", 0))


def to_host_list(tensor: Any) -> list:
    """Copy a small device tensor to host and return its values as a flat Python list."""
    if is_torch_tensor(tensor):
        return tensor.detach().cpu().flatten().tolist()
    import numpy as np

    return np.asarray(tensor).flatten().tolist()


def allocate_byte_workspace(framework: str, nbytes: int, device: Any) -> Any:
    """Allocate an internal uint8 workspace buffer in the caller's framework allocator.

    The buffer is written by kernels through its raw pointer and never surfaced as a
    framework array, so allocating it as a (zero-initialized, for JAX) framework tensor
    is safe; the caller must keep a reference alive for the compiled kernel's lifetime.
    """
    nbytes = max(int(nbytes), 1)
    if framework == "torch":
        import torch

        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        if isinstance(device, Device):
            # Canonical descriptor device -> the corresponding jax device
            device = jax.devices("gpu")[device.index or 0] if device.type == "cuda" else None
        buffer = jnp.zeros((nbytes,), dtype=jnp.uint8, device=device)
        # Materialize before anyone reads its pointer
        return jax.block_until_ready(buffer)
    raise ValueError(f"Cannot allocate a workspace for framework '{framework}'")


def pad_to_ndim(tensor: Any, ndim: int) -> Any:
    """Append size-1 dims up to ndim; works for any framework tensor exposing reshape."""
    shape = get_shape(tensor)
    if len(shape) >= ndim:
        return tensor
    unsqueeze = getattr(tensor, "unsqueeze", None)
    if callable(unsqueeze):
        while len(get_shape(tensor)) < ndim:
            tensor = tensor.unsqueeze(-1)
        return tensor
    return tensor.reshape(shape + (1,) * (ndim - len(shape)))


# cutlass type -> framework-neutral dtype name; built lazily to keep cutlass import lazy.
_cutlass_to_name_dict = None


def _cutlass_to_dtype_name(cutlass_dtype: Any) -> str:
    global _cutlass_to_name_dict
    if _cutlass_to_name_dict is None:
        import cutlass

        names = {
            "Float16": "float16",
            "BFloat16": "bfloat16",
            "Float32": "float32",
            "Float64": "float64",
            "Uint8": "uint8",
            "Int8": "int8",
            "Int32": "int32",
            "Int64": "int64",
            "Boolean": "bool",
            "Float8E4M3FN": "float8_e4m3fn",
            "Float8E5M2": "float8_e5m2",
            "Float8E8M0FNU": "float8_e8m0fnu",
            "Float4E2M1FN": "float4_e2m1fn",
        }
        _cutlass_to_name_dict = {getattr(cutlass, attr): name for attr, name in names.items() if getattr(cutlass, attr, None) is not None}
    name = _cutlass_to_name_dict.get(cutlass_dtype)
    if name is None:
        raise ValueError(f"No dtype name known for cutlass type {cutlass_dtype}.")
    return name


def framework_dtype(dtype: Any, framework: str) -> Any:
    """Convert a dtype (cutlass/torch/numpy/ml_dtypes/str) to the given framework's dtype object.

    Used when the wrapper APIs allocate output tensors in the caller's framework.
    """
    canonical = _convert_to_cutlass_data_type(dtype)
    name = _cutlass_to_dtype_name(canonical)
    if framework == "torch":
        import torch

        if name == "float4_e2m1fn":
            return torch.float4_e2m1fn_x2
        return getattr(torch, name)
    if framework == "jax":
        if name == "float4_e2m1fn":
            raise ValueError("JAX has no packed fp4 dtype; use a uint8 container tensor instead.")
        import numpy as np

        try:
            return np.dtype(name)
        except TypeError:
            import ml_dtypes

            return np.dtype(getattr(ml_dtypes, name))
    raise ValueError(f"Cannot materialize dtype {dtype} for framework '{framework}'.")
