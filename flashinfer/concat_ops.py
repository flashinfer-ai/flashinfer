"""
Copyright (c) 2025 by FlashInfer team.

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
from typing import Literal

import torch

from .api_logging import flashinfer_api
from .trace.templates.attention import concat_mla_k_trace

ConcatMLABackend = Literal["default", "cake"]
_NUM_HEADS = 128
_NOPE_DIM = 128
_ROPE_DIM = 64
_OUTPUT_DIM = _NOPE_DIM + _ROPE_DIM
_VECTOR_BYTES = 16
_MAX_SAFE_TOKENS = (2**31 - 1) // 8
_OUTPUT_STRIDES = {(24576, 192, 1), (32768, 256, 1)}
_INPUT_STRIDE_PROFILES = {
    ((16384, 128, 1), (64, 64, 1)),
    ((32768, 256, 1), (64, 64, 1)),
    ((32768, 256, 1), (192, 192, 1)),
}


@functools.cache
def get_concat_mla_module():
    from .jit.dsv3_optimizations import gen_concat_mla_module

    return gen_concat_mla_module().build_and_load()


def _require_cake_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.ndim != 3 or value.stride(2) != 1:
        raise ValueError(f"{name} must be rank 3 with last-dimension stride 1")
    return value


def _require_cake_vector_alignment(
    value: torch.Tensor, *, name: str, element_bytes: int
) -> None:
    if value.data_ptr() % _VECTOR_BYTES:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")
    if any(value.stride(axis) * element_bytes % _VECTOR_BYTES for axis in (0, 1)):
        raise ValueError(f"{name} row and head strides must be 16-byte aligned")


def _validate_cake_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> tuple[int, int]:
    k = _require_cake_tensor(k, name="k")
    k_nope = _require_cake_tensor(k_nope, name="k_nope")
    k_rope = _require_cake_tensor(k_rope, name="k_rope")
    supported_dtypes = {
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }
    if k.dtype not in supported_dtypes:
        raise ValueError(f"unsupported Cake concat_mla_k dtype {k.dtype}")
    if k_nope.dtype != k.dtype or k_rope.dtype != k.dtype:
        raise ValueError("k, k_nope, and k_rope must have the same dtype")
    if k_nope.device != k.device or k_rope.device != k.device:
        raise ValueError("k, k_nope, and k_rope must be on the same device")
    capability = torch.cuda.get_device_capability(k.device)
    if capability != (10, 3):
        raise RuntimeError(
            "the Cake concat_mla_k backend requires exact compute capability "
            f"10.3, got {capability[0]}.{capability[1]}"
        )

    tokens = int(k.shape[0])
    if tuple(k.shape) != (tokens, _NUM_HEADS, _OUTPUT_DIM):
        raise ValueError("k must have shape [tokens, 128, 192]")
    if tuple(k_nope.shape) != (tokens, _NUM_HEADS, _NOPE_DIM):
        raise ValueError("k_nope must have shape [tokens, 128, 128]")
    if tuple(k_rope.shape) != (tokens, 1, _ROPE_DIM):
        raise ValueError("k_rope must have shape [tokens, 1, 64]")
    if tokens < 0 or tokens > _MAX_SAFE_TOKENS:
        raise ValueError(
            f"Cake concat_mla_k token count must be in [0, {_MAX_SAFE_TOKENS}]"
        )

    k_stride = tuple(int(stride) for stride in k.stride())
    k_nope_stride = tuple(int(stride) for stride in k_nope.stride())
    k_rope_stride = tuple(int(stride) for stride in k_rope.stride())
    if k_stride not in _OUTPUT_STRIDES:
        raise ValueError(
            "k stride must be exactly contiguous [24576,192,1] or "
            "padded [32768,256,1] for backend='cake'"
        )
    if (k_nope_stride, k_rope_stride) not in _INPUT_STRIDE_PROFILES:
        raise ValueError(
            "k_nope/k_rope strides must match a proven contiguous, "
            "nope_strided, or both_strided profile for backend='cake'"
        )

    element_bytes = int(k.element_size())
    for name, value in (("k", k), ("k_nope", k_nope), ("k_rope", k_rope)):
        _require_cake_vector_alignment(
            value,
            name=name,
            element_bytes=element_bytes,
        )
    if tokens:
        storage_pointers = {
            int(k.untyped_storage().data_ptr()),
            int(k_nope.untyped_storage().data_ptr()),
            int(k_rope.untyped_storage().data_ptr()),
        }
        if len(storage_pointers) != 3:
            raise ValueError("k, k_nope, and k_rope storage must not overlap")
    return tokens, element_bytes


def _concat_mla_k_cake(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    backend: Literal["cake"],
) -> None:
    tokens, element_bytes = _validate_cake_concat_mla_k(k, k_nope, k_rope)
    if not tokens:
        return None
    from .jit.cake_concat_mla_k import get_cake_concat_mla_k_module

    with torch.cuda.device(k.device):
        get_cake_concat_mla_k_module().run(
            k.view(torch.uint8),
            k_nope.view(torch.uint8),
            k_rope.view(torch.uint8),
            element_bytes,
            int(k.stride(0)) * element_bytes,
            int(k.stride(1)) * element_bytes,
            int(k_nope.stride(0)) * element_bytes,
            int(k_nope.stride(1)) * element_bytes,
            int(k_rope.stride(0)) * element_bytes,
            tokens,
            1,
            1,
        )
    return None


@flashinfer_api(trace=concat_mla_k_trace)
def concat_mla_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    backend: ConcatMLABackend = "default",
) -> None:
    r"""Concatenate k_nope and k_rope tensors for MLA attention.



    This function efficiently concatenates:
      - k_nope: per-head nope values
      - k_rope: shared rope values (broadcast to all heads)

    Supported dtypes: ``torch.bfloat16``, ``torch.float16``,
      ``torch.float8_e4m3fn``, ``torch.float8_e5m2``.

    Key optimizations:
      - Warp-based processing with software pipelining
      - Vectorized memory access (compile-time dispatch per dtype)
      - L2 prefetching for next row while processing current
      - Register reuse for rope values across all heads in a chunk

    Parameters
    ----------
    k : torch.Tensor
        Output tensor, shape: ``[num_tokens, num_heads, nope_dim + rope_dim]``.
        Modified in-place.
    k_nope : torch.Tensor
        The nope part of k, shape: ``[num_tokens, num_heads, nope_dim]``.
    k_rope : torch.Tensor
        The rope part of k (shared), shape: ``[num_tokens, 1, rope_dim]``.
        This is broadcast to all heads.
    backend : {"default", "cake"}, optional
        Backend implementation. The source-only ``"cake"`` backend is
        available on exact SM103a for the documented fixed shape and layouts.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_tokens = 2048
    >>> num_heads = 128
    >>> nope_dim = 128
    >>> rope_dim = 64
    >>> # BF16 example
    >>> k = torch.empty(num_tokens, num_heads, nope_dim + rope_dim, dtype=torch.bfloat16, device="cuda")
    >>> k_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=torch.bfloat16, device="cuda")
    >>> k_rope = torch.randn(num_tokens, 1, rope_dim, dtype=torch.bfloat16, device="cuda")
    >>> flashinfer.concat_ops.concat_mla_k(k, k_nope, k_rope)

    Note
    ----
    This kernel is specifically optimized for:
    - ``num_heads = 128``
    - ``nope_dim = 128``
    - ``rope_dim = 64``
    """
    if backend == "cake":
        return _concat_mla_k_cake(k, k_nope, k_rope, backend="cake")
    if backend != "default":
        raise ValueError(f"unsupported concat_mla_k backend: {backend!r}")
    get_concat_mla_module().concat_mla_k(k, k_nope, k_rope)
    return None
