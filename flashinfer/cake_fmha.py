"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  Issue
#4323 DCP metadata is registered separately as an additive profile and does
not change these base entrypoints.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .jit.cake_fmha import get_cake_fmha_manifest, load_cake_fmha_compat_module
from .utils import get_compute_capability


def get_cake_fmha_module(device: torch.device):
    """Load the authenticated Cake FMHA module for a B200 or B300 device."""

    from .jit.cpp_ext import is_cuda_version_at_least

    capability = get_compute_capability(device)
    if capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError("Cake FMHA on B200 requires CUDA 12.8 or newer")
        target = "sm100a"
    elif capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError("Cake FMHA on B300 requires CUDA 12.9 or newer")
        target = "sm103a"
    else:
        raise RuntimeError(
            "Cake FMHA requires compute capability 10.0 (B200/GB200) or "
            f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
        )
    return load_cake_fmha_compat_module(target)


def cake_fmha_manifest() -> dict[str, Any]:
    """Return a copy of the authenticated product/capability manifest."""

    return copy.deepcopy(get_cake_fmha_manifest())


def cake_batch_decode_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-decode ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_decode_with_kv_cache`.  The Cake backend is
    explicit and never replaces FlashInfer's default backend selection.
    """

    from .decode import trtllm_batch_decode_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_decode_with_kv_cache requires backend='cake'")
    return trtllm_batch_decode_with_kv_cache(*args, backend="cake", **kwargs)


def cake_batch_context_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-context ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_context_with_kv_cache`.  The Cake backend is
    explicit and never replaces FlashInfer's conventional default.
    """

    from .prefill import trtllm_batch_context_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_context_with_kv_cache requires backend='cake'")
    return trtllm_batch_context_with_kv_cache(*args, backend="cake", **kwargs)


__all__ = [
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "get_cake_fmha_module",
]
