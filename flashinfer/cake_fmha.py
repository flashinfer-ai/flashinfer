"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  Issue
#4323 DCP metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from .jit.cake_fmha import (
    CakeFmhaTarget,
    get_cake_fmha_manifest,
    load_cake_fmha_compat_module,
    load_cake_fmha_decode_native_bf16_module,
)
from .utils import get_compute_capability


@dataclass(frozen=True)
class CakeFmhaDecodeRoute:
    """One exact manifest-backed optimized decode specialization."""

    target: CakeFmhaTarget
    batch_size: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    has_sink: bool
    has_window: bool
    use_scale_ptr: bool
    retain_kv_l2: bool


def _cake_fmha_target(device: torch.device) -> CakeFmhaTarget:
    """Resolve the exact Blackwell cubin target without cross-arch fallback."""

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
    return target


def get_cake_fmha_module(device: torch.device):
    """Load the authenticated complete-domain compatibility module."""

    target = _cake_fmha_target(device)
    return load_cake_fmha_compat_module(target)


def select_cake_fmha_decode_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    q_len: int | None,
    max_seq_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    o_scale: float | None,
    sinks: torch.Tensor | None,
    kv_layout: str,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor | None,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    enable_block_sparse_attention: bool,
) -> CakeFmhaDecodeRoute | None:
    """Select BF16 decode only when the exported route preserves every input."""

    if q_len is None or q_len <= 0 or batch_size <= 0 or max_seq_len <= 0:
        return None
    if kv_layout != "HND" or not uses_shared_paged_kv_idx:
        return None
    if cum_seq_lens_q is not None or enable_block_sparse_attention:
        return None
    if key_block_scales is not None or value_block_scales is not None:
        return None
    if skip_softmax_threshold_scale_factor not in (None, 0.0):
        return None
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (query, key_cache, value_cache, out)
    ):
        return None
    if query.ndim != 3 or not query.is_contiguous() or query.shape[2] != 128:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape or key_cache.shape[2:] != (16, 128):
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    if query.shape[0] != batch_size * q_len:
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        return None
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size:
        return None
    if seq_lens.dtype not in (torch.int32, torch.uint32):
        return None
    if isinstance(bmm2_scale, torch.Tensor) or float(bmm2_scale) != 1.0:
        return None
    if o_scale is not None and float(o_scale) != 1.0:
        return None
    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
    ):
        return None

    local_blocks = max(1, (max_seq_len + 127) // 128)
    return CakeFmhaDecodeRoute(
        target=_cake_fmha_target(device),
        batch_size=batch_size,
        q_len=q_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        has_sink=sinks is not None,
        has_window=window_left >= 0,
        use_scale_ptr=isinstance(bmm1_scale, torch.Tensor),
        retain_kv_l2=local_blocks <= 9,
    )


def get_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
):
    """Load an optimized decode module, or the authenticated portable fallback."""

    if route is None:
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA decode route target does not match the device")
    return load_cake_fmha_decode_native_bf16_module(
        route.target,
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
        has_sink=route.has_sink,
        has_window=route.has_window,
        use_scale_ptr=route.use_scale_ptr,
        retain_kv_l2=route.retain_kv_l2,
    )


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
    "CakeFmhaDecodeRoute",
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "get_cake_fmha_decode_module",
    "get_cake_fmha_module",
    "select_cake_fmha_decode_route",
]
