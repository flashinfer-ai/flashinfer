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

MagiAttention Flex-Flash-Attention kernel wrapper.

This module provides the single-GPU FFA implementation used by FlashInfer's
``backend="ffa"`` paths. The public package initializer re-exports the API
from here, while ``flashinfer.prefill`` owns the main FlashInfer user-facing
backend integration.

* Mask type enumeration (:class:`FFAMaskType`)
* Ranges helpers (:func:`causal_ranges`, :func:`full_ranges`,
  :func:`varlen_causal_ranges`)
* Function-style prefill API (:func:`flex_prefill`, :func:`causal_prefill`,
  :func:`varlen_causal_prefill`)
* FlashInfer-style plan/run wrapper (:class:`BatchPrefillFFAWrapper`) that
  mirrors :class:`flashinfer.BatchPrefillWithRaggedKVCacheWrapper`

Requires: ``pip install magi_attention``
"""

from __future__ import annotations

import math
import weakref
from typing import Callable, List, Optional, Tuple, Union

import torch

_magi_available: Optional[bool] = None
_ffa_func: Optional[Callable] = None
_single_ranges_cache = {}
_varlen_ranges_cache = {}
_qkv_shape_cache = set()


def _check_magi() -> None:
    """Raise ImportError with install instructions if ``magi_attention`` is missing."""
    global _magi_available
    if _magi_available is None:
        try:
            import magi_attention.functional.flex_flash_attn  # noqa: F401

            _magi_available = True
        except ImportError:
            _magi_available = False
    if not _magi_available:
        raise ImportError(
            "flashinfer.ffa_kernels requires 'magi_attention' to be installed. "
            "Install it with: pip install magi_attention   "
            "or see https://github.com/SandAI-org/MagiAttention"
        )


def _get_ffa_func():
    global _ffa_func
    _check_magi()
    if _ffa_func is not None:
        return _ffa_func
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

    _ffa_func = flex_flash_attn_func
    return _ffa_func


def _check_qkv_shape_for_ffa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_qo_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be 3-D tensors (tokens, heads, head_dim)")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not q.is_cuda:
        raise ValueError("FFA requires q, k, and v to be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("FFA requires q, k, and v to be contiguous tensors")
    if k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same token and head dimensions")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("FFA requires q, k, and v to have the same head_dim")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if num_qo_heads is not None and q.shape[1] != num_qo_heads:
        raise ValueError(
            f"q.shape[1] ({q.shape[1]}) does not match planned num_qo_heads "
            f"({num_qo_heads})"
        )
    if num_kv_heads is not None and k.shape[1] != num_kv_heads:
        raise ValueError(
            f"k.shape[1] ({k.shape[1]}) does not match planned num_kv_heads "
            f"({num_kv_heads})"
        )
    if head_dim is not None and q.shape[-1] != head_dim:
        raise ValueError(
            f"q.shape[-1] ({q.shape[-1]}) does not match planned head_dim "
            f"({head_dim})"
        )


def _check_qkv_shape_for_ffa_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_qo_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> None:
    key = (
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.dtype,
        k.dtype,
        v.dtype,
        q.device,
        k.device,
        v.device,
        q.is_contiguous(),
        k.is_contiguous(),
        v.is_contiguous(),
        num_qo_heads,
        num_kv_heads,
        head_dim,
    )
    if key in _qkv_shape_cache:
        return
    _check_qkv_shape_for_ffa(
        q,
        k,
        v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if len(_qkv_shape_cache) > 2048:
        _qkv_shape_cache.clear()
    _qkv_shape_cache.add(key)


def _validate_range_tensor(
    ranges: torch.Tensor, name: str
) -> Tuple[torch.Tensor, int]:
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"{name} must have shape (num_ranges, 2)")
    ranges_cpu = ranges.to("cpu", dtype=torch.int32)
    if torch.any(ranges_cpu < 0):
        raise ValueError(f"{name} must be non-negative")
    if torch.any(ranges_cpu[:, 1] < ranges_cpu[:, 0]):
        raise ValueError(f"{name} end must be greater than or equal to start")
    max_end = int(ranges_cpu[:, 1].max().item()) if ranges_cpu.numel() else 0
    return ranges_cpu, max_end


def _check_range_bound(name: str, max_end: int, limit: int) -> None:
    if max_end > limit:
        raise ValueError(f"{name} end ({max_end}) exceeds tensor length ({limit})")


def _check_tensor_shape_dtype_device(
    tensor: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name}.shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name}.device must be {device}, got {tensor.device}")


# ---------------------------------------------------------------------------
# Mask type constants (mirrors Magi AttnMaskType int encoding)
# ---------------------------------------------------------------------------


class FFAMaskType:
    """Integer constants for the ``attn_type_map`` tensor.

    - 0 = FULL
    - 1 = CAUSAL  (bottom-right aligned)
    - 2 = INVERSE_CAUSAL (top-left aligned)
    - 3 = BIDIRECTIONAL_CAUSAL (diagonal)
    """

    FULL: int = 0
    CAUSAL: int = 1
    INVERSE_CAUSAL: int = 2
    BIDIRECTIONAL_CAUSAL: int = 3


# ---------------------------------------------------------------------------
# Range generators
# ---------------------------------------------------------------------------


def causal_ranges(
    qo_len: int,
    kv_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``(q_ranges, k_ranges, attn_type_map)`` for a single causal segment."""
    q_ranges = torch.tensor([[0, qo_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, kv_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor(
        [FFAMaskType.CAUSAL], dtype=torch.int32, device=device
    )
    return q_ranges, k_ranges, attn_type_map


def full_ranges(
    qo_len: int,
    kv_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``(q_ranges, k_ranges, attn_type_map)`` for a single full-attention segment."""
    q_ranges = torch.tensor([[0, qo_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, kv_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor(
        [FFAMaskType.FULL], dtype=torch.int32, device=device
    )
    return q_ranges, k_ranges, attn_type_map


def _cached_single_ranges(
    qo_len: int,
    kv_len: int,
    device: torch.device,
    mask_type: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (str(device), qo_len, kv_len, mask_type)
    cached = _single_ranges_cache.get(key)
    if cached is None:
        if len(_single_ranges_cache) >= 2048:
            _single_ranges_cache.clear()
        q_ranges = torch.tensor([[0, qo_len]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, kv_len]], dtype=torch.int32, device=device)
        attn_type_map = torch.tensor([mask_type], dtype=torch.int32, device=device)
        cached = (q_ranges, k_ranges, attn_type_map)
        _single_ranges_cache[key] = cached
    return cached


def _range_cache_token(tensor: torch.Tensor) -> Tuple[object, ...]:
    return (
        id(tensor),
        str(tensor.device),
        tuple(tensor.shape),
        tensor.dtype,
        int(getattr(tensor, "_version", 0)),
    )


def _cached_varlen_causal_ranges(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        str(device),
        _range_cache_token(cu_seqlens_q),
        _range_cache_token(cu_seqlens_k),
    )
    entry = _varlen_ranges_cache.get(key)
    if entry is not None:
        q_ref, k_ref, cached = entry
        if q_ref() is cu_seqlens_q and k_ref() is cu_seqlens_k:
            return cached
        _varlen_ranges_cache.pop(key, None)

    if len(_varlen_ranges_cache) > 1024:
        _varlen_ranges_cache.clear()
    cached = varlen_causal_ranges(cu_seqlens_q, cu_seqlens_k, device)
    _varlen_ranges_cache[key] = (
        weakref.ref(cu_seqlens_q),
        weakref.ref(cu_seqlens_k),
        cached,
    )
    return cached


def varlen_causal_ranges(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``(q_ranges, k_ranges, attn_type_map)`` for multi-document varlen causal.

    Parameters
    ----------
    cu_seqlens_q : torch.Tensor
        Cumulative sequence lengths for Q, shape ``(batch_size + 1,)``.
    cu_seqlens_k : torch.Tensor
        Cumulative sequence lengths for K, shape ``(batch_size + 1,)``.

    Each document ``i`` maps to one causal slice:
        q_range = [cu_seqlens_q[i], cu_seqlens_q[i+1])
        k_range = [cu_seqlens_k[i], cu_seqlens_k[i+1])
    """
    cu_q = cu_seqlens_q.to("cpu", dtype=torch.int32)
    cu_k = cu_seqlens_k.to("cpu", dtype=torch.int32)
    if cu_q.numel() == 0 or cu_k.numel() == 0:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be non-empty")
    batch_size = len(cu_q) - 1
    if len(cu_k) - 1 != batch_size:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have same batch_size")
    if batch_size <= 0:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe at least one segment")
    if int(cu_q[0]) != 0 or int(cu_k[0]) != 0:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must start with 0")
    if torch.any(cu_q[1:] < cu_q[:-1]) or torch.any(cu_k[1:] < cu_k[:-1]):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be non-decreasing")

    q_ranges_list = []
    k_ranges_list = []
    for i in range(batch_size):
        q_ranges_list.append([int(cu_q[i]), int(cu_q[i + 1])])
        k_ranges_list.append([int(cu_k[i]), int(cu_k[i + 1])])

    q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
    attn_type_map = torch.full(
        (batch_size,), FFAMaskType.CAUSAL, dtype=torch.int32, device=device
    )
    return q_ranges, k_ranges, attn_type_map


def _indptr_to_ranges(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    causal: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate ``qo_indptr`` / ``kv_indptr`` + ``causal`` into FFA ranges."""
    cu_q = qo_indptr.to("cpu", dtype=torch.int32)
    cu_k = kv_indptr.to("cpu", dtype=torch.int32)
    if cu_q.numel() == 0 or cu_k.numel() == 0:
        raise ValueError("qo_indptr and kv_indptr must be non-empty")
    batch_size = len(cu_q) - 1
    if len(cu_k) - 1 != batch_size:
        raise ValueError(
            "qo_indptr and kv_indptr must describe the same number of segments"
        )
    if batch_size <= 0:
        raise ValueError("qo_indptr and kv_indptr must describe at least one segment")
    if int(cu_q[0]) != 0 or int(cu_k[0]) != 0:
        raise ValueError("qo_indptr and kv_indptr must start with 0")
    if torch.any(cu_q[1:] < cu_q[:-1]) or torch.any(cu_k[1:] < cu_k[:-1]):
        raise ValueError("qo_indptr and kv_indptr must be non-decreasing")

    q_ranges_list = [[int(cu_q[i]), int(cu_q[i + 1])] for i in range(batch_size)]
    k_ranges_list = [[int(cu_k[i]), int(cu_k[i + 1])] for i in range(batch_size)]

    q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
    fill = FFAMaskType.CAUSAL if causal else FFAMaskType.FULL
    attn_type_map = torch.full(
        (batch_size,), fill, dtype=torch.int32, device=device
    )
    return q_ranges, k_ranges, attn_type_map


# ---------------------------------------------------------------------------
# Function-style prefill API
# ---------------------------------------------------------------------------


def _flex_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: Optional[torch.Tensor] = None,
    *,
    kv_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    deterministic: bool = False,
    pack_gqa: bool = False,
    return_lse: bool = False,
    validate_ranges: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Internal implementation for MagiAttention Flex-Flash-Attention prefill.

    ``validate_ranges=False`` is used only by helpers that construct trusted
    ranges internally and want to avoid GPU-to-CPU synchronization.
    """
    ffa_func = _get_ffa_func()

    if validate_ranges:
        q_ranges_cpu, q_max_end = _validate_range_tensor(q_ranges, "q_ranges")
        k_ranges_cpu, k_max_end = _validate_range_tensor(k_ranges, "k_ranges")
    else:
        if q_ranges.ndim != 2 or q_ranges.shape[1] != 2:
            raise ValueError("q_ranges must have shape (num_ranges, 2)")
        if k_ranges.ndim != 2 or k_ranges.shape[1] != 2:
            raise ValueError("k_ranges must have shape (num_ranges, 2)")
        q_ranges_cpu = q_ranges
        k_ranges_cpu = k_ranges
        q_max_end = k_max_end = -1
    if q_ranges_cpu.shape[0] != k_ranges_cpu.shape[0]:
        raise ValueError("q_ranges and k_ranges must have same num_ranges")
    num_ranges = q_ranges_cpu.shape[0]
    if attn_type_map is not None:
        if attn_type_map.ndim != 1:
            raise ValueError("attn_type_map must be a 1-D tensor")
        if attn_type_map.shape[0] != num_ranges:
            raise ValueError("attn_type_map length must equal num_ranges")
        if validate_ranges:
            attn_type_map_cpu = attn_type_map.to("cpu", dtype=torch.int32)
            if torch.any((attn_type_map_cpu < 0) | (attn_type_map_cpu > 3)):
                raise ValueError("attn_type_map values must be one of 0, 1, 2, 3")
        else:
            attn_type_map_cpu = attn_type_map
    else:
        attn_type_map_cpu = None

    kv_layout = kv_layout.upper()
    if kv_layout == "HND":
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
    elif kv_layout != "NHD":
        raise ValueError(f"kv_layout must be 'NHD' or 'HND', got '{kv_layout}'")

    _check_qkv_shape_for_ffa_cached(q, k, v)
    if validate_ranges:
        _check_range_bound("q_ranges", q_max_end, q.shape[0])
        _check_range_bound("k_ranges", k_max_end, k.shape[0])

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    device = q.device
    q_ranges = q_ranges_cpu.to(device=device, dtype=torch.int32)
    k_ranges = k_ranges_cpu.to(device=device, dtype=torch.int32)
    if attn_type_map_cpu is not None:
        attn_type_map = attn_type_map_cpu.to(device=device, dtype=torch.int32)

    out, meta = ffa_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=sm_scale,
        softcap=logits_soft_cap,
        deterministic=deterministic,
        pack_gqa=pack_gqa,
        disable_fwd_atomic_reduction=False,
    )

    if return_lse:
        return out, meta.lse
    return out


def flex_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: Optional[torch.Tensor] = None,
    *,
    kv_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    deterministic: bool = False,
    pack_gqa: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Prefill attention using MagiAttention Flex-Flash-Attention kernels.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(num_tokens_q, num_qo_heads, head_dim)``.
    k, v : torch.Tensor
        Key/value tensors.
        NHD layout: ``(num_tokens_kv, num_kv_heads, head_dim)``
        HND layout: ``(num_kv_heads, num_tokens_kv, head_dim)``
    q_ranges : torch.Tensor
        Shape ``(num_ranges, 2)`` int32, each row ``[start, end)``.
    k_ranges : torch.Tensor
        Shape ``(num_ranges, 2)`` int32, each row ``[start, end)``.
    attn_type_map : Optional[torch.Tensor]
        Shape ``(num_ranges,)`` int32. Per-slice mask type (see ``FFAMaskType``).
        None means FULL for all slices.
    kv_layout : str
        ``"NHD"`` (default) or ``"HND"``. HND is transposed internally.
    sm_scale : Optional[float]
        Softmax scale. Default ``1/sqrt(head_dim)``.
    logits_soft_cap : float
        Logits soft capping value. Default ``0.0`` (disabled).
    deterministic : bool
        Enable deterministic mode.
    pack_gqa : bool
        Pack GQA heads for small seqlen_q optimization.
    return_lse : bool
        Whether to return log-sum-exp.

    Returns
    -------
    If return_lse=False: ``out (num_tokens_q, num_qo_heads, head_dim)``
    If return_lse=True:  ``(out, lse)`` where lse shape ``(num_tokens_q, num_qo_heads)``
    """
    return _flex_prefill_impl(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        kv_layout=kv_layout,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        deterministic=deterministic,
        pack_gqa=pack_gqa,
        return_lse=return_lse,
        validate_ranges=True,
    )


def causal_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-segment causal prefill via FFA.

    Equivalent to ``flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)``
    but routed through the MagiAttention FFA kernel.
    """
    qo_len = q.shape[0]
    kv_len = k.shape[1] if kv_layout.upper() == "HND" else k.shape[0]
    qr, kr, atm = _cached_single_ranges(qo_len, kv_len, q.device, FFAMaskType.CAUSAL)
    return _flex_prefill_impl(
        q,
        k,
        v,
        qr,
        kr,
        atm,
        kv_layout=kv_layout,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
        validate_ranges=False,
    )


def varlen_causal_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    kv_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Multi-document varlen causal prefill via FFA.

    Each document is a separate causal slice.
    """
    qr, kr, atm = _cached_varlen_causal_ranges(
        cu_seqlens_q, cu_seqlens_k, q.device
    )
    return _flex_prefill_impl(
        q,
        k,
        v,
        qr,
        kr,
        atm,
        kv_layout=kv_layout,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
        validate_ranges=False,
    )


# ---------------------------------------------------------------------------
# FlashInfer-style plan/run wrapper
# ---------------------------------------------------------------------------


class BatchPrefillFFAWrapper:
    r"""Plan/run-style wrapper for MagiAttention Flex-Flash-Attention prefill.

    Mirrors the lifecycle of
    :class:`flashinfer.BatchPrefillWithRaggedKVCacheWrapper` (``__init__`` ->
    ``plan`` -> ``run``) so FFA can be dropped into FlashInfer-style model code
    without bespoke bookkeeping.

    Two equivalent ways to describe the mask in :meth:`plan`:

    1. FlashInfer style - pass ``qo_indptr`` / ``kv_indptr`` plus ``causal``.
       Each segment ``[qo_indptr[i], qo_indptr[i+1])`` is treated as an
       independent attention slice (FULL when ``causal=False``, CAUSAL
       otherwise).
    2. FFA native style - pass ``q_ranges`` / ``k_ranges`` / ``attn_type_map``
       directly, unlocking heterogeneous mask types per slice
       (FULL / CAUSAL / INVERSE_CAUSAL / BIDIRECTIONAL_CAUSAL).

    The two styles are mutually exclusive; supplying both raises ``ValueError``.

    Example
    -------
    >>> import torch, flashinfer
    >>> from flashinfer.ffa_kernels import BatchPrefillFFAWrapper
    >>> workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = BatchPrefillFFAWrapper(workspace, kv_layout="NHD")
    >>> qo_indptr = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda:0")
    >>> kv_indptr = qo_indptr.clone()
    >>> wrapper.plan(
    ...     qo_indptr=qo_indptr,
    ...     kv_indptr=kv_indptr,
    ...     num_qo_heads=32,
    ...     num_kv_heads=8,
    ...     head_dim=128,
    ...     causal=True,
    ... )
    >>> q = torch.randn(256, 32, 128, dtype=torch.bfloat16, device="cuda:0")
    >>> k = torch.randn(256, 8, 128, dtype=torch.bfloat16, device="cuda:0")
    >>> v = torch.randn(256, 8, 128, dtype=torch.bfloat16, device="cuda:0")
    >>> o = wrapper.run(q, k, v)
    >>> o.shape
    torch.Size([256, 32, 128])
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
    ) -> None:
        r"""Construct the wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            A user-reserved workspace buffer. Kept for API parity with
            :class:`flashinfer.BatchPrefillWithRaggedKVCacheWrapper`; the FFA
            kernel manages its own scratch and does not currently consume this
            buffer, but the device it lives on determines where plan tensors
            are placed.
        kv_layout : str
            Either ``"NHD"`` (default) or ``"HND"``. ``"HND"`` inputs are
            transposed to ``"NHD"`` before calling the FFA kernel.
        """
        kv_layout = kv_layout.upper()
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(
                f"kv_layout must be 'NHD' or 'HND', got '{kv_layout}'"
            )
        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._q_ranges: Optional[torch.Tensor] = None
        self._k_ranges: Optional[torch.Tensor] = None
        self._attn_type_map: Optional[torch.Tensor] = None
        self._q_max_end: int = 0
        self._k_max_end: int = 0

        self._num_qo_heads: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._sm_scale: Optional[float] = None
        self._logits_soft_cap: float = 0.0
        self._deterministic: bool = False
        self._pack_gqa: bool = False
        self._disable_fwd_atomic_reduction: bool = False
        self._auto_range_merge: bool = False
        self._ref_block_size: Optional[Tuple[int, int]] = None
        self._max_seqlen_q: Optional[int] = None
        self._swap_ab: bool = False
        self._sparse_load: bool = False
        self._planned: bool = False
        self._ffa_func: Optional[Callable] = None
        self._validated_run_signature: Optional[Tuple[object, ...]] = None

    @property
    def kv_layout(self) -> str:
        return self._kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor
    ) -> None:
        """Reset the float workspace buffer (kept for API parity)."""
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        if self._q_ranges is not None:
            self._q_ranges = self._q_ranges.to(device=self.device, dtype=torch.int32)
        if self._k_ranges is not None:
            self._k_ranges = self._k_ranges.to(device=self.device, dtype=torch.int32)
        if self._attn_type_map is not None:
            self._attn_type_map = self._attn_type_map.to(
                device=self.device, dtype=torch.int32
            )
        self._validated_run_signature = None

    def plan(
        self,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = False,
        q_ranges: Optional[torch.Tensor] = None,
        k_ranges: Optional[torch.Tensor] = None,
        attn_type_map: Optional[Union[torch.Tensor, List[int]]] = None,
        sm_scale: Optional[float] = None,
        logits_soft_cap: float = 0.0,
        deterministic: bool = False,
        pack_gqa: bool = False,
        disable_fwd_atomic_reduction: bool = False,
        auto_range_merge: bool = False,
        ref_block_size: Optional[Tuple[int, int]] = None,
        max_seqlen_q: Optional[int] = None,
        swap_ab: bool = False,
        sparse_load: bool = False,
    ) -> None:
        r"""Plan the attention computation by resolving the mask description.

        Either pass ``qo_indptr``/``kv_indptr`` (+ optional ``causal``) **or**
        pass ``q_ranges``/``k_ranges``/``attn_type_map`` - not both.

        Parameters
        ----------
        qo_indptr, kv_indptr : Optional[torch.Tensor]
            Cumulative segment lengths with shape ``[batch_size + 1]``.
            When supplied, each segment ``i`` becomes one FFA slice. Ignored
            if ``q_ranges`` is provided.
        num_qo_heads : int
        num_kv_heads : int
        head_dim : int
        causal : bool
            Mask type used to fill ``attn_type_map`` when translating from
            ``qo_indptr``/``kv_indptr``. Defaults to ``False`` (FULL).
        q_ranges, k_ranges : Optional[torch.Tensor]
            Shape ``(num_slices, 2)`` int32, each row ``[start, end)``.
        attn_type_map : Optional[torch.Tensor | List[int]]
            Shape ``(num_slices,)`` int32 (or a Python list). See
            :class:`FFAMaskType` for the integer encoding.
            Defaults to all FULL.
        sm_scale : Optional[float]
            Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
        logits_soft_cap : float
        deterministic : bool
        pack_gqa : bool
        disable_fwd_atomic_reduction : bool
            Passed through to MagiAttention for sparse forward paths.
        auto_range_merge : bool
            Passed through to MagiAttention range merging logic.
        ref_block_size : Optional[Tuple[int, int]]
            Optional MagiAttention sparse reference block size.
        max_seqlen_q : Optional[int]
            Optional MagiAttention maximum query sequence length hint.
        swap_ab : bool
            Passed through to MagiAttention sparse/dense kernel selection.
        sparse_load : bool
            Passed through to MagiAttention sparse load path.

        Notes
        -----
        This wrapper supports the single-GPU forward prefill subset used by
        FlashInfer ``backend="ffa"``. Backward, distributed Context Parallel
        execution, attention sinks, and ``return_max_logits`` are intentionally
        outside this wrapper's current scope.
        """
        self._ffa_func = _get_ffa_func()

        # Resolve the two input styles into a single (q_ranges, k_ranges, attn_type_map) triple.
        if q_ranges is not None or k_ranges is not None:
            if qo_indptr is not None or kv_indptr is not None:
                raise ValueError(
                    "Pass either (qo_indptr, kv_indptr) or (q_ranges, k_ranges), not both"
                )
            if q_ranges is None or k_ranges is None:
                raise ValueError("q_ranges and k_ranges must be provided together")
            q_ranges_cpu, q_max_end = _validate_range_tensor(q_ranges, "q_ranges")
            k_ranges_cpu, k_max_end = _validate_range_tensor(k_ranges, "k_ranges")
            if q_ranges_cpu.shape[0] != k_ranges_cpu.shape[0]:
                raise ValueError(
                    "q_ranges and k_ranges must have the same number of slices"
                )
            num_slices = q_ranges_cpu.shape[0]

            q_ranges_dev = q_ranges_cpu.to(device=self.device, dtype=torch.int32)
            k_ranges_dev = k_ranges_cpu.to(device=self.device, dtype=torch.int32)

            if attn_type_map is None:
                attn_type_map_dev = torch.full(
                    (num_slices,),
                    FFAMaskType.FULL,
                    dtype=torch.int32,
                    device=self.device,
                )
            elif isinstance(attn_type_map, torch.Tensor):
                if attn_type_map.ndim != 1:
                    raise ValueError("attn_type_map must be a 1-D tensor")
                if attn_type_map.shape[0] != num_slices:
                    raise ValueError(
                        "attn_type_map length must equal number of slices"
                    )
                attn_type_map_cpu = attn_type_map.to("cpu", dtype=torch.int32)
                if torch.any((attn_type_map_cpu < 0) | (attn_type_map_cpu > 3)):
                    raise ValueError("attn_type_map values must be one of 0, 1, 2, 3")
                attn_type_map_dev = attn_type_map_cpu.to(
                    device=self.device, dtype=torch.int32
                )
            else:
                if len(attn_type_map) != num_slices:
                    raise ValueError(
                        "attn_type_map length must equal number of slices"
                    )
                if any(mask_type not in (0, 1, 2, 3) for mask_type in attn_type_map):
                    raise ValueError("attn_type_map values must be one of 0, 1, 2, 3")
                attn_type_map_dev = torch.tensor(
                    list(attn_type_map), dtype=torch.int32, device=self.device
                )
        else:
            if qo_indptr is None or kv_indptr is None:
                raise ValueError(
                    "Must provide either (qo_indptr, kv_indptr) or (q_ranges, k_ranges)"
                )
            q_ranges_dev, k_ranges_dev, attn_type_map_dev = _indptr_to_ranges(
                qo_indptr, kv_indptr, causal, self.device
            )
            _, q_max_end = _validate_range_tensor(q_ranges_dev, "q_ranges")
            _, k_max_end = _validate_range_tensor(k_ranges_dev, "k_ranges")

        self._q_ranges = q_ranges_dev
        self._k_ranges = k_ranges_dev
        self._attn_type_map = attn_type_map_dev
        self._q_max_end = q_max_end
        self._k_max_end = k_max_end

        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._sm_scale = (
            sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        self._logits_soft_cap = logits_soft_cap
        self._deterministic = deterministic
        self._pack_gqa = pack_gqa
        self._disable_fwd_atomic_reduction = disable_fwd_atomic_reduction
        self._auto_range_merge = auto_range_merge
        self._ref_block_size = ref_block_size
        self._max_seqlen_q = max_seqlen_q
        self._swap_ab = swap_ab
        self._sparse_load = sparse_load
        self._planned = True
        self._validated_run_signature = None

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Execute the planned FFA attention.

        Parameters
        ----------
        q : torch.Tensor
            Shape ``(num_tokens_q, num_qo_heads, head_dim)``.
        k, v : torch.Tensor
            Shape depends on ``kv_layout``:
            NHD -> ``(num_tokens_kv, num_kv_heads, head_dim)``
            HND -> ``(num_kv_heads, num_tokens_kv, head_dim)``
        out : Optional[torch.Tensor]
            Optional output buffer. Allocated internally when ``None``. The FFA
            kernel writes into a freshly-allocated tensor; if ``out`` is
            supplied, the kernel result is copied into it.
        lse : Optional[torch.Tensor]
            Optional LSE buffer, shape ``(num_tokens_q, num_qo_heads)`` float32.
        return_lse : bool

        Returns
        -------
        Either the output tensor, or ``(out, lse)`` when ``return_lse=True``.
        """
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")

        if out is None and lse is None and not return_lse:
            return self.run_fast(q, k, v)

        q, k, v = self._validate_run_inputs(q, k, v)

        if out is not None:
            _check_tensor_shape_dtype_device(
                out, tuple(q.shape), q.dtype, q.device, "out"
            )
        if return_lse and lse is not None:
            _check_tensor_shape_dtype_device(
                lse, (q.shape[0], q.shape[1]), torch.float32, q.device, "lse"
            )

        ffa_out, meta = self._call_ffa_validated(q, k, v)

        if out is not None:
            out.copy_(ffa_out)
        else:
            out = ffa_out

        if return_lse:
            ffa_lse = meta.lse
            if lse is not None:
                lse.copy_(ffa_lse)
            else:
                lse = ffa_lse
            return out, lse
        return out

    def run_fast(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        r"""Execute the planned FFA attention without output or LSE buffers.

        This is the low-overhead plan/run path used by FlashInfer's
        ``backend="ffa"`` integration. It still validates q/k/v against the
        plan, but avoids the extra buffer handling in :meth:`run`.
        """
        if not self._planned:
            raise RuntimeError("plan() must be called before run_fast()")
        q, k, v = self._validate_run_inputs(q, k, v)
        return self._run_validated(q, k, v)

    def _call_ffa_validated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        ffa_func = self._ffa_func
        if ffa_func is None:
            ffa_func = _get_ffa_func()
            self._ffa_func = ffa_func
        return ffa_func(
            q,
            k,
            v,
            q_ranges=self._q_ranges,
            k_ranges=self._k_ranges,
            attn_type_map=self._attn_type_map,
            softmax_scale=self._sm_scale,
            softcap=self._logits_soft_cap,
            deterministic=self._deterministic,
            pack_gqa=self._pack_gqa,
            disable_fwd_atomic_reduction=self._disable_fwd_atomic_reduction,
            ref_block_size=self._ref_block_size,
            max_seqlen_q=self._max_seqlen_q,
            auto_range_merge=self._auto_range_merge,
            swap_ab=self._swap_ab,
            sparse_load=self._sparse_load,
        )

    def _validate_run_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._kv_layout == "HND":
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

        run_signature = (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.dtype,
            k.dtype,
            v.dtype,
            q.device,
            k.device,
            v.device,
            q.is_contiguous(),
            k.is_contiguous(),
            v.is_contiguous(),
        )
        if run_signature == self._validated_run_signature:
            return q, k, v

        _check_qkv_shape_for_ffa_cached(
            q,
            k,
            v,
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
        if q.device != self.device:
            raise ValueError(
                f"q/k/v device ({q.device}) does not match planned device "
                f"({self.device})"
            )
        _check_range_bound("planned q_ranges", self._q_max_end, q.shape[0])
        _check_range_bound("planned k_ranges", self._k_max_end, k.shape[0])
        self._validated_run_signature = run_signature
        return q, k, v

    def _run_validated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        ffa_out, _ = self._call_ffa_validated(q, k, v)
        return ffa_out


__all__ = [
    "FFAMaskType",
    "BatchPrefillFFAWrapper",
    "flex_prefill",
    "causal_prefill",
    "varlen_causal_prefill",
    "causal_ranges",
    "full_ranges",
    "varlen_causal_ranges",
]
