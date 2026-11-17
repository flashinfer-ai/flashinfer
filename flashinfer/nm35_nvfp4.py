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

Attention kernels for Nemotron 3.5 Super with an NVFP4 KV cache (``sm_100a``).

``backend="nm35-nvfp4"`` of
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` and
:func:`flashinfer.prefill.trtllm_batch_context_with_kv_cache` runs each call on
one of three kernels written for this model's attention geometry, or on
``trtllm-gen`` where that is faster or the call is outside the geometry:

* FP8 E4M3 query and output, NVFP4 K/V with FP8 E4M3 block scales;
* 32 query heads, 2 KV heads, head dimension 128;
* 128-token pages.

This module holds the routing rules. They are functions of the call's shape,
of whether it passes ``cum_seq_lens_q`` and, for decode, of whether the kernels
were built with the fused NVFP4 multiply (CUDA 13.4 or newer). A selected kernel
can still decline a call it cannot serve, such as a too-small workspace; that
call then runs on trtllm-gen.
"""

import collections
import functools
import logging
import os
import threading
from typing import Dict, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .utils import get_compute_capability

logger = logging.getLogger(__name__)

BACKEND = "nm35-nvfp4"
# Oldest toolkit the kernels are built and tested with. A prebuilt module runs on
# any driver that serves sm_100a.
MIN_CUDA_TOOLKIT = "13.0"
# From this toolkit on the decode kernels dequantize NVFP4 with one PTX
# instruction per four values. Older assemblers reject it and get a bit-identical
# sequence through FP16 instead (fp4_mul.cuh), which makes decode slower.
FUSED_DEQUANT_CUDA_TOOLKIT = "13.4"

NUM_QO_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128

DECODE_MTP = "mtp"
DECODE_BASE = "base"
_DECODE_KERNEL_IDS = {DECODE_MTP: 0, DECODE_BASE: 1}

# ---------------------------------------------------------------------------
# Routing thresholds. They were measured on GB200 (152 SMs) and are applied to
# every sm_100 device.
# ---------------------------------------------------------------------------

# Calls whose longest KV is at most this many tokens run on trtllm-gen, which is
# as fast or faster there. This holds for decode and for prefill.
TRTLLM_GEN_MAX_KV_LEN = 8192
# The kernels serve at most this many query tokens per request and requests per call.
DECODE_MAX_Q_LEN = 8
DECODE_MAX_BATCH_SIZE = 256
# With 6 query tokens the "mtp" kernel wins only for the smallest batches.
DECODE_Q6_MTP_MAX_BATCH_SIZE = 2

# Built without the fused NVFP4 multiply, the decode kernels win on fewer shapes
# (measured with CUDA 13.0). Per query length: the largest batch they serve at
# any KV, and the KV above which they also serve batches of at least
# NON_FUSED_DECODE_LARGE_BATCH_SIZE (None: never). Query lengths not listed run
# on trtllm-gen.
NON_FUSED_DECODE_LARGE_BATCH_SIZE = 16
NON_FUSED_DECODE_ROUTES: Dict[int, Tuple[int, Optional[int]]] = {
    1: (1, 65536),
    2: (DECODE_MAX_BATCH_SIZE, None),
    3: (1, None),
    4: (1, None),
    5: (2, 32768),
    8: (2, TRTLLM_GEN_MAX_KV_LEN),
}

# The prefill kernel first dequantizes every visible NVFP4 page into the
# workspace, which costs 64 KiB per page plus a fixed 72 MiB. It serves one
# request per call, which keeps that at 584 MiB for a 1M-token KV; calls with
# more requests run on trtllm-gen.
PREFILL_MAX_BATCH_SIZE = 1
PREFILL_MAX_Q_LEN = 32768
# Mirror kPrefillFp8PageBytes and kPrefillSplitReserveBytes of
# include/flashinfer/attention/nm35_nvfp4/host.h, whose gate is the one
# that decides. The formula is repeated here because a server sizes its
# workspace before the kernel module is loaded.
_PREFILL_FP8_PAGE_BYTES = NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
_PREFILL_SPLIT_RESERVE_BYTES = 72 << 20

# Calls can arrive from several threads, and ``counter[key] += 1`` is not atomic.
# An uncontended lock costs far less than the kernel launch it accompanies.
_route_lock = threading.Lock()
_route_counts: Dict[Tuple[str, str], int] = collections.Counter()


def _count_route(stage: str, kernel: str) -> None:
    with _route_lock:
        _route_counts[(stage, kernel)] += 1


@flashinfer_api
def route_counts() -> Dict[Tuple[str, str], int]:
    """Return how many calls each ``(stage, kernel)`` pair served in this process.

    ``stage`` is ``"decode"`` or ``"prefill"``; ``kernel`` is ``"mtp"``,
    ``"base"``, ``"prefill"`` or ``"trtllm-gen"``.
    """
    with _route_lock:
        return dict(_route_counts)


@flashinfer_api
def reset_route_counts() -> None:
    """Clear the counters returned by :func:`route_counts`."""
    with _route_lock:
        _route_counts.clear()


@flashinfer_api
def select_decode_kernel(
    batch_size: int,
    q_len: int,
    max_kv_len: int,
    ragged: bool = False,
    fused: Optional[bool] = None,
) -> Optional[str]:
    """Pick the decode kernel for a call inside the supported geometry.

    Built without the fused NVFP4 multiply (a CUDA toolkit older than 13.4),
    the kernels win on fewer shapes, so fewer calls are routed to them.

    Parameters
    ----------
    batch_size : int
        Requests in the call.
    q_len : int
        Query tokens of the longest request (1 without speculative decoding).
    max_kv_len : int
        KV tokens of the longest request.
    ragged : bool
        Whether the call passes ``cum_seq_lens_q``, so that its requests can
        differ in query length (also between replays of a CUDA graph).
        ``"mtp"`` does not serve such calls, and ``"base"`` runs them one query
        token at a time, so only the lengths ``"base"`` is selected for anyway
        stay off trtllm-gen.
    fused : Optional[bool]
        Whether the kernels use the fused NVFP4 multiply. ``None`` asks the
        kernel module, loading it if a call can reach a kernel.

    Returns
    -------
    Optional[str]
        ``"mtp"``, ``"base"``, or ``None`` when the call should run on trtllm-gen.
    """
    if max_kv_len <= TRTLLM_GEN_MAX_KV_LEN:
        return None
    if q_len < 1 or q_len > DECODE_MAX_Q_LEN:
        return None
    if batch_size < 1 or batch_size > DECODE_MAX_BATCH_SIZE:
        return None
    if q_len in (2, 4, 6, 8) and ragged:
        return None
    if fused is None:
        fused = _fused_fp4_mul()
    if not fused:
        max_batch_size, large_batch_max_kv_len = NON_FUSED_DECODE_ROUTES.get(
            q_len, (0, None)
        )
        if batch_size > max_batch_size and (
            large_batch_max_kv_len is None
            or batch_size < NON_FUSED_DECODE_LARGE_BATCH_SIZE
            or max_kv_len <= large_batch_max_kv_len
        ):
            return None
    if q_len == 6 and batch_size <= DECODE_Q6_MTP_MAX_BATCH_SIZE:
        return DECODE_MTP
    if q_len in (6, 7):
        return None
    if q_len in (2, 4, 8):
        return DECODE_MTP
    return DECODE_BASE  # 1, 3 or 5 query tokens


@flashinfer_api
def select_prefill_kernel(batch_size: int, max_q_len: int, max_kv_len: int) -> bool:
    """Return whether a prefill call inside the supported geometry runs on the
    Nemotron kernel (``True``) or on trtllm-gen (``False``).

    Parameters
    ----------
    batch_size : int
        Requests in the call.
    max_q_len : int
        Query tokens of the longest request.
    max_kv_len : int
        KV tokens of the longest request: its cached prefix plus its query.
    """
    return (
        1 <= batch_size <= PREFILL_MAX_BATCH_SIZE
        and 1 <= max_q_len <= PREFILL_MAX_Q_LEN
        and max_kv_len > TRTLLM_GEN_MAX_KV_LEN
    )


@flashinfer_api
def prefill_workspace_bytes(max_kv_len: int) -> int:
    """Return the ``workspace_buffer`` size the prefill kernel needs, in bytes.

    A prefill call whose workspace is smaller than this runs on trtllm-gen, so
    size the buffer once for the longest KV the server will see. This is a pure
    function: it does not load the kernel module.

    Parameters
    ----------
    max_kv_len : int
        Longest KV a request can attend to, in tokens: its cached prefix plus
        the query chunk. A model's maximum context length is an upper bound.
        The kernel dequantizes exactly this many tokens, so a call must never
        pass a ``max_kv_len`` below the true length of its request.
    """
    if max_kv_len < 1:
        raise ValueError("max_kv_len must be positive")
    pages = (max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    return 2 * pages * _PREFILL_FP8_PAGE_BYTES + _PREFILL_SPLIT_RESERVE_BYTES


@flashinfer_api
def is_supported_device(device: torch.device) -> bool:
    """Return whether the kernels run on ``device`` (compute capability 10.0).

    Parameters
    ----------
    device : torch.device
        A CUDA device.
    """
    return tuple(get_compute_capability(device)) == (10, 0)


def _has_geometry(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    out: torch.Tensor,
) -> bool:
    return (
        k_block_scales is not None
        and query.dtype == torch.float8_e4m3fn
        and out.dtype == torch.float8_e4m3fn
        and k_cache.dtype == torch.uint8
        and query.dim() == 3
        and query.size(1) == NUM_QO_HEADS
        and query.size(2) == HEAD_DIM
        and k_cache.dim() == 4
        and k_cache.size(1) == NUM_KV_HEADS
        and k_cache.size(2) == PAGE_SIZE
        and k_cache.size(3) == HEAD_DIM // 2
        and is_supported_device(query.device)
    )


@functools.cache
def _build_blocker() -> Optional[str]:
    """Return why the kernel module cannot be built here, or ``None`` when it
    is prebuilt or its build can be attempted."""
    from .jit.cpp_ext import get_cuda_version, is_cuda_version_at_least
    from .jit.nm35_nvfp4 import gen_nm35_nvfp4_fmha_module

    # FLASHINFER_NVCC names another compiler than the toolkit's, so the toolkit
    # version says nothing about it.
    if gen_nm35_nvfp4_fmha_module().is_compiled or "FLASHINFER_NVCC" in os.environ:
        return None
    try:
        if is_cuda_version_at_least(MIN_CUDA_TOOLKIT):
            return None
        found = str(get_cuda_version())
    except RuntimeError:
        found = "none"
    return (
        "its kernels are not prebuilt and compiling them needs the CUDA "
        f"{MIN_CUDA_TOOLKIT} toolkit or newer (found {found})"
    )


@functools.cache
def _module():
    """Return the kernel module, or ``None`` when it cannot be built here."""
    from .jit.nm35_nvfp4 import load_nm35_nvfp4_fmha_module

    # Do not start a build that cannot succeed: it would run inside the first
    # routed call and only then fail.
    blocker = _build_blocker()
    if blocker is not None:
        logger.warning(
            "backend=%r is unavailable, so every call runs on trtllm-gen: %s.",
            BACKEND,
            blocker,
        )
        return None
    try:
        module = load_nm35_nvfp4_fmha_module()
    except RuntimeError as error:  # a failed JIT build
        logger.error(
            "backend=%r failed to build, so every call runs on trtllm-gen: %s",
            BACKEND,
            error,
        )
        return None
    if not module.nm35_nvfp4_fused_fp4_mul():
        logger.info(
            "backend=%r was built with a CUDA toolkit older than %s: its decode "
            "kernels use the slower NVFP4 dequantization path and serve fewer "
            "shapes.",
            BACKEND,
            FUSED_DEQUANT_CUDA_TOOLKIT,
        )
    return module


@functools.cache
def _fused_fp4_mul() -> bool:
    """Return whether the decode kernels use the fused NVFP4 multiply. Without
    a kernel module every call runs on trtllm-gen, so the answer is moot."""
    module = _module()
    return module is None or bool(module.nm35_nvfp4_fused_fp4_mul())


def try_decode(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    enable_pdl: bool,
    cum_seq_lens_q: Optional[torch.Tensor],
    plain: bool,
) -> bool:
    """Run a decode call on a Nemotron kernel if one serves it.

    The tensors are the ones ``trtllm_batch_decode_with_kv_cache`` passes to
    trtllm-gen: ``HND`` K/V, ``bmm1_scale`` as a float or a log2 device tensor.
    ``plain`` says the call uses no feature beyond causal dense attention.
    Returns ``False`` without touching ``out`` when the call must run on
    trtllm-gen.
    """
    kernel = None
    if plain and _has_geometry(query, k_cache, k_block_scales, out):
        # A call with cum_seq_lens_q runs as a ragged one even when its rows
        # are uniform now: under a CUDA graph the offsets can change between
        # replays, while the route stays the one chosen at capture.
        ragged = cum_seq_lens_q is not None or query.size(0) != batch_size * max_q_len
        kernel = select_decode_kernel(batch_size, max_q_len, max_kv_len, ragged)
    if kernel is not None and _module() is None:
        kernel = None
    if kernel is not None:
        declined = _module().nm35_nvfp4_decode(
            _DECODE_KERNEL_IDS[kernel],
            out,
            query,
            k_cache,
            v_cache,
            k_block_scales,
            v_block_scales,
            workspace_buffer,
            block_tables,
            seq_lens,
            batch_size,
            max_q_len,
            max_kv_len,
            bmm1_scale,
            bmm2_scale,
            enable_pdl,
            cum_seq_lens_q,
        )
        if declined:
            logger.debug("decode kernel %r declined the call: %s", kernel, declined)
            kernel = None
    _count_route("decode", kernel or "trtllm-gen")
    return kernel is not None


def try_prefill(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    plain: bool,
) -> bool:
    """Run a prefill call on the Nemotron kernel if it serves it.

    Same conventions as :func:`try_decode`. Returns ``False`` without touching
    ``out`` when the call must run on trtllm-gen, which includes a
    ``workspace_buffer`` smaller than :func:`prefill_workspace_bytes`: the
    kernel declines it.
    """
    served = (
        plain
        and _has_geometry(query, k_cache, k_block_scales, out)
        and select_prefill_kernel(batch_size, max_q_len, max_kv_len)
        and _module() is not None
    )
    if served:
        declined = _module().nm35_nvfp4_prefill(
            out,
            query,
            k_cache,
            v_cache,
            k_block_scales,
            v_block_scales,
            workspace_buffer,
            block_tables,
            seq_lens,
            cum_seq_lens_q,
            batch_size,
            max_q_len,
            max_kv_len,
            bmm1_scale,
            bmm2_scale,
        )
        if declined:
            logger.debug("prefill kernel declined the call: %s", declined)
        served = not declined
    _count_route("prefill", "prefill" if served else "trtllm-gen")
    return served


__all__ = [
    "BACKEND",
    "MIN_CUDA_TOOLKIT",
    "FUSED_DEQUANT_CUDA_TOOLKIT",
    "DECODE_BASE",
    "DECODE_MTP",
    "is_supported_device",
    "prefill_workspace_bytes",
    "reset_route_counts",
    "route_counts",
    "select_decode_kernel",
    "select_prefill_kernel",
]
