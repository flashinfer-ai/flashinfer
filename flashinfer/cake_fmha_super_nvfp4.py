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

Cake FMHA routes ``decode_super_nvfp4`` and ``context_super_nvfp4``: kernels for
Nemotron 3.5 Super with an NVFP4 KV cache on ``sm_100a`` (decode ``"mtp"`` and
``"base"``, and prefill).

Every ``backend="cake"`` call with this geometry takes one of the two routes:
FP8 E4M3 query, NVFP4 K/V of 128-token ``HND`` pages with E4M3 block scales,
32 query heads, 2 KV heads, head dimension 128. The route runs the call on one
of the kernels, or on trtllm-gen where trtllm-gen is as fast or the kernels do
not serve the call. With the fused NVFP4 multiply (CUDA 13.4 or newer) the
kernels serve causal dense calls with FP8 output and:

* decode with more than 8192 KV tokens and 1 to 256 requests: ``"mtp"`` for
  2, 4 or 8 query tokens per request and for 6 with at most 2 requests,
  ``"base"`` for 1, 3 or 5, also when the call passes ``cum_seq_lens_q``;
* prefill of a single request with more than 8192 KV tokens, at most 32768
  query tokens and a ``workspace_buffer`` of at least
  :func:`prefill_workspace_bytes`.

Built without the fused multiply, decode routes fewer shapes to the kernels
(``NON_FUSED_DECODE_ROUTES``). A call a kernel would decline, and every call
when the kernel module cannot be built, runs on trtllm-gen.
"""

import functools
import logging
from typing import Dict, Optional, Tuple, Union

import torch

from .utils import get_compute_capability, log2e

logger = logging.getLogger(__name__)

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
PREFILL = "prefill"
# The route's kernel when none of these kernels takes the call.
TRTLLM_GEN = "trtllm-gen"
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

# The prefill kernel serves one request per call: it first dequantizes every
# visible NVFP4 page into the workspace, which costs 64 KiB per page plus a
# fixed 72 MiB (584 MiB for a 1M-token KV).
PREFILL_MAX_BATCH_SIZE = 1
PREFILL_MAX_Q_LEN = 32768
# Mirror kPrefillFp8PageBytes and kPrefillSplitReserveBytes of
# csrc/cake_fmha_super_nvfp4/host.h: a server sizes its workspace before the
# kernel module is loaded.
_PREFILL_FP8_PAGE_BYTES = NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
_PREFILL_SPLIT_RESERVE_BYTES = 72 << 20


def select_decode_kernel(
    batch_size: int,
    q_len: int,
    max_kv_len: int,
    ragged: bool = False,
    fused: Optional[bool] = None,
) -> Optional[str]:
    """Pick the decode kernel for a call inside the geometry.

    Parameters
    ----------
    batch_size : int
        Requests in the call.
    q_len : int
        Query tokens of the longest request (1 without speculative decoding).
    max_kv_len : int
        KV tokens of the longest request.
    ragged : bool
        Whether the call passes ``cum_seq_lens_q``. Only ``"base"`` serves such
        calls, one query token at a time.
    fused : Optional[bool]
        Whether the kernels use the fused NVFP4 multiply. ``None`` asks the
        kernel module.

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
        fused = _uses_fused_fp4_mul()
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


def select_prefill_kernel(
    batch_size: int, max_q_len: int, max_kv_len: int
) -> Optional[str]:
    """Return ``"prefill"`` for a prefill call inside the geometry the kernel
    should serve, or ``None`` when it should run on trtllm-gen.

    ``max_kv_len`` is the longest request's cached prefix plus its query.
    """
    if (
        1 <= batch_size <= PREFILL_MAX_BATCH_SIZE
        and 1 <= max_q_len <= PREFILL_MAX_Q_LEN
        and max_kv_len > TRTLLM_GEN_MAX_KV_LEN
    ):
        return PREFILL
    return None


def prefill_workspace_bytes(max_kv_len: int) -> int:
    """Return the ``workspace_buffer`` size the prefill kernel needs, in bytes.

    A prefill call with a smaller workspace runs on trtllm-gen, so size the
    buffer once for the longest KV the server will see. This does not load the
    kernel module.

    Parameters
    ----------
    max_kv_len : int
        Longest KV a request can attend to, in tokens: its cached prefix plus
        the query chunk. A call must never pass a ``max_kv_len`` below the true
        length of its request.
    """
    if max_kv_len < 1:
        raise ValueError("max_kv_len must be positive")
    pages = (max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    return 2 * pages * _PREFILL_FP8_PAGE_BYTES + _PREFILL_SPLIT_RESERVE_BYTES


def has_geometry(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
) -> bool:
    """Whether a call takes these routes: the geometry on an ``sm_100`` device."""
    return (
        k_block_scales is not None
        and v_block_scales is not None
        and query.dtype == torch.float8_e4m3fn
        and query.dim() == 3
        and tuple(query.shape[1:]) == (NUM_QO_HEADS, HEAD_DIM)
        and k_cache.dtype == torch.uint8
        and v_cache.dtype == torch.uint8
        and k_cache.dim() == 4
        and tuple(k_cache.shape[1:]) == (NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 2)
        and _is_supported_device(query.device)
    )


def _is_supported_device(device: torch.device) -> bool:
    return device.type == "cuda" and tuple(get_compute_capability(device)) == (10, 0)


@functools.cache
def _module():
    """Return the kernel module, or ``None`` when it fails to build."""
    from .jit.cake_fmha_super_nvfp4 import load_cake_fmha_super_nvfp4_module

    try:
        module = load_cake_fmha_super_nvfp4_module()
    except RuntimeError as error:  # a failed JIT build, e.g. CUDA older than 13.0
        logger.error(
            "The Cake FMHA super_nvfp4 kernels failed to build, so backend='cake' "
            "runs those calls on trtllm-gen: %s",
            error,
        )
        return None
    if not module.super_nvfp4_fused_fp4_mul():
        logger.info(
            "The Cake FMHA super_nvfp4 kernels were built with a CUDA toolkit "
            "older than %s: decode uses the slower NVFP4 dequantization path.",
            FUSED_DEQUANT_CUDA_TOOLKIT,
        )
    return module


@functools.cache
def _uses_fused_fp4_mul() -> bool:
    """Whether the decode kernels use the fused NVFP4 multiply. True without a
    module: every call then runs on trtllm-gen."""
    module = _module()
    return module is None or bool(module.super_nvfp4_fused_fp4_mul())


def select_decode_route(
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
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    enable_pdl: bool,
    dense_causal: bool,
) -> Optional[str]:
    """Return the kernel of a decode call with the geometry (``"mtp"``,
    ``"base"`` or ``"trtllm-gen"``), or ``None`` for any other call.

    ``dense_causal`` says the call uses no feature beyond causal dense attention
    with the shared page-index layout. A kernel is chosen only when it accepts
    the call (the module's ``super_nvfp4_decode_reason``).
    """
    if not has_geometry(query, k_cache, v_cache, k_block_scales, v_block_scales):
        return None
    # A call with cum_seq_lens_q runs as a ragged one even when its rows are
    # uniform now: under a CUDA graph the offsets can change between replays.
    ragged = cum_seq_lens_q is not None or query.size(0) != batch_size * max_q_len
    kernel = None
    if dense_causal:
        kernel = select_decode_kernel(batch_size, max_q_len, max_kv_len, ragged)
    if kernel is not None:
        module = _module()
        if module is None or module.super_nvfp4_decode_reason(
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
        ):
            kernel = None
    return kernel or TRTLLM_GEN


def select_prefill_route(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
    workspace_buffer: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    dense_causal: bool,
) -> Optional[str]:
    """Return the kernel of a prefill call with the geometry (``"prefill"`` or
    ``"trtllm-gen"``), or ``None`` for any other call. Same conventions as
    :func:`select_decode_route`."""
    if not has_geometry(query, k_cache, v_cache, k_block_scales, v_block_scales):
        return None
    kernel = None
    if dense_causal and workspace_buffer is not None and cum_seq_lens_q is not None:
        kernel = select_prefill_kernel(batch_size, max_q_len, max_kv_len)
    if kernel is not None:
        module = _module()
        if module is None or module.super_nvfp4_prefill_reason(
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
        ):
            kernel = None
    return kernel or TRTLLM_GEN


def run_decode(
    kernel: str,
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    enable_pdl: bool,
) -> None:
    """Launch a decode kernel on the current stream. ``bmm1_scale`` is a float
    or a log2 device tensor, as trtllm-gen takes it."""
    declined = _module().super_nvfp4_decode(
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
        raise RuntimeError(
            f"Cake FMHA decode_super_nvfp4 {kernel!r} declined the call its route "
            f"selected: {declined}"
        )


def run_prefill(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
) -> None:
    """Launch the prefill kernel on the current stream. A device
    ``bmm1_scale`` is the raw scale Cake's context API receives."""
    if isinstance(bmm1_scale, torch.Tensor):
        # The kernel reads the log2 form, as trtllm-gen does.
        bmm1_scale = bmm1_scale * log2e
    declined = _module().super_nvfp4_prefill(
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
        raise RuntimeError(
            "Cake FMHA context_super_nvfp4 declined the call its route selected: "
            f"{declined}"
        )
