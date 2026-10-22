# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation, scratch sizing and launch of the SM120 DSA indexer backend."""

import functools
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

from ...utils import device_support_pdl, is_sm120f_supported

_CSRC = Path(__file__).resolve().parent / "csrc"
_HEAD_DIM = 128
_CAPACITY = 8192  # candidates a scanned row keeps
_MAX_TOP_K = _CAPACITY // 2
# Mirrors of the kernels' constants: kSampleSlots; kResSlots, kSelClaimers and
# kResList; the resident classes' res_class_cap and res_class_slots.
_SAMPLE_SLOTS = 64 * 128
_RES_SLOTS, _RES_CLAIMERS, _RES_LIST = 4, 11, 2048
_RES_CAP = 16384  # the first resident class's keys
_CLASS_CAPS = (1, 2, 3)
_CLASS_SLOTS = (4, 2, 2)
# The scan runs on chunks of this many rows (a multiple of every block_q), so
# its buffers (64 KB a row) stay bounded whatever the call's rows.
_SCAN_ROWS = 4096


@functools.cache
def _module():
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "sm120_dsa_indexer",
        [_CSRC / "binding.cu"],
        # No fast math: the scores must match DeepGEMM's bit for bit. -uumn:
        # ptxas keeps setmaxnreg in kernels with calls (with the kernels' knob).
        extra_cuda_cflags=[*nvcc_flags, "-O3", "-Xptxas=-uumn"],
        extra_include_paths=[_CSRC],
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
    ).build_and_load()


@functools.cache
def _row_sel_keys() -> int:
    return int(_module().decode_row_sel_keys())


@functools.cache
def _long_words() -> int:
    return int(_module().decode_long_words())


@functools.cache
def _sm_count(dev: torch.device) -> int:
    # The device's: an upper bound for scratch sizing (the kernels size their
    # grids from the launch context's SMs).
    return torch.cuda.get_device_properties(dev).multi_processor_count


def _check_device(dev: torch.device) -> None:
    if dev.type != "cuda" or not is_sm120f_supported(dev):
        raise RuntimeError("backend='sm120' needs an SM12x GPU and CUDA 12.9 or newer")


def _check_common(num_heads: int, top_k: int) -> None:
    if num_heads not in (32, 64):
        raise ValueError("backend='sm120' supports 32 or 64 indexer heads")
    if not 1 <= top_k <= _MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {_MAX_TOP_K}]")


def _outputs(
    indices: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    return_scores: bool,
    num_rows: int,
    top_k: int,
    dev: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The caller's outputs or new ones; no scores with return_scores=False
    (the kernels then skip the score writes)."""
    if indices is None:
        indices = torch.empty(num_rows, top_k, dtype=torch.int32, device=dev)
    if not return_scores:
        if scores is not None:
            raise ValueError("scores given with return_scores=False")
        return indices, None
    if scores is None:
        scores = torch.empty(num_rows, top_k, dtype=torch.float32, device=dev)
    return indices, scores


# Per-device decode buffers, grown as calls need: calls on concurrent streams
# of one device must not overlap. A replaced buffer stays alive for the CUDA
# graphs captured with it.
_LOGITS: dict = {}
_LONG_WS: dict = {}
_RETIRED: list = []


def _decode_logits(
    dev: torch.device, num_rows: int, max_context_len: int
) -> torch.Tensor:
    """The scan's output, [num_rows, stride] float32 with 16-byte rows."""
    key = (dev.type, dev.index)
    stride = (max(max_context_len, 1) + 3) // 4 * 4
    buf = _LOGITS.get(key)
    if buf is None or buf.numel() < num_rows * stride:
        if buf is not None:
            _RETIRED.append(buf)
        buf = torch.empty(max(num_rows * stride, 4), dtype=torch.float32, device=dev)
        _LOGITS[key] = buf
    return buf[: num_rows * stride].view(num_rows, stride)


def _decode_long_ws(dev: torch.device, num_rows: int) -> torch.Tensor:
    """The long rows' selection state (decode_common.cuh), zero at first use
    and kept consistent by the kernels from then on."""
    key = (dev.type, dev.index)
    words = num_rows * _long_words()
    buf = _LONG_WS.get(key)
    if buf is None or buf.numel() < words:
        if torch.cuda.is_current_stream_capturing():
            # Zeroed inside a capture, the state would be zero only once the
            # graph is replayed.
            raise RuntimeError(
                "backend='sm120': run the paged form once outside CUDA graph "
                f"capture with at least {num_rows} rows before capturing it"
            )
        if buf is not None:
            _RETIRED.append(buf)
        buf = torch.zeros(words, dtype=torch.int32, device=dev)
        _LONG_WS[key] = buf
    return buf[:words]


def _routes(
    bound: int,
    num_keys: int,
    num_rows: int,
    num_segments: int,
    offsets: Optional[torch.Tensor],
    ratio: int,
) -> Tuple[int, bool]:
    """(resident classes as a bit mask, scan) the call may need, from its
    shapes and the bound on a row's keys alone."""
    # One segment, its queries the tail of its keys: every row sees at least
    # this many keys.
    least = (
        num_keys - num_rows + 1
        if num_segments == 1 and offsets is None and ratio == 1
        else 0
    )
    classes = 0
    for i, cap in enumerate(_CLASS_CAPS):
        lo = 0 if i == 0 else _RES_CAP * _CLASS_CAPS[i - 1]
        if bound > lo and least <= _RES_CAP * cap:
            classes |= 1 << i
    return classes, bound > _RES_CAP * _CLASS_CAPS[-1]


def _target(top_k: int) -> int:
    """Candidates the threshold aims at: 2 * top_k, at most the geometric mean
    of top_k and the capacity."""
    return min(2 * top_k, int(math.sqrt(top_k * _CAPACITY)))


def _res_cta_words(block_q: int, stride: int, slots: int = _RES_SLOTS) -> int:
    """Resident scratch words a CTA (res_cta_words)."""
    return slots * block_q * stride + _RES_CLAIMERS * (
        2 * _RES_LIST + (stride // 32 + 31) // 32 * 32
    )


def ragged_scratch_bytes(
    num_rows: int,
    top_k: int = 2048,
    num_heads: int = 32,
    device: Optional[torch.device] = None,
) -> int:
    """An upper bound of the scratch bytes a ragged call of ``num_rows`` rows
    allocates, whatever its keys."""
    if num_rows <= 0:
        return 0
    block_q = 128 // num_heads
    q_blocks = (num_rows + block_q - 1) // block_q
    sr = min(num_rows, _SCAN_ROWS)
    # Per scan row: tau, count, sample, candidates, flagged rows, exact state
    # and sample slots; four counters a call.
    scan = 4 * sr * (13 + _SAMPLE_SLOTS + 2 * _CAPACITY) + 4 * 4
    # row_ks, row_ke, tile prefix, class masks.
    rows = 4 * (2 * num_rows + q_blocks + (num_rows + 255) // 256)
    dev = (
        device
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    ctas = min(q_blocks, _sm_count(dev))
    res = max(
        _res_cta_words(block_q, _RES_CAP * cap, slots)
        for cap, slots in zip(_CLASS_CAPS, _CLASS_SLOTS, strict=True)
    )
    return scan + rows + 4 * ctas * res


def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    k_scale: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    top_k: int = 2048,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_k: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    return_scores: bool = True,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The ragged form (see flashinfer.dsa_indexer.dsa_indexer_topk)."""
    fp4 = q_scale is not None
    if fp4:
        if q.dtype not in (torch.uint8, torch.int8) or k.dtype not in (
            torch.uint8,
            torch.int8,
        ):
            raise ValueError("MXFP4 (q_scale given) takes packed uint8 / int8 q and k")
        row_bytes = _HEAD_DIM // 2
    else:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
            raise ValueError("backend='sm120' takes float8_e4m3fn q and k")
        row_bytes = _HEAD_DIM
    if (
        q.dim() != 3
        or q.shape[2] != row_bytes
        or k.dim() != 2
        or k.shape[1] != row_bytes
    ):
        raise ValueError(f"q must be [T, H, {row_bytes}] and k [Tkv, {row_bytes}]")
    num_rows, num_heads, _ = q.shape
    _check_common(num_heads, top_k)
    if ratio < 1:
        raise ValueError("ratio must be positive")
    if max_seqlen_k is not None and max_seqlen_k < 0:
        raise ValueError("max_seqlen_k must be non-negative")
    dev = q.device
    _check_device(dev)
    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    if fp4:
        # Four ue8m0 bytes a key / (row, head), as one int32.
        k_scale = k_scale.contiguous().view(torch.int32).reshape(-1)
        q_scale = q_scale.contiguous().view(torch.int32).reshape(num_rows, num_heads)
    else:
        k_scale = k_scale.contiguous().view(-1)
    indices, scores = _outputs(indices, scores, return_scores, num_rows, top_k, dev)
    # A row sees at most its segment's keys.
    bound = k.shape[0] if max_seqlen_k is None else min(k.shape[0], max_seqlen_k)
    if num_rows == 0:
        return indices, scores
    if bound == 0:
        # No row sees a key: no route runs.
        indices.fill_(-1)
        if scores is not None:
            scores.fill_(-math.inf)
        return indices, scores

    i32 = dict(dtype=torch.int32, device=dev)
    f32 = dict(dtype=torch.float32, device=dev)
    num_segments = cu_seqlens_q.numel() - 1
    classes, scan = _routes(
        bound, k.shape[0], num_rows, num_segments, q_causal_offsets, ratio
    )
    sr = min(num_rows, _SCAN_ROWS) if scan else 0
    row_ks = torch.empty(num_rows, **i32)
    row_ke = torch.empty(num_rows, **i32)
    tau = torch.empty(sr, **f32)
    count = torch.empty(sr, **i32)
    sample = torch.empty(sr, _SAMPLE_SLOTS, **f32)
    cand_score = torch.empty(sr, _CAPACITY, **f32)
    cand_id = torch.empty(sr, _CAPACITY, **i32)
    fallback_rows = torch.empty(2, sr, **i32)
    fallback_count = torch.empty(4, **i32)  # zeroed on the device
    exact_state = torch.empty(sr, 8, **i32)
    sample_slots = torch.empty(sr, **i32)
    block_q = 128 // num_heads
    q_blocks = (num_rows + block_q - 1) // block_q
    tile_prefix = torch.empty(q_blocks, **i32)
    res_scratch, res_bound, row_classes = None, 0, None
    if classes:
        # Score rows as long as the class's cap, or the bound when shorter.
        res_bound = (bound + 31) // 32 * 32
        cta_words = max(
            _res_cta_words(block_q, min(res_bound, _RES_CAP * cap), slots)
            for i, (cap, slots) in enumerate(
                zip(_CLASS_CAPS, _CLASS_SLOTS, strict=True)
            )
            if classes >> i & 1
        )
        res_scratch = torch.empty(min(q_blocks, _sm_count(dev)) * cta_words, **f32)
        # With several classes, a class's launch without rows of its class
        # leaves at once, by the rows' class masks (a word per 256 rows).
        if classes & (classes - 1):
            row_classes = torch.empty((num_rows + 255) // 256, **i32)

    _module().run(
        q,
        k,
        k_scale,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        q_causal_offsets,
        int(ratio),
        int(top_k),
        _CAPACITY,
        _target(top_k),
        row_ks,
        row_ke,
        tau,
        count,
        sample,
        cand_score,
        cand_id,
        fallback_rows,
        fallback_count,
        indices,
        scores,
        tile_prefix,
        exact_state,
        sample_slots,
        q_scale,
        res_scratch,
        row_classes,
        _RES_CAP,
        int(res_bound),
        int(classes),
        bool(scan),
        device_support_pdl(dev) if enable_pdl is None else bool(enable_pdl),
    )
    return indices, scores


def dsa_indexer_topk_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    w: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    top_k: int = 2048,
    max_context_len: int,
    page_rows: Optional[int] = None,
    q_scale: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    return_scores: bool = True,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The paged form (see flashinfer.dsa_indexer.dsa_indexer_topk).

    q ``[B, next_n, H, 128]`` e4m3, or packed MXFP4 ``[B, next_n, H, 64]`` with
    ``q_scale [B, next_n, H]``; ``context_lens [B, next_n]``; ``max_context_len``
    bounds every row's keys and fixes the launches.
    """
    fp4 = q_scale is not None
    if q.dim() != 4:
        raise ValueError("q must be [B, next_n, H, D]")
    batch, next_n, num_heads, _ = q.shape
    row_bytes = _HEAD_DIM // 2 if fp4 else _HEAD_DIM
    if q.shape[3] != row_bytes:
        raise ValueError(f"q must be [B, next_n, H, {row_bytes}]")
    if fp4 and q.dtype not in (torch.uint8, torch.int8):
        raise ValueError("MXFP4 (q_scale given) takes packed uint8 / int8 q")
    if not fp4 and q.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 q must be float8_e4m3fn")
    _check_common(num_heads, top_k)
    if not 1 <= next_n <= 16:
        raise ValueError("the decode form supports next_n 1 to 16")
    num_rows = batch * next_n
    if context_lens.numel() != num_rows:
        raise ValueError("context_lens must be [B, next_n]")
    if page_rows is None:
        page_rows = kv_cache.shape[1]
    dev = q.device
    _check_device(dev)
    q = q.contiguous().view(num_rows, num_heads, row_bytes)
    w = w.contiguous()
    if fp4:
        q_scale = q_scale.contiguous().view(torch.int32).reshape(num_rows, num_heads)
    # The cache as [num_pages, page_stride] bytes (pages may carry a gap).
    page_stride = kv_cache.stride(0) * kv_cache.element_size()
    pages = torch.as_strided(
        kv_cache.view(torch.uint8), (kv_cache.shape[0], page_stride), (page_stride, 1)
    )
    row_len = context_lens.reshape(num_rows).contiguous()
    indices, scores = _outputs(indices, scores, return_scores, num_rows, top_k, dev)
    if num_rows == 0:
        return indices, scores
    # Units of one or two rows of a request are scored per key tile; a
    # request's units share its block table row and stream its keys together
    # (from DRAM once). Pairs halve the key traffic per row and pay on long
    # scans: past 16 rows (8 with MXFP4) or from next_n 4. Fewer next_n = 2
    # rows keep one-row units, whose scan builds the long rows' histograms
    # more cheaply.
    unit = (
        2 if next_n % 2 == 0 and (next_n >= 4 or num_rows > (8 if fp4 else 16)) else 1
    )
    _module().run_decode(
        q,
        pages,
        w,
        row_len,
        block_table,
        int(page_rows),
        unit,
        next_n // unit,
        int(top_k),
        int(max_context_len),
        q_scale,
        indices,
        scores,
        _decode_logits(dev, num_rows, int(max_context_len)),
        _decode_long_ws(dev, num_rows) if max_context_len > _row_sel_keys() else None,
        device_support_pdl(dev) if enable_pdl is None else bool(enable_pdl),
    )
    return indices, scores
