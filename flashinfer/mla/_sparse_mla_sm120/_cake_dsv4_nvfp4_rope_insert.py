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

"""Cake SM120 DeepSeek-V4 NVFP4 cache writers: fused GPT-J RoPE + NVFP4 quantize + paged insert.

The device code is generated from the Cake kernel schedules into
``csrc/cake_dsv4/sm_120a`` (``cake_dsv4_nvfp4_rope_insert_*``); this module
owns the host side: tensor validation, cache geometry (runtime page size and
page stride for the 3-D / HND / NHD layouts, padded pools), the ``q_out``
allocation and the two public entries. The writers produce the 384-byte
NVFP4 record read by
:func:`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` with
``backend="cake"`` and ``kv_cache_format="nvfp4"`` (and by the SM120
``"sparse"`` backend), byte for byte what
:func:`flashinfer.mla.nvfp4_quantize_append_sparse_mla_cache` writes for the
same roped BF16 row, in one launch per layer instead of a torch-level RoPE
followed by the append.

Two entries share one kernel body:

* :func:`cake_dsv4_nvfp4_rope_quantize_insert` -- the target-model write of
  the sliding-window pool: rotates ``q`` into a head-padded ``q_out``,
  rotates and quantizes ``kv`` and inserts it by physical slot (the
  semantics of the fused FP8 insert op of the serving stack, Q RoPE
  applied, no Q norm, head-major Q).
* :func:`cake_dsv4_nvfp4_kv_rope_quantize_insert` -- KV only, for the
  compressed pool (``compress_ratio`` 1 or 2, boundary rows only) and for
  speculative-decoding context inserts (``compress_ratio=1``).

Record layout (``Dsv4Nvfp4Layout``): per physical page of ``page_size``
tokens, ``page_size * 352`` data bytes -- token ``t`` at ``t * 352``: 224 bytes
of packed E2M1 for the 448 NoPE dims (even index in the low nibble) followed
by the 64 RoPE values as 128 bytes of BF16 bits -- then ``page_size * 32``
scale bytes -- token ``t`` at ``page_size * 352 + t * 32``: 28 E4M3 group-16
scales and 4 bytes written as zero. The page stride (``cache.stride(0)``) may
exceed ``page_size * 384`` (padded pools) and must be a multiple of 16 bytes.
Public views: 3-D ``[num_pages, page_size, 384]``, HND
``[num_pages, 1, page_size, 384]`` or NHD ``[num_pages, page_size, 1, 384]``.

RoPE semantics: GPT-J pairs ``(2p, 2p + 1)`` on the last 64 dims
(``448 .. 511``) with the fp32 ``cos_sin_cache[pos]`` row (``cos[32] | sin[32]``,
pair ``p`` uses column ``p - 224``), all arithmetic in fp32:
``even = x_even * cos - x_odd * sin``, ``odd = x_even * sin + x_odd * cos``.
The roped KV row is rounded to BF16 (round-to-nearest-even) *before*
quantization; the 64 RoPE values are stored as those BF16 bits and the 448
NoPE values are quantized exactly like the append writer (group-16 ``amax``,
E4M3 ``amax / 6`` scale, E2M1 codes). ``q_out`` holds the fp32 rotation of
each live head rounded to BF16 (or a BF16 copy when ``apply_q_rope=False``);
padded heads are zero.

Boundary rule (KV entry): row ``i`` is inserted when ``i < slot_mapping.numel()``,
``0 <= slot_mapping[i] < num_pages * page_size`` and
``(positions[i] + 1) % compress_ratio == 0``; its cos/sin row is
``positions[i] // compress_ratio * compress_ratio``. ``compress_ratio=1``
inserts every addressed row at its own position. The QKV entry uses ratio 1
and additionally processes every Q row, including rows
``>= slot_mapping.numel()`` (data-parallel padding).

Contract: both entries are CUDA-Graph safe -- the only allocation is
``q_out`` (``torch.empty``, like the serving stack's fused op), there is no
host synchronisation and no data-dependent host decision, so a captured
graph replays bitwise. Negative and out-of-range slots are skipped, only the
addressed records are written, ``positions`` must index inside
``cos_sin_cache`` (not checked on the device) and a valid slot must not
repeat inside one call (caller precondition inherited from the serving
stack). Interleaved Q layouts are out of scope.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from ...api_logging import flashinfer_api
from ...jit.cake_dsv4_nvfp4_rope_insert import (
    cake_dsv4_nvfp4_rope_insert_available,
    cake_dsv4_nvfp4_rope_insert_manifest,
    gen_cake_dsv4_nvfp4_rope_insert_module,
)
from ...utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_HEAD_DIM = 512
_ROPE_DIM = 64
_BYTES_PER_TOKEN = 384
_DATA_BYTES_PER_TOKEN = 352
_SCALE_BYTES_PER_TOKEN = 32
_Q_HEAD_PADDED_CHOICES = (8, 16, 32, 64, 128)
_COMPRESS_RATIOS = (1, 2)
_SLOT_DTYPES = (torch.int32, torch.int64)
_VECTOR_ALIGNMENT = 16


def cake_dsv4_nvfp4_rope_insert_format_info() -> Dict[str, object]:
    """Static facts of the fused writers (record geometry, dispatch choices, provenance).

    Reads the generated manifest; ``kernels_available`` is ``False`` while the tree only holds
    the placeholder manifest (every launch then raises ``FileNotFoundError``).
    """

    manifest = cake_dsv4_nvfp4_rope_insert_manifest()
    return {
        "head_dim": int(manifest["head_dim"]),
        "rope_dim": int(manifest["rope_dim"]),
        "bytes_per_token": int(manifest["bytes_per_token"]),
        "data_bytes_per_token": int(manifest["data_bytes_per_token"]),
        "scale_bytes_per_token": int(manifest["scale_bytes_per_token"]),
        "q_head_padded_choices": tuple(
            int(h) for h in manifest["q_head_padded_choices"]
        ),
        "compress_ratios": tuple(int(r) for r in manifest["compress_ratios"]),
        "slot_dtypes": tuple(str(d) for d in manifest["slot_dtypes"]),
        "threads": int(manifest["threads"]),
        "entries": dict(manifest["entries"]),
        "identity": manifest["identity"],
        "kernel_commit": manifest["kernel_commit"],
        "kernels_available": cake_dsv4_nvfp4_rope_insert_available(),
    }


# ----------------------------------------------------------------------------- validation


def _check_cuda(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")


def _check_aligned(tensor: torch.Tensor, name: str) -> None:
    if tensor.data_ptr() % _VECTOR_ALIGNMENT:
        raise ValueError(f"{name} must be {_VECTOR_ALIGNMENT}-byte aligned")


def _cache_geometry(cache: torch.Tensor) -> Tuple[int, int, int]:
    """``(num_pages, page_size, page_stride_bytes)`` of a 3-D / HND / NHD NVFP4 cache view."""

    _check_cuda(cache, "cache")
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim not in (3, 4) or cache.shape[-1] != _BYTES_PER_TOKEN:
        raise ValueError(
            f"cache must be [num_pages, page_size, {_BYTES_PER_TOKEN}], HND "
            f"[num_pages, 1, page_size, {_BYTES_PER_TOKEN}] or NHD "
            f"[num_pages, page_size, 1, {_BYTES_PER_TOKEN}], got shape={tuple(cache.shape)}"
        )
    if cache.ndim == 3:
        page_dim = 1
    elif cache.shape[1] == 1:
        page_dim = 2
    elif cache.shape[2] == 1:
        page_dim = 1
    else:
        raise ValueError(
            "cache must have a singleton latent-head dimension at axis 1 or 2"
        )
    num_pages = int(cache.shape[0])
    page_size = int(cache.shape[page_dim])
    if num_pages < 1 or page_size < 1:
        raise ValueError("cache must hold at least one page with at least one row")
    if cache.stride(-1) != 1 or cache.stride(page_dim) != _BYTES_PER_TOKEN:
        raise ValueError(
            "cache entries must be contiguous inside each page with strides "
            f"(..., {_BYTES_PER_TOKEN}, 1), got {cache.stride()}"
        )
    page_stride = int(cache.stride(0))
    if page_stride < page_size * _BYTES_PER_TOKEN:
        raise ValueError(
            f"cache page stride {page_stride} is smaller than the logical "
            f"{page_size * _BYTES_PER_TOKEN}-byte page"
        )
    if page_stride % _VECTOR_ALIGNMENT:
        raise ValueError(
            f"cache page stride must be a multiple of {_VECTOR_ALIGNMENT} bytes, got {page_stride}"
        )
    _check_aligned(cache, "cache")
    return num_pages, page_size, page_stride


def _check_q(q: torch.Tensor) -> int:
    _check_cuda(q, "q")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must have dtype torch.bfloat16, got {q.dtype}")
    if q.ndim != 3 or q.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"q must be [num_tokens, num_heads, {_HEAD_DIM}], got shape={tuple(q.shape)}"
        )
    if q.shape[1] < 1:
        raise ValueError("q must hold at least one head")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    _check_aligned(q, "q")
    return int(q.shape[0])


def _check_kv(kv: torch.Tensor, num_tokens: int | None) -> int:
    _check_cuda(kv, "kv")
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"kv must have dtype torch.bfloat16, got {kv.dtype}")
    if kv.ndim != 2 or kv.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"kv must be [num_tokens, {_HEAD_DIM}], got shape={tuple(kv.shape)}"
        )
    if not kv.is_contiguous():
        raise ValueError("kv must be contiguous")
    if num_tokens is not None and kv.shape[0] != num_tokens:
        raise ValueError(
            f"kv must hold one row per query token ({num_tokens}), got {kv.shape[0]}"
        )
    _check_aligned(kv, "kv")
    return int(kv.shape[0])


def _check_slot_mapping(slot_mapping: torch.Tensor, num_tokens: int) -> int:
    _check_cuda(slot_mapping, "slot_mapping")
    if slot_mapping.dtype not in _SLOT_DTYPES:
        raise ValueError(
            "slot_mapping must have dtype torch.int32 or torch.int64, got "
            f"{slot_mapping.dtype}"
        )
    if slot_mapping.ndim != 1 or not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be a contiguous 1D tensor")
    if slot_mapping.numel() > num_tokens:
        raise ValueError(
            f"slot_mapping holds {slot_mapping.numel()} entries but only {num_tokens} "
            "token rows were given; slot_mapping may be shorter than positions "
            "(padded rows are not inserted), never longer"
        )
    return int(slot_mapping.numel())


def _check_positions(positions: torch.Tensor, num_tokens: int) -> None:
    _check_cuda(positions, "positions")
    if positions.dtype != torch.int64:
        # The kernels read int64 positions (the dtype the serving stack passes); converting an
        # int32 tensor here would add a hidden allocation and launch to the hot path.
        raise ValueError(
            f"positions must have dtype torch.int64, got {positions.dtype}; convert with "
            "positions.to(torch.int64) outside the hot path"
        )
    if positions.ndim != 1 or not positions.is_contiguous():
        raise ValueError("positions must be a contiguous 1D tensor")
    if positions.numel() != num_tokens:
        raise ValueError(
            f"positions must hold one entry per token row ({num_tokens}), got {positions.numel()}"
        )


def _check_cos_sin_cache(cos_sin_cache: torch.Tensor) -> None:
    _check_cuda(cos_sin_cache, "cos_sin_cache")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError(
            f"cos_sin_cache must have dtype torch.float32, got {cos_sin_cache.dtype}"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != _ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache must be [max_position, {_ROPE_DIM}] (cos[32] | sin[32]), "
            f"got shape={tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.shape[0] < 1:
        raise ValueError("cos_sin_cache must hold at least one position")
    if not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous")
    _check_aligned(cos_sin_cache, "cos_sin_cache")


def _check_same_device(**tensors: torch.Tensor) -> None:
    devices = {name: tensor.device for name, tensor in tensors.items()}
    if len(set(devices.values())) > 1:
        raise ValueError(
            "all tensors must be on the same CUDA device, got "
            + ", ".join(f"{name}={device}" for name, device in devices.items())
        )


# ----------------------------------------------------------------------------- module


@functools.cache
def get_cake_dsv4_nvfp4_rope_insert_module():
    """Build (or load) the generated writers and wrap the two FFI entries as custom ops."""

    manifest = cake_dsv4_nvfp4_rope_insert_manifest()
    module = gen_cake_dsv4_nvfp4_rope_insert_module().build_and_load()
    qkv_entry = getattr(module, manifest["entries"]["qkv"])
    kv_entry = getattr(module, manifest["entries"]["kv"])

    @register_custom_op(
        "flashinfer::cake_dsv4_nvfp4_rope_insert_qkv",
        mutates_args=("q_out", "cache"),
    )
    def _qkv(
        q: torch.Tensor,
        q_out: torch.Tensor,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        apply_q_rope: bool,
    ) -> None:
        qkv_entry(
            q, q_out, kv, cache, slot_mapping, positions, cos_sin_cache, apply_q_rope
        )

    @register_fake_op("flashinfer::cake_dsv4_nvfp4_rope_insert_qkv")
    def _fake_qkv(*_args, **_kwargs) -> None:
        return None

    @register_custom_op(
        "flashinfer::cake_dsv4_nvfp4_rope_insert_kv",
        mutates_args=("cache",),
    )
    def _kv(
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        compress_ratio: int,
    ) -> None:
        kv_entry(kv, cache, slot_mapping, positions, cos_sin_cache, compress_ratio)

    @register_fake_op("flashinfer::cake_dsv4_nvfp4_rope_insert_kv")
    def _fake_kv(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(qkv=_qkv, kv=_kv, raw_qkv=qkv_entry, raw_kv=kv_entry)


# ----------------------------------------------------------------------------- public API


@supported_compute_capability([120, 121])
@flashinfer_api
def cake_dsv4_nvfp4_rope_quantize_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    q_head_padded: int = 0,
    apply_q_rope: bool = True,
) -> torch.Tensor:
    r"""Rotate ``q``, then rotate, quantize and insert ``kv`` into the DeepSeek-V4 NVFP4 cache.

    One launch replaces the torch-level GPT-J RoPE of the query and latent KV,
    the head padding of the query and
    :func:`flashinfer.mla.nvfp4_quantize_append_sparse_mla_cache`: the cache
    bytes are identical to appending the BF16-rounded roped row, and ``q_out``
    is the fp32 rotation of each live head rounded to BF16 with zero-filled
    padded heads. Rows ``>= slot_mapping.numel()`` (data-parallel padding) get
    their ``q_out`` rows but are not inserted; negative and out-of-range slots
    are skipped.

    Parameters
    ----------
    q : torch.Tensor
        Contiguous CUDA BF16 ``[num_tokens, num_heads, 512]`` queries (448 NoPE
        dims, then 64 RoPE dims), head-major.
    kv : torch.Tensor
        Contiguous CUDA BF16 ``[num_tokens, 512]`` latent KV rows, one per query
        token (unrotated).
    cache : torch.Tensor
        Destination opaque uint8 NVFP4 paged cache: 3-D
        ``[num_pages, page_size, 384]``, HND ``[num_pages, 1, page_size, 384]``
        or NHD ``[num_pages, page_size, 1, 384]``, any positive page size,
        16-byte-multiple page stride (padded pools accepted). Only the
        addressed data rows and scale slots are written.
    slot_mapping : torch.Tensor
        Contiguous 1-D CUDA int32 or int64 tensor with at most ``num_tokens``
        entries; ``slot_mapping[i] = page_id * page_size + entry_id`` for row
        ``i``. Negative and out-of-range entries are padding and skipped; a
        valid slot must not repeat inside one call.
    positions : torch.Tensor
        Contiguous 1-D CUDA int64 ``[num_tokens]`` positions indexing
        ``cos_sin_cache`` (``positions[i] < cos_sin_cache.shape[0]`` is a caller
        precondition; it is not checked on the device). int32 is rejected
        rather than converted, so the hot path stays free of hidden launches.
    cos_sin_cache : torch.Tensor
        Contiguous CUDA fp32 ``[max_position, 64]`` table, ``cos[32] | sin[32]``
        per position (pair ``p`` of the 64 RoPE dims uses column ``p``).
    q_head_padded : int
        Padded head count of ``q_out``: one of 8, 16, 32, 64, 128 and at
        least ``num_heads``; ``0`` (default) writes the cache only -- identical to
        :func:`cake_dsv4_nvfp4_kv_rope_quantize_insert` with ``compress_ratio=1`` --
        and returns an empty ``[num_tokens, 0, 512]`` tensor.
    apply_q_rope : bool
        Rotate the live query heads (default). ``False`` copies them unchanged
        (padded heads are still zero-filled).

    Returns
    -------
    torch.Tensor
        ``q_out`` -- a new BF16 ``[num_tokens, q_head_padded, 512]`` tensor
        (``torch.empty``, fully written). The cache is updated in place.
        ``num_tokens == 0`` returns the empty tensor without a launch.
    """
    num_tokens = _check_q(q)
    _check_kv(kv, num_tokens)
    _cache_geometry(cache)
    num_insert = _check_slot_mapping(slot_mapping, num_tokens)
    _check_positions(positions, num_tokens)
    _check_cos_sin_cache(cos_sin_cache)
    _check_same_device(
        q=q,
        kv=kv,
        cache=cache,
        slot_mapping=slot_mapping,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
    )
    num_heads = int(q.shape[1])
    q_head_padded = int(q_head_padded)
    if q_head_padded == 0:
        q_out = torch.empty(
            (num_tokens, 0, _HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
        if num_tokens == 0 or num_insert == 0:
            return q_out
        get_cake_dsv4_nvfp4_rope_insert_module().kv(
            kv, cache, slot_mapping, positions, cos_sin_cache, 1
        )
        return q_out
    if q_head_padded not in _Q_HEAD_PADDED_CHOICES:
        raise ValueError(
            f"q_head_padded must be 0 or one of {_Q_HEAD_PADDED_CHOICES}, got {q_head_padded}"
        )
    if q_head_padded < num_heads:
        raise ValueError(
            f"q_head_padded ({q_head_padded}) must be at least the number of query heads "
            f"({num_heads})"
        )
    q_out = torch.empty(
        (num_tokens, q_head_padded, _HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    if num_tokens == 0:
        return q_out
    get_cake_dsv4_nvfp4_rope_insert_module().qkv(
        q, q_out, kv, cache, slot_mapping, positions, cos_sin_cache, bool(apply_q_rope)
    )
    return q_out


@supported_compute_capability([120, 121])
@flashinfer_api
def cake_dsv4_nvfp4_kv_rope_quantize_insert(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    compress_ratio: int = 1,
) -> None:
    r"""Rotate, quantize and insert latent KV rows into the DeepSeek-V4 NVFP4 cache (KV only).

    The compressed-pool and speculative-context writer: row ``i`` is inserted
    when ``i < slot_mapping.numel()``, its slot is valid and
    ``(positions[i] + 1) % compress_ratio == 0``; the RoPE row is
    ``cos_sin_cache[positions[i] // compress_ratio * compress_ratio]``. With
    ``compress_ratio=1`` every addressed row is inserted at its own position.
    The cache bytes equal
    :func:`flashinfer.mla.nvfp4_quantize_append_sparse_mla_cache` applied to
    the BF16-rounded roped rows.

    Parameters
    ----------
    kv : torch.Tensor
        Contiguous CUDA BF16 ``[num_tokens, 512]`` latent KV rows (unrotated).
    cache : torch.Tensor
        Destination opaque uint8 NVFP4 paged cache (3-D, HND or NHD; any
        positive page size; 16-byte-multiple page stride). Only the addressed
        data rows and scale slots are written.
    slot_mapping : torch.Tensor
        Contiguous 1-D CUDA int32 or int64 tensor with at most ``num_tokens``
        entries; negative and out-of-range slots are skipped, and the caller
        may pass ``-1`` for non-boundary rows instead of relying on
        ``compress_ratio``. A valid slot must not repeat inside one call.
    positions : torch.Tensor
        Contiguous 1-D CUDA int64 ``[num_tokens]`` positions; every inserted
        row's cos/sin row must index inside ``cos_sin_cache`` (not checked on
        the device). int32 is rejected rather than converted.
    cos_sin_cache : torch.Tensor
        Contiguous CUDA fp32 ``[max_position, 64]`` table, ``cos[32] | sin[32]``.
    compress_ratio : int
        1 or 2: the boundary rule and the cos/sin row rule above.

    Returns
    -------
    None
        The cache is updated in place; ``num_tokens == 0`` or an empty
        ``slot_mapping`` returns without a launch. CUDA-Graph safe (no
        allocation, no host synchronisation).
    """
    num_tokens = _check_kv(kv, None)
    _cache_geometry(cache)
    num_insert = _check_slot_mapping(slot_mapping, num_tokens)
    _check_positions(positions, num_tokens)
    _check_cos_sin_cache(cos_sin_cache)
    _check_same_device(
        kv=kv,
        cache=cache,
        slot_mapping=slot_mapping,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
    )
    compress_ratio = int(compress_ratio)
    if compress_ratio not in _COMPRESS_RATIOS:
        raise ValueError(
            f"compress_ratio must be one of {_COMPRESS_RATIOS}, got {compress_ratio}"
        )
    if num_tokens == 0 or num_insert == 0:
        return
    get_cake_dsv4_nvfp4_rope_insert_module().kv(
        kv, cache, slot_mapping, positions, cos_sin_cache, compress_ratio
    )


__all__ = [
    "cake_dsv4_nvfp4_kv_rope_quantize_insert",
    "cake_dsv4_nvfp4_rope_insert_format_info",
    "cake_dsv4_nvfp4_rope_quantize_insert",
    "get_cake_dsv4_nvfp4_rope_insert_module",
]
