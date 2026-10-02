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

"""Cake SM120 DeepSeek-V4 NVFP4 sparse MLA (``backend="cake"`` on SM120/SM121): decode + prefill.

The device code is generated from the Cake kernel schedules into
``csrc/cake_dsv4/sm_120a``; this module owns the host side: cache geometry
(runtime page size and page stride for the 3-D / HND / NHD layouts, padded
pools), the decode split / head-tile planners, the prefill planner (head
tiles), the decode-vs-prefill crossover for ``num_tokens > 1``,
caller-owned split scratch, the public-API workspace carve and the
``SparseMLASm120Wrapper`` route.

Two kernel families share one tensor contract:

* **decode** (``cake_sparse_mla_sm120_dsv4_nvfp4_decode``): one CTA per
  (token, 16- or 32-head block, split of 64-candidate chunks), split-K
  partials merged by a second launch; any candidate count.
* **prefill** (``cake_sparse_mla_sm120_dsv4_nvfp4_prefill``): one CTA per
  (token, ``16 * head_tiles``-head block) over *all* of the token's chunks
  (direct epilogue, no scratch, no merge), ``head_tiles`` in {1, 2, 4} (2 for
  head counts divisible by 32, 4 for head counts divisible by 64); at most 16
  chunks (``topk + extra_topk <= 1024`` slots) per token.

Contract (shared with the SM120 NVFP4 ``"sparse"`` backend):

* ``q`` ``[T, H, 512]`` BF16; ``output`` ``[T, H, 512]`` BF16; ``out_lse``
  ``[T, H]`` fp32 base-2 (times ``lse_scale``).
* ``kv_cache`` / ``extra_kv_cache``: packed NVFP4 pages (``page_size * 352``
  data bytes followed by ``page_size * 32`` scale bytes) as ``[P, page, 384]``,
  HND ``[P, 1, page, 384]`` or NHD ``[P, page, 1, 384]`` uint8 views; any
  positive page size; the page stride may exceed the payload (16-byte multiple).
* ``indices`` / ``extra_indices`` ``[T, topk]`` (or ``[T, 1, topk]``) int32,
  ``-1`` masks a slot; ``topk_length`` / ``extra_topk_length`` ``[T]`` int32;
  ``attn_sink`` ``[H]`` fp32 (sigmoid gate of the output, logaddexp into LSE).
* An empty row (no valid slot in either cache) writes zeros and ``-inf`` LSE,
  or the sink-only LSE when ``attn_sink`` is given.
* Split-K: ``num_splits`` CTAs per (token, 16-head block) write BF16 /
  fp32 partials to caller-owned ``mid_out`` ``[T, H, S, 512]`` /
  ``mid_lse`` ``[T, H, S]`` (``S >= num_splits``); a merge launch combines
  them.  ``num_splits == 1`` writes the output directly.  Allocation-free
  and CUDA-graph safe when the scratch is caller-owned.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from ...jit.cake_sparse_mla_sm120_dsv4_nvfp4 import (
    cake_sparse_mla_sm120_dsv4_nvfp4_manifest,
    gen_cake_sparse_mla_sm120_dsv4_nvfp4_module,
)
from ...utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_D_QK = 512
_D_V = 512
_BYTES_PER_TOKEN = 384
_CHUNK = 64
# Below this SM count the planners follow the GB10 (SM121, 48 SMs) sweeps instead of the
# GB202 (RTX PRO 6000 / RTX 5090, 188 / 170 SMs) ones.
_SMALL_DIE_SMS = 64


def _manifest() -> dict:
    return cake_sparse_mla_sm120_dsv4_nvfp4_manifest()


def cake_sparse_mla_sm120_dsv4_nvfp4_format_info() -> dict:
    """Static facts of the Cake SM120 NVFP4 route (mirrors ``dsv4_nvfp4_format_info``)."""

    manifest = _manifest()
    return {
        "query_dim": int(manifest["head_dim"]),
        "value_dim": int(manifest["value_dim"]),
        "bytes_per_token": int(manifest["bytes_per_token"]),
        "chunk_width": int(manifest["candidates_per_chunk"]),
        "heads_per_block": int(manifest["heads_per_block"]),
        "max_chunks_per_block": int(manifest["max_chunks_per_block"]),
        "heads": tuple(int(h) for h in manifest["head_counts"]),
        "two_tile_heads": tuple(
            int(h) for h in manifest.get("two_tile_head_counts", ())
        ),
        "runtime_page": True,
        "runtime_extra_page": True,
        "kernel_commit": manifest["kernel_commit"],
        # Prefill family: head counts with an instance, the valid head-tile counts per head count and the
        # per-token chunk limit of the single-CTA (no split) schedule.
        "prefill_heads": tuple(int(h) for h in manifest.get("prefill_head_counts", ())),
        "prefill_head_tiles": {
            int(h): tuple(int(t) for t in tiles)
            for h, tiles in manifest.get("prefill_head_tiles", {}).items()
        },
        "prefill_max_chunks": int(
            manifest.get("prefill_max_chunks", manifest["max_chunks_per_block"])
        ),
        "prefill_kernel_commit": manifest.get("prefill_kernel_commit"),
    }


def cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads() -> Tuple[int, ...]:
    return tuple(int(h) for h in _manifest()["head_counts"])


def cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk: int, extra_topk: int = 0) -> int:
    """64-candidate chunks of one token: the upper bound on ``num_splits``."""

    return (int(topk) + _CHUNK - 1) // _CHUNK + (int(extra_topk) + _CHUNK - 1) // _CHUNK


def cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
) -> int:
    """Return the number of 16-head tiles per decode CTA (1 or 2).

    Two tiles (32 heads per CTA sharing one candidate gather) follow the
    measured one-tile vs two-tile sweeps on RTX PRO 6000, with ``ctas`` the
    one-tile grid ``num_tokens * num_heads / 16``:

    * ``ctas >= SMs`` (a full wave or more): two tiles for every head count;
    * below one wave the halved grid costs SM coverage and the split planner's
      extra split costs a merge, so the shared gather has to pay for it:
      H = 32 pairs at ``ctas >= 2/3 SMs``, or at ``ctas >= SMs / 3`` with at
      least 8 chunks; H = 64 pairs only for ``SMs / 3 <= ctas < 2/3 SMs`` with at
      least 8 chunks; H >= 96 pairs only with at least 16 chunks at
      ``ctas >= SMs / 3``.

    GB10 (SM121, 48 SMs; ``num_sms < 64``) differs only at a full wave with two
    chunks: a two-chunk CTA gathers too little to pay for the doubled serial MMA
    of two tiles once the one-tile grid is more than two waves (H = 64 and
    H = 128 at 32 tokens: one tile 2-3 % faster), while up to two waves the
    pair still folds the partial second wave into one resident wave (H = 128 at
    8 tokens: 1.04-1.07x) and H = 32 always pairs because one CTA then covers
    the token's whole head set (1.05-1.07x).  Rows with >= 4 chunks pair at a
    full wave on both; the sub-wave rules are the same on both.

    Head counts not divisible by 32 have no two-tile instance.
    """

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    hpb = info["heads_per_block"]
    num_heads = int(num_heads)
    num_sms = int(num_sms)
    if num_heads % (2 * hpb) != 0:
        return 1
    ctas = int(num_tokens) * (num_heads // hpb)
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if ctas >= num_sms:
        if (
            num_sms < _SMALL_DIE_SMS
            and chunks <= 2
            and num_heads > 2 * hpb
            and ctas > 2 * num_sms
        ):
            return 1
        return 2
    third = num_sms // 3
    two_thirds = (2 * num_sms) // 3
    if num_heads == 2 * hpb:
        return 2 if ctas >= two_thirds or (chunks >= 8 and ctas >= third) else 1
    if num_heads == 4 * hpb:
        return 2 if chunks >= 8 and third <= ctas < two_thirds else 1
    return 2 if chunks >= 16 and ctas >= third else 1


def cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
    max_splits: int = 16,
    head_tiles: int = 1,
) -> Tuple[int, int]:
    """Return ``(num_splits, chunks_per_block)`` for one decode call.

    Measured split rules (RTX PRO 6000 Blackwell / RTX 5090; GB10 below), with
    the grid counted in CTAs of ``16 * head_tiles`` heads:

    * two chunks never split: one CTA pipelining both chunks beats two CTAs
      plus a merge;
    * while the unsplit grid covers at most 40 % of the SMs, take the largest
      power-of-two split that keeps the grid within 80 % of the SMs and leaves
      at least two chunks per CTA (one chunk per CTA only pays off for a lone
      CTA);
    * once the unsplit grid fills the SMs, split in two only for 16-chunk work
      whose doubled grid ends in a wave at least half full;
    * everything else runs unsplit;
    * independently, ``chunks_per_block <= max_chunks_per_block`` always holds
      (the CTA index table), so more than 1024 candidates force
      ``num_splits >= ceil(chunks / 16)``; ``max_splits`` below that raises.

    GB10 (SM121, 48 SMs; ``num_sms < 64``): the LPDDR gather saturates with a
    handful of CTAs, so "fill 80 % of the SMs" over-splits every small grid
    there.  The same loop runs with tighter caps, measured on a 108-row sweep:

    * the split grid stays within ``SMs / 3`` CTAs (16): one token with 128
      heads (8 CTAs) is best at two splits, with <= 64 heads at four;
    * the distinct gather streams ``tokens * splits`` stay within ``SMs / 6``
      (8): eight distinct tokens never gain from a split (+10-28 %), only the
      shared single-token gather does;
    * at least two chunks per CTA also for a lone CTA (one chunk per CTA is
      4-10 % slower than two);
    * no wave-quantization split on a full grid (16-chunk rows are flat).
    """

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    hpb = info["heads_per_block"]
    max_cpb = info["max_chunks_per_block"]
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if chunks < 1:
        raise ValueError("topk must be at least 1")
    min_splits = -(-chunks // max_cpb)
    if min_splits > max_splits:
        raise ValueError(
            f"{chunks} chunks need at least {min_splits} splits (at most {max_cpb} "
            f"chunks per block), max_splits={max_splits}"
        )
    cta_heads = hpb * int(head_tiles)
    head_blocks = (int(num_heads) + cta_heads - 1) // cta_heads
    base_ctas = int(num_tokens) * head_blocks
    splits = min_splits
    if chunks > 2 and max_splits > 1:
        num_sms = int(num_sms)
        small_die = num_sms < _SMALL_DIE_SMS
        grid_cap = num_sms // 3 if small_die else num_sms * 4 // 5
        # tokens * splits <= base_ctas * splits, so the stream cap is a no-op on GB202
        stream_cap = num_sms // 6 if small_die else grid_cap
        if base_ctas * 2 <= grid_cap:
            min_cpb = 2 if small_die or base_ctas > 1 else 1
            want = min_splits
            while (
                want * 2 <= max_splits
                and base_ctas * want * 2 <= grid_cap
                and int(num_tokens) * want * 2 <= stream_cap
                and -(-chunks // (want * 2)) >= min_cpb
            ):
                want *= 2
            splits = want
        elif not small_die and base_ctas >= num_sms and chunks >= 16:
            tail = (base_ctas * 2) % num_sms
            if tail == 0 or tail * 2 >= num_sms:
                splits = max(splits, 2)
    cpb = -(-chunks // splits)
    splits = -(-chunks // cpb)
    return splits, cpb


def _resolve_plan(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    device: torch.device,
    num_splits: Optional[int],
    max_splits: int,
    head_tiles: Optional[int],
) -> Tuple[int, int, int]:
    """Resolve ``(head_tiles, num_splits, chunks_per_block)`` like the kernel module's launcher."""

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    num_sms = _num_sms(device)
    ht = (
        int(head_tiles)
        if head_tiles is not None
        else cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=num_sms,
        )
    )
    if ht not in (1, 2) or (ht == 2 and num_heads not in info["two_tile_heads"]):
        raise ValueError(f"head_tiles={ht} is not valid for {num_heads} heads")
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if num_splits is None:
        splits, cpb = cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=num_sms,
            max_splits=max_splits,
            head_tiles=ht,
        )
        return ht, splits, cpb
    num_splits = int(num_splits)
    if num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    cpb = -(-chunks // num_splits)
    max_cpb = info["max_chunks_per_block"]
    if cpb > max_cpb:
        raise ValueError(
            f"num_splits={num_splits} leaves {cpb} chunks per CTA; the index table holds at most {max_cpb}"
        )
    return ht, -(-chunks // cpb), cpb


def cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0
) -> int:
    """Workspace bytes that cover every split plan of this shape (partials + LSE + alignment slack)."""

    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    rows = int(num_tokens) * int(num_heads)
    return rows * chunks * (_D_V * 2 + 4) + rows * 4 + 3 * 16


@functools.cache
def _num_sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


# Decode-vs-prefill crossover and prefill planner thresholds for ``backend="cake"`` on the
# sm_120a SKUs, measured by paired crossover sweeps of the generating kernel family (T 1..128,
# page 64; RTX PRO 6000 Blackwell Server Edition 188 SMs and RTX 5090 170 SMs).  The decode
# route's split planner is wave-quantised by the SM count, so the thresholds on which the two
# SKUs disagree are keyed by ``num_sms >= large_sms``.  Mirrors the kernel module's
# ``plan_prefill``; the parity test in ``tests/attention`` compares the two over the grid.
_PREFILL_POLICY = {
    # Head counts >= this follow the wide-head rules (four 16-head tiles per CTA available for H % 64 == 0).
    "wide_heads": 64,
    # Devices with at least this many SMs take the ``*_large_sms`` thresholds (RTX PRO 6000: 188), the others the
    # ``*_small_sms`` ones (RTX 5090: 170).  Smaller dies (GB10 / SM121, 48 SMs) take the RTX 5090 thresholds
    # unmeasured; the prefill family is compile-only there.
    "large_sms": 180,
    # Candidate counts (topk + extra_topk) >= this follow the ``*_many`` rules, smaller ones the ``*_few`` rules.
    "many_candidates": 512,
    # Wide heads: decode up to this many tokens (decode wins through T=8 at >= 512 candidates and through T=16 at
    # fewer on both SKUs; the two-tile prefill wins at 16 by 4-6 % and from 24 on).
    "wide_decode_tokens_many": 8,
    "wide_decode_tokens_few": 16,
    # Narrow heads (< wide_heads) with >= 512 candidates: decode up to this many tokens (PRO 6000: the one-tile
    # prefill wins from T=16 by 5-10 %; RTX 5090: decode wins through T=32 by 2-9 %).  With fewer candidates the
    # one-tile prefill wins at every token count on both SKUs.
    "narrow_decode_tokens_many_large_sms": 8,
    "narrow_decode_tokens_many_small_sms": 32,
    # Wide heads (H % 32 == 0): two head tiles up to this token count, the largest tile count above it, at every
    # candidate count and on both SKUs (at T=128 four tiles win on every wide A/B row except the PRO 6000 H128
    # K512 band below; H64 K512 four tiles 17 % faster; from T=512 four tiles win by 7-35 % everywhere).
    "wide_two_tile_tokens": 96,
    # Large die only, at least this many heads and a candidate count in ``two_tile_k512_band`` (half-open): two
    # tiles through this token count (PRO 6000: two tiles 1.6-4.0 % faster at T=128 in three paired rounds; above
    # k=512 the evidence is mixed within +-3.4 % so the four-tile default stays; the RTX 5090 prefers four tiles
    # by 12-16 % at every candidate count).
    "wide_two_tile_tokens_h128_k512_large_sms": 128,
    "two_tile_h128_heads": 128,
    "two_tile_k512_band": (512, 640),
}


def cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(
    num_heads: int,
) -> Tuple[int, ...]:
    """Head-tile counts (``16 * tiles`` heads per CTA) with a prefill instance for ``num_heads``.

    Empty when the head count has no prefill instance (8 heads).  1 always, 2 for
    head counts divisible by 32, 4 for head counts divisible by 64.
    """

    tiles = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()["prefill_head_tiles"]
    return tuple(tiles.get(int(num_heads), ()))


def cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
) -> int:
    """Return ``head_tiles`` for one prefill call (measured crossover, see ``_PREFILL_POLICY``).

    For ``num_heads >= 64`` with a two-tile instance (head count divisible by
    32): two tiles (32 heads per CTA) up to the two-tile token limit, the
    largest instance above it (4 tiles for head counts divisible by 64, 2 for
    96 heads; 80 / 112 heads only have the one-tile instance).  The limit is 96
    tokens at every candidate count on both SKUs, except 128 heads with 512 to
    639 candidates on the large die (RTX PRO 6000), where it is 128 tokens.
    Narrower head counts take their largest instance at every token count: two
    tiles for 32 heads, one tile for 16 and 48 heads.
    """

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    tiles = cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(num_heads)
    if not tiles:
        raise ValueError(
            f"Cake SM120 DSv4 NVFP4 sparse-MLA prefill supports {info['prefill_heads']} "
            f"query heads, got {num_heads}"
        )
    num_tokens = int(num_tokens)
    num_heads = int(num_heads)
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    policy = _PREFILL_POLICY
    if num_heads >= policy["wide_heads"]:
        two_tile_tokens = policy["wide_two_tile_tokens"]
        candidates = int(topk) + int(extra_topk)
        band_lo, band_hi = policy["two_tile_k512_band"]
        if (
            num_sms >= policy["large_sms"]
            and num_heads >= policy["two_tile_h128_heads"]
            and band_lo <= candidates < band_hi
        ):
            two_tile_tokens = policy["wide_two_tile_tokens_h128_k512_large_sms"]
        if num_tokens <= two_tile_tokens and 2 in tiles:
            head_tiles = 2
        else:
            head_tiles = max(tiles)
    else:
        head_tiles = max(tiles)
    return head_tiles


def cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
) -> str:
    """``"decode"`` or ``"prefill"`` for one ``backend="cake"`` call (measured crossover).

    Decode always serves head counts without a prefill instance (8 heads) and
    candidate lists beyond the prefill's 16-chunk table (more than 1024
    candidates); partial 64-wide chunks are served by the prefill's masked
    tail.  Otherwise, with ``candidates = topk + extra_topk``:

    * ``num_heads >= 64``: decode for ``num_tokens <= 8`` at 512 and more
      candidates and for ``num_tokens <= 16`` below; prefill above;
    * ``num_heads < 64``: at 512 and more candidates decode for
      ``num_tokens <= 8`` on devices with >= 180 SMs (RTX PRO 6000) and for
      ``num_tokens <= 32`` on smaller devices (RTX 5090); below 512 candidates
      prefill at every token count.

    ``num_sms`` keys the thresholds on which the two sm_120a SKUs disagree
    (the decode split planner's wave structure follows the SM count).
    """

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    num_tokens = int(num_tokens)
    num_heads = int(num_heads)
    topk = int(topk)
    extra_topk = int(extra_topk)
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if num_heads not in info["prefill_heads"] or chunks > info["prefill_max_chunks"]:
        return "decode"
    candidates = topk + extra_topk
    policy = _PREFILL_POLICY
    many = candidates >= policy["many_candidates"]
    large = num_sms >= policy["large_sms"]
    if num_heads >= policy["wide_heads"]:
        decode_tokens = policy[
            "wide_decode_tokens_many" if many else "wide_decode_tokens_few"
        ]
    elif many:
        decode_tokens = policy[
            "narrow_decode_tokens_many_large_sms"
            if large
            else "narrow_decode_tokens_many_small_sms"
        ]
    else:
        decode_tokens = 0
    return "decode" if num_tokens <= decode_tokens else "prefill"


def _cache_geometry(cache: torch.Tensor, name: str) -> Tuple[torch.Tensor, int, int]:
    """Flatten a paged NVFP4 cache view to ``(flat uint8 storage span, page_size, page_stride_bytes)``."""

    if not cache.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {cache.device}")
    if cache.dtype != torch.uint8:
        raise ValueError(f"{name} must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim not in (3, 4) or cache.shape[-1] != _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} must be [num_pages, page_size, {_BYTES_PER_TOKEN}], HND "
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
            f"{name} must have a singleton latent-head dimension at axis 1 or 2"
        )
    num_pages = int(cache.shape[0])
    page_size = int(cache.shape[page_dim])
    if num_pages < 1 or page_size < 1:
        raise ValueError(f"{name} must hold at least one page with at least one row")
    if cache.stride(-1) != 1 or cache.stride(page_dim) != _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} entries must be contiguous inside each page with strides "
            f"(..., {_BYTES_PER_TOKEN}, 1), got {cache.stride()}"
        )
    page_stride = int(cache.stride(0))
    if page_stride < page_size * _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} page stride {page_stride} is smaller than the logical "
            f"{page_size * _BYTES_PER_TOKEN}-byte page"
        )
    if page_stride % 16:
        raise ValueError(
            f"{name} page stride must be a multiple of 16 bytes, got {page_stride}"
        )
    if cache.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    span = (num_pages - 1) * page_stride + page_size * _BYTES_PER_TOKEN
    flat = cache.as_strided((span,), (1,), cache.storage_offset())
    return flat, page_size, page_stride


def _normalize_indices(
    indices: torch.Tensor, name: str, num_tokens: int
) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError(
            f"{name} must be a [num_tokens, topk] or [num_tokens, 1, topk] int32 tensor"
        )
    if indices.shape[0] != num_tokens:
        raise ValueError(f"{name} must have {num_tokens} rows, got {indices.shape[0]}")
    if indices.shape[1] < 1:
        raise ValueError(f"{name} must select at least one slot per token")
    return indices.contiguous()


def _normalize_length(
    length: Optional[torch.Tensor], name: str, num_tokens: int
) -> Optional[torch.Tensor]:
    if length is None:
        return None
    if length.ndim != 1 or length.dtype != torch.int32 or length.shape[0] < num_tokens:
        raise ValueError(
            f"{name} must be a 1-D int32 tensor with at least {num_tokens} entries"
        )
    return length.contiguous()


@functools.cache
def get_cake_sparse_mla_sm120_dsv4_nvfp4_module():
    module = gen_cake_sparse_mla_sm120_dsv4_nvfp4_module().build_and_load()
    entry = getattr(module, _manifest()["entry"])

    @register_custom_op(
        "flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_decode",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
    def _decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        output: torch.Tensor,
        out_lse: torch.Tensor,
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        page_size: int,
        page_stride_bytes: int,
        extra_page_size: int,
        extra_page_stride_bytes: int,
        num_splits: int,
        chunks_per_block: int,
        head_tiles: int,
        sm_scale: float,
        lse_scale: float,
    ) -> None:
        entry(
            q,
            kv_cache,
            indices,
            extra_kv_cache,
            extra_indices,
            topk_length,
            extra_topk_length,
            attn_sink,
            output,
            out_lse,
            mid_out,
            mid_lse,
            page_size,
            page_stride_bytes,
            extra_page_size,
            extra_page_stride_bytes,
            num_splits,
            chunks_per_block,
            head_tiles,
            sm_scale,
            lse_scale,
        )

    @register_fake_op("flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_decode")
    def _fake_decode(*_args, **_kwargs) -> None:
        return None

    prefill_entry = getattr(module, _manifest()["prefill_entry"])

    @register_custom_op(
        "flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_prefill",
        mutates_args=("output", "out_lse"),
    )
    def _prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        output: torch.Tensor,
        out_lse: torch.Tensor,
        page_size: int,
        page_stride_bytes: int,
        extra_page_size: int,
        extra_page_stride_bytes: int,
        head_tiles: int,
        sm_scale: float,
        lse_scale: float,
    ) -> None:
        prefill_entry(
            q,
            kv_cache,
            indices,
            extra_kv_cache,
            extra_indices,
            topk_length,
            extra_topk_length,
            attn_sink,
            output,
            out_lse,
            page_size,
            page_stride_bytes,
            extra_page_size,
            extra_page_stride_bytes,
            head_tiles,
            sm_scale,
            lse_scale,
        )

    @register_fake_op("flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_prefill")
    def _fake_prefill(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(
        decode=_decode, raw_decode=entry, prefill=_prefill, raw_prefill=prefill_entry
    )


def _prepare_inputs(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    extra_kv_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
) -> SimpleNamespace:
    """Validate the tensors the decode and prefill entries share and resolve the flat cache views."""

    if q.ndim != 3 or q.shape[-1] != _D_QK:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    heads = cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads()
    if num_heads not in heads:
        raise ValueError(
            f"Cake SM120 DSv4 NVFP4 sparse MLA supports {heads} query heads, got {num_heads}"
        )
    if (
        output.shape != q.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be a contiguous bfloat16 tensor of shape {tuple(q.shape)}"
        )
    if (
        out_lse.shape != (num_tokens, num_heads)
        or out_lse.dtype != torch.float32
        or not out_lse.is_contiguous()
    ):
        raise ValueError(
            f"out_lse must be a contiguous float32 tensor of shape {(num_tokens, num_heads)}"
        )
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    indices = _normalize_indices(indices, "indices", num_tokens)
    kv_flat, page_size, page_stride = _cache_geometry(kv_cache, "kv_cache")
    extra_flat = None
    extra_page_size = 0
    extra_page_stride = 0
    extra_topk = 0
    if extra_kv_cache is not None and extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
        extra_topk = int(extra_indices.shape[1])
        extra_flat, extra_page_size, extra_page_stride = _cache_geometry(
            extra_kv_cache, "extra_kv_cache"
        )
    else:
        extra_indices = None
    if attn_sink is not None:
        if (
            attn_sink.ndim != 1
            or attn_sink.dtype != torch.float32
            or attn_sink.shape[0] < num_heads
        ):
            raise ValueError(
                f"attn_sink must be a 1-D float32 tensor with at least {num_heads} entries"
            )
        attn_sink = attn_sink.contiguous()
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_heads=num_heads,
        indices=indices,
        topk=int(indices.shape[1]),
        kv_flat=kv_flat,
        page_size=page_size,
        page_stride=page_stride,
        extra_flat=extra_flat,
        extra_indices=extra_indices,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        extra_page_stride=extra_page_stride,
        topk_length=_normalize_length(topk_length, "topk_length", num_tokens),
        extra_topk_length=_normalize_length(
            extra_topk_length, "extra_topk_length", num_tokens
        ),
        attn_sink=attn_sink,
    )


@supported_compute_capability([120, 121])
def cake_sparse_mla_sm120_dsv4_nvfp4_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    max_splits: int = 16,
    head_tiles: Optional[int] = None,
) -> Dict[str, int]:
    """Run the allocation-free Cake SM120 NVFP4 sparse-MLA decode.

    Writes ``output`` and ``out_lse`` in place and returns the resolved plan
    ``{"head_tiles", "num_splits", "chunks_per_block"}``.  ``mid_out`` /
    ``mid_lse`` are required when the plan splits (``num_splits > 1``); size
    them with ``cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks`` splits to cover
    every plan.  ``head_tiles`` (1 or 2, head counts divisible by 32) and
    ``num_splits`` override the planners.
    """

    p = _prepare_inputs(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    num_tokens, num_heads = p.num_tokens, p.num_heads
    ht, splits, cpb = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=p.topk,
        extra_topk=p.extra_topk,
        device=q.device,
        num_splits=num_splits,
        max_splits=max_splits,
        head_tiles=head_tiles,
    )
    if splits > 1:
        if mid_out is None or mid_lse is None:
            raise ValueError(
                f"this shape splits into {splits} CTAs per head block; pass caller-owned mid_out / mid_lse scratch"
            )
        if (
            mid_out.ndim != 4
            or mid_out.shape[0] != num_tokens
            or mid_out.shape[1] != num_heads
            or mid_out.shape[2] < splits
            or mid_out.shape[3] != _D_V
            or mid_out.dtype != torch.bfloat16
            or not mid_out.is_contiguous()
        ):
            raise ValueError(
                f"mid_out must be a contiguous bfloat16 [{num_tokens}, {num_heads}, >= {splits}, {_D_V}] tensor, "
                f"got {tuple(mid_out.shape)} {mid_out.dtype}"
            )
        if (
            mid_lse.ndim != 3
            or tuple(mid_lse.shape) != tuple(mid_out.shape[:3])
            or mid_lse.dtype != torch.float32
            or not mid_lse.is_contiguous()
        ):
            raise ValueError(
                f"mid_lse must be a contiguous float32 {tuple(mid_out.shape[:3])} tensor, got {tuple(mid_lse.shape)}"
            )
    else:
        mid_out = None
        mid_lse = None
    get_cake_sparse_mla_sm120_dsv4_nvfp4_module().decode(
        q,
        p.kv_flat,
        p.indices,
        p.extra_flat,
        p.extra_indices,
        p.topk_length,
        p.extra_topk_length,
        p.attn_sink,
        output,
        out_lse,
        mid_out,
        mid_lse,
        p.page_size,
        p.page_stride,
        p.extra_page_size,
        p.extra_page_stride,
        splits,
        cpb,
        ht,
        float(sm_scale),
        float(lse_scale),
    )
    return {"head_tiles": ht, "num_splits": splits, "chunks_per_block": cpb}


@supported_compute_capability([120, 121])
def cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    head_tiles: Optional[int] = None,
) -> Dict[str, int]:
    """Run the allocation-free Cake SM120 NVFP4 sparse-MLA prefill (one launch, no scratch).

    Same tensor contract as :func:`cake_sparse_mla_sm120_dsv4_nvfp4_decode`
    (``q`` ``[T, H, 512]`` BF16, HND / NHD / 3-D packed caches with runtime
    page size and stride, ``[T, topk]`` int32 indices with ``-1`` masks,
    optional lengths / sink / second cache, caller-owned ``output`` and
    ``out_lse``), but one CTA runs every chunk of a token: the candidate list
    is limited to 16 chunks (``topk + extra_topk <= 1024`` slots) and the
    head count needs a prefill instance (16 .. 128, multiples of 16).  Writes
    ``output`` and ``out_lse`` in place and returns the resolved plan
    ``{"head_tiles", "num_ctas"}`` (one CTA per (token, head block) item);
    ``head_tiles`` (see :func:`cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles`)
    overrides :func:`cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill`.
    """

    p = _prepare_inputs(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    tiles = cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles(p.num_heads)
    if not tiles:
        raise ValueError(
            f"Cake SM120 DSv4 NVFP4 sparse-MLA prefill supports {info['prefill_heads']} "
            f"query heads, got {p.num_heads}"
        )
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(p.topk, p.extra_topk)
    if chunks > info["prefill_max_chunks"]:
        raise ValueError(
            f"the prefill kernel holds at most {info['prefill_max_chunks']} chunks of "
            f"{info['chunk_width']} candidates per token; topk {p.topk}"
            + (f" + extra_topk {p.extra_topk}" if p.extra_topk else "")
            + f" is {chunks} chunks (use the split decode)"
        )
    num_sms = _num_sms(q.device)
    plan_tiles = cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
        num_tokens=p.num_tokens,
        num_heads=p.num_heads,
        topk=p.topk,
        extra_topk=p.extra_topk,
        num_sms=num_sms,
    )
    ht = int(head_tiles) if head_tiles is not None else plan_tiles
    if ht not in tiles:
        raise ValueError(
            f"head_tiles={ht} is not valid for the {p.num_heads}-head prefill (choices {tiles})"
        )
    items = p.num_tokens * (p.num_heads // (info["heads_per_block"] * ht))
    get_cake_sparse_mla_sm120_dsv4_nvfp4_module().prefill(
        q,
        p.kv_flat,
        p.indices,
        p.extra_flat,
        p.extra_indices,
        p.topk_length,
        p.extra_topk_length,
        p.attn_sink,
        output,
        out_lse,
        p.page_size,
        p.page_stride,
        p.extra_page_size,
        p.extra_page_stride,
        ht,
        float(sm_scale),
        float(lse_scale),
    )
    return {"head_tiles": ht, "num_ctas": items}


@supported_compute_capability([120, 121])
def _cake_nvfp4_sparse_mla_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    head_tiles: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocating convenience entry (tests / benchmarks): returns ``(output, out_lse)``."""

    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        lse_scale=lse_scale,
        head_tiles=head_tiles,
    )
    return output, out_lse


@supported_compute_capability([120, 121])
def _cake_nvfp4_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    head_tiles: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocating convenience entry (tests / benchmarks): returns ``(output, out_lse)``."""

    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    topk = int(indices.shape[-1])
    extra_topk = int(extra_indices.shape[-1]) if extra_indices is not None else 0
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    mid_out = torch.empty(
        (num_tokens, num_heads, chunks, _D_V), dtype=torch.bfloat16, device=q.device
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, chunks), dtype=torch.float32, device=q.device
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=num_splits,
        head_tiles=head_tiles,
    )
    return output, out_lse


def functional_run(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    *,
    lengths: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    extra: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
) -> torch.Tensor:
    """Public-API route: pick decode or prefill, carve the split scratch (and the LSE when the caller passes none) from ``workspace``."""

    from ._prepared import _workspace_tensor_view

    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    kernel = cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_sms=_num_sms(q.device),
    )
    if kernel == "prefill":
        if lse is None:
            lse, _ = _workspace_tensor_view(
                workspace,
                byte_offset=0,
                shape=(num_tokens, num_heads),
                dtype=torch.float32,
                alignment=16,
            )
            if lse is None:
                raise ValueError(
                    "attention workspace insufficient for the Cake prefill LSE: need at least "
                    f"{num_tokens * num_heads * 4 + 16} bytes for {num_tokens} tokens x {num_heads} heads"
                )
        cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q,
            cache,
            indices,
            output,
            lse,
            scale,
            topk_length=lengths,
            attn_sink=sink,
            extra_kv_cache=extra,
            extra_indices=extra_indices,
            extra_topk_length=extra_lengths,
            lse_scale=lse_scale,
        )
        return lse
    ht, splits, _ = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        device=q.device,
        num_splits=None,
        max_splits=16,
        head_tiles=None,
    )
    requirements: list[tuple[tuple[int, ...], torch.dtype]] = []
    if splits > 1:
        requirements.append(((num_tokens, num_heads, splits, _D_V), torch.bfloat16))
        requirements.append(((num_tokens, num_heads, splits), torch.float32))
    if lse is None:
        requirements.append(((num_tokens, num_heads), torch.float32))
    views = []
    offset = 0
    for shape, dtype in requirements:
        view, offset = _workspace_tensor_view(
            workspace, byte_offset=offset, shape=shape, dtype=dtype, alignment=16
        )
        if view is None:
            raise ValueError(
                "attention workspace insufficient for the resolved Cake plan: need at least "
                f"{cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(num_tokens, num_heads, topk, extra_topk)} "
                f"bytes for {num_tokens} tokens x {num_heads} heads x topk {topk}"
                + (f" + extra topk {extra_topk}" if extra_topk else "")
            )
        views.append(view)
    mid_out = views[0] if splits > 1 else None
    mid_lse = views[1] if splits > 1 else None
    result = lse if lse is not None else views[-1]
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        cache,
        indices,
        output,
        result,
        scale,
        topk_length=lengths,
        attn_sink=sink,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
        head_tiles=ht,
    )
    return result


def _arena(
    wrapper, name: str, numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    arenas: Dict[str, torch.Tensor] = wrapper.__dict__.setdefault("_cake_arenas", {})
    current = arenas.get(name)
    if current is None or current.numel() < numel or current.device != device:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("warm up this attention shape before CUDA graph capture")
        current = torch.empty(max(numel, 1), dtype=dtype, device=device)
        arenas[name] = current
    return current


def wrapper_run(
    wrapper,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    prefill_impl: Optional[str] = None,
    return_lse: bool = False,
    lse_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """``SparseMLASm120Wrapper.run`` for ``backend="cake"``: decode / prefill crossover, wrapper-owned grow-only scratch.

    ``prefill_impl`` keeps the SM120 NVFP4 contract (``None`` / ``"auto"`` /
    ``"mg"`` accepted); the Cake route always selects its own kernel through
    :func:`cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel`.
    """

    if q.ndim == 4 and q.shape[1] == 1:
        q = q.squeeze(1)
    if output.ndim == 4 and output.shape[1] == 1:
        output = output.squeeze(1)
    if q.ndim != 3:
        raise ValueError("q must be [T,H,D] or [T,1,H,D]")
    if q.device != wrapper._device:
        raise ValueError("tensors must be on the Wrapper device")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    if (
        wrapper._max_num_tokens is not None and num_tokens > wrapper._max_num_tokens
    ) or (wrapper._max_num_heads is not None and num_heads > wrapper._max_num_heads):
        raise ValueError("query exceeds max_num_tokens/max_num_heads")
    if prefill_impl not in (None, "auto", "mg"):
        raise ValueError("NVFP4 prefill_impl must be None, auto, or mg")
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    if (mid_out is None) != (mid_lse is None):
        raise ValueError("mid_out and mid_lse must be provided together")
    if out_lse is None:
        lse = _arena(wrapper, "lse", num_tokens * num_heads, torch.float32, q.device)[
            : num_tokens * num_heads
        ].view(num_tokens, num_heads)
    else:
        lse = out_lse[:num_tokens, :num_heads]
        if not lse.is_contiguous():
            raise ValueError(
                "out_lse must be a contiguous [num_tokens, num_heads] float32 buffer"
            )
    kernel = cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_sms=_num_sms(q.device),
    )
    if kernel == "prefill":
        cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
            q,
            kv_cache,
            indices,
            output,
            lse,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            lse_scale=lse_scale,
        )
        return lse if return_lse else None
    ht, splits, _ = _resolve_plan(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        device=q.device,
        num_splits=None,
        max_splits=16,
        head_tiles=None,
    )
    if splits > 1 and mid_out is None:
        rows = num_tokens * num_heads * splits
        mid_out = _arena(wrapper, "mid_out", rows * _D_V, torch.bfloat16, q.device)[
            : rows * _D_V
        ].view(num_tokens, num_heads, splits, _D_V)
        mid_lse = _arena(wrapper, "mid_lse", rows, torch.float32, q.device)[:rows].view(
            num_tokens, num_heads, splits
        )
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        kv_cache,
        indices,
        output,
        lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
        head_tiles=ht,
    )
    return lse if return_lse else None


__all__ = [
    "cake_sparse_mla_sm120_dsv4_nvfp4_decode",
    "cake_sparse_mla_sm120_dsv4_nvfp4_format_info",
    "cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks",
    "cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles",
    "cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill",
    "cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits",
    "cake_sparse_mla_sm120_dsv4_nvfp4_prefill",
    "cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles",
    "cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes",
    "cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel",
    "cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads",
    "get_cake_sparse_mla_sm120_dsv4_nvfp4_module",
]
