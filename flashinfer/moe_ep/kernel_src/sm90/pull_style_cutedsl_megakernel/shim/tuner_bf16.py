# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel tuning knobs for the SM90 Hopper BF16 MegaMoE frontend.

Twin of :mod:`.tuner` (the FP8 knob surface) for
``Sm90MegaMoE(SwapAB)Bf16Kernel``: the same two knob classes (correctness vs
perf), the same ``with_knobs`` application (imported from :mod:`.tuner`),
and a knob-cache binding under the BF16 key.  Deltas vs FP8:

* no ``fp8_accum_mode`` axis and no scale mode -- the heuristic table
  (``moe_hopper_bf16/heuristic_config.py``) is keyed on the token count only;
* tile K is 64 (two-byte operands); the kernel accepts any multiple of 64;
* ``group_hint`` domain adds the table's 264 (~5 experts per scheduler
  group) and ``ALL_EXPERTS_GROUP`` (one group) values.

The built-in heuristic (:func:`default_knobs`) wraps the kernel drop's
token-bucket table, so ``knobs=None`` without a cache entry reproduces the
drop driver's launch configs exactly.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, Optional, Tuple

from .tuner import with_knobs

# group_hint that puts every expert into one scheduler group
# (heuristic_config.ALL_EXPERTS_GROUP; mirrored so this module imports
# without the kernel packages on sys.path).
_ALL_EXPERTS_GROUP = 1 << 20

CORRECTNESS_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "in_kernel_fc2_reduce": (False, True),
    "token_back_mode": ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
    "load_balance_mode": ("static", "atomic_counter"),
}

_TILE_K = 64
_NONSWAP_TILES = ((64, 128, _TILE_K), (64, 256, _TILE_K))
# Keep in sync with hopper_bf16._SWAPAB_TILE_N_CHOICES.
_SWAPAB_TILES = tuple((m, n, _TILE_K) for m in (128, 256) for n in (8, 16, 32, 64, 128))
_CLUSTER_SHAPES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))

PERF_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "swap_ab": (False, True),
    "pingpong": (False, True),
    # How many of the 4 dispatch warps do token-comm work (the rest idle);
    # output-invariant partitioning (rank-local, no wire-format coupling).
    "active_dispatch_warps": (1, 2, 4),
    # FC1 store offload / early fc1_done publication / producer fold:
    # output-invariant (publication timing and warp layout only).
    "fc1_store_offload": (False, True),
    "fc1_early_done_publish": (False, True),
    "fold_producer_warps": (False, True),
    "mma_tiler_mnk": _NONSWAP_TILES + _SWAPAB_TILES,
    "cluster_shape_mnk": _CLUSTER_SHAPES,
    # ``group_hint=None`` means "use max_active_clusters" (occupancy hint).
    "group_hint": (None, 64, 128, 256, 264, 512, _ALL_EXPERTS_GROUP),
    # Tail-split pair tasks; legal only with swap-AB cga (1, 2, 1) or
    # non-swap cga (2, 1, 1).  Output-invariant.
    "tail_split_pairs": (False, True),
    "flag_batch": (1, 2, 4, 8),
    "epi_flag_batch": ((1, 1), (2, 2), (2, 4), (4, 4), (4, 8)),
}

# Geometry knob names resolved as one unit by the heuristic table / cache.
GEOMETRY_KNOBS = (
    "swap_ab",
    "pingpong",
    "mma_tiler_mnk",
    "cluster_shape_mnk",
)


def default_knobs(num_tokens: int) -> Dict[str, Any]:
    """Default geometry knobs for a compile-time token count (buffer size).

    Wraps the kernel drop's token-bucket heuristic table
    (``heuristic_config.select_heuristic_config``), keyed on the buffer
    capacity, so the ``knobs=None`` fallback matches the drop driver's
    launch configs.  The table also carries the per-bucket
    ``token_back_mode``, ``group_hint`` and ``tail_split_pairs``; perf knobs
    it does not cover (``flag_batch`` / ``epi_flag_batch``) are left unset --
    the config defaults apply.

    Returns a fresh dict each call.
    """
    from moe_hopper_bf16.heuristic_config import select_heuristic_config

    sel = select_heuristic_config(max(int(num_tokens), 1))
    c = sel.config
    return {
        "swap_ab": c.swap_ab,
        "pingpong": c.pingpong,
        "mma_tiler_mnk": tuple(c.mma_tiler_mnk),
        "cluster_shape_mnk": tuple(c.cluster_shape_mnk),
        "token_back_mode": c.token_back_mode,
        "group_hint": c.group_hint,
        "tail_split_pairs": c.tail_split_pairs,
    }


def is_valid(knobs: Dict[str, Any], *, apply_topk_in_fc1: bool = True) -> bool:
    """``True`` if ``knobs`` is a compilable SM90 BF16 MegaMoE combo.

    Mirrors the kernel ctor / config ``__post_init__`` rules; unspecified
    knobs fall back to the kernel defaults, so a partial dict is fine.
    """
    swap_ab = bool(knobs.get("swap_ab", False))
    pingpong = bool(knobs.get("pingpong", False))
    m, n, k = knobs.get("mma_tiler_mnk", (64, 128, _TILE_K))
    cm, cn, ck = knobs.get("cluster_shape_mnk", (1, 1, 1))
    in_kernel = knobs.get("in_kernel_fc2_reduce", False)

    if swap_ab:
        if m not in (128, 256) or n not in (8, 16, 32, 64, 128):
            return False
        if pingpong and m != 128:
            return False
    else:
        if m != 64 or n not in (128, 256):
            return False
        if pingpong and n != 128:
            return False
    if k <= 0 or k % _TILE_K != 0:
        return False
    if ck != 1 or (cm, cn) not in ((1, 1), (2, 1), (1, 2), (2, 2)):
        return False
    if knobs.get("tail_split_pairs", False):
        # Exactly two CTAs along the tokens and one along the weights.
        token_cluster, weight_cluster = (cn, cm) if swap_ab else (cm, cn)
        if token_cluster != 2 or weight_cluster != 1:
            return False
    if knobs.get("token_back_mode") not in (
        None,
        "epi_warps",
        "standalone_warps",
        "reuse_dispatch_warps",
    ):
        return False
    # Kernel invariant: the in-kernel reduce collapses topk before a separate
    # reducer could apply routing weights.
    if in_kernel and not apply_topk_in_fc1:
        return False
    if knobs.get("active_dispatch_warps", 1) not in (1, 2, 4):
        return False
    if knobs.get("load_balance_mode", "static") not in ("static", "atomic_counter"):
        return False
    return True


def iter_candidates(
    *,
    include_correctness: bool = False,
    base: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield valid knob dicts (cross-product), each merged onto ``base``.

    ``include_correctness=False`` (default) sweeps only the perf knobs
    (output invariant); set ``True`` to also enumerate the correctness
    knobs.  Illegal combos (per :func:`is_valid`) are skipped.
    """
    space = dict(PERF_KNOBS)
    if include_correctness:
        space = {**CORRECTNESS_KNOBS, **space}
    names = list(space)
    for values in itertools.product(*(space[n] for n in names)):
        knobs = dict(base or {})
        knobs.update(zip(names, values, strict=False))
        if is_valid(knobs):
            yield knobs


# --- knob cache binding ------------------------------------------------------
#
# The JSON cache file is shared with the FP8 tree (same env / path / version);
# this frontend keys its entries with dtype "bf16" and the scale-mode slot
# "none".  FP8 entries carry a real scale mode and SM100 entries have no
# scale-mode field at all, so nothing cross-matches.


def lookup_knobs(
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    device: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Cached knob dict for this BF16 session key, or ``None`` on miss."""
    from .hopper_bf16 import BF16_KNOB_CACHE_DTYPE, BF16_KNOB_CACHE_SCALE_MODE
    from .knob_cache import lookup_knobs as _lookup

    return _lookup(
        dtype=BF16_KNOB_CACHE_DTYPE,
        fp8_scale_mode=BF16_KNOB_CACHE_SCALE_MODE,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        device=device,
    )


def record_knobs(
    knobs: Dict[str, Any],
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    device: Optional[str] = None,
    p50_us: Optional[float] = None,
    source: str = "autotune",
) -> Optional[str]:
    """Upsert one tuned BF16 entry (see :func:`.knob_cache.record_knobs`)."""
    from .hopper_bf16 import BF16_KNOB_CACHE_DTYPE, BF16_KNOB_CACHE_SCALE_MODE
    from .knob_cache import record_knobs as _record

    return _record(
        knobs,
        dtype=BF16_KNOB_CACHE_DTYPE,
        fp8_scale_mode=BF16_KNOB_CACHE_SCALE_MODE,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        device=device,
        p50_us=p50_us,
        source=source,
    )


def resolve_knobs(
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
) -> Tuple[Dict[str, Any], str]:
    """Pure-lookup knob resolution: cache hit, else built-in heuristic.

    Returns ``(knobs, source)`` where source is ``"cache"`` or
    ``"heuristic"``.
    """
    cached = lookup_knobs(
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
    )
    if cached is not None:
        return cached, "cache"
    return default_knobs(max_tokens), "heuristic"


__all__ = [
    "CORRECTNESS_KNOBS",
    "GEOMETRY_KNOBS",
    "PERF_KNOBS",
    "default_knobs",
    "is_valid",
    "iter_candidates",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    "with_knobs",
]
