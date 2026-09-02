# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel tuning knobs for the SM90 Hopper FP8 MegaMoE frontend.

Sibling-fork mirror of ``kernel_src/cutedsl_megamoe/shim/tuner.py``,
retargeted at ``Sm90MegaMoE(SwapAB)Fp8Kernel``.  Exposes the SM90 tuning
surface so callers -- or an autotuner -- can enumerate valid tactics and
apply them to a :class:`.hopper_fp8.MegaMoEHopperFp8Config`.

Two knob classes (same taxonomy as the SM100 tree):

  * **correctness knobs** change a code path / output, so an autotuner must
    keep the value it validated against (``in_kernel_fc2_reduce``,
    ``token_back_mode``, ``load_balance_mode``).  ``in_kernel_fc2_reduce``
    additionally makes the output accumulation order nondeterministic.
  * **perf knobs** do not change the output and are free to sweep for speed.
    Unlike SM100 (where tile/cluster are correctness knobs), the SM90 fork's
    ``swap_ab`` / ``pingpong`` / ``mma_tiler_mnk`` / ``cluster_shape_mnk`` /
    ``fp8_accum_mode`` select numerically equivalent execution geometries,
    so they live in the perf class together with ``group_hint`` /
    ``flag_batch`` / ``epi_flag_batch``.

The built-in heuristic (:func:`default_knobs`) wraps the kernel drop's
token-bucket table (``moe_hopper_fp8/heuristic_config.py``), so
``knobs=None`` without a cache entry reproduces the shim's established
default launch configs exactly.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Dict, Iterator, Optional, Tuple

# --- knob value-sets (geometry domains mirror the kernel ctor validation;
# see kernel_fp8_glu_fc12{,_swapab}.py _validate_mma_tiler_and_cluster_shape).

CORRECTNESS_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "in_kernel_fc2_reduce": (False, True),
    "token_back_mode": ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
    "load_balance_mode": ("static", "atomic_counter"),
}

_NONSWAP_TILES = ((64, 128, 128), (64, 256, 128))
_SWAPAB_TILES = tuple((m, n, 128) for m in (128, 256) for n in (16, 32, 64, 128))
_CLUSTER_SHAPES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))

PERF_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "swap_ab": (False, True),
    "pingpong": (False, True),
    "mma_tiler_mnk": _NONSWAP_TILES + _SWAPAB_TILES,
    "cluster_shape_mnk": _CLUSTER_SHAPES,
    "fp8_accum_mode": ("1xacc", "2xacc"),
    # ``group_hint=None`` means "use max_active_clusters" (occupancy hint).
    "group_hint": (None, 64, 128, 256, 512),
    "flag_batch": (1, 2, 4, 8),
    "epi_flag_batch": ((1, 1), (2, 2), (2, 4), (4, 4), (4, 8)),
}

# Geometry knob names resolved as one unit by the heuristic table / cache.
GEOMETRY_KNOBS = (
    "swap_ab",
    "pingpong",
    "mma_tiler_mnk",
    "cluster_shape_mnk",
    "fp8_accum_mode",
)


def default_knobs(
    num_tokens: int, *, fp8_scale_mode: str = "per_tensor"
) -> Dict[str, Any]:
    """Default geometry knobs for a compile-time token count (buffer size).

    Wraps the kernel drop's token-bucket heuristic table
    (``heuristic_config.select_heuristic_config``), keyed on
    ``fp8_scale_mode`` and the buffer capacity, so the ``knobs=None``
    fallback matches the shim's established default launch configs.
    Perf knobs the table does not cover (``group_hint`` / ``flag_batch`` /
    ``epi_flag_batch``) are left unset -- the config defaults apply.

    Returns a fresh dict each call.
    """
    from moe_hopper_fp8.heuristic_config import select_heuristic_config

    sel = select_heuristic_config(fp8_scale_mode, max(int(num_tokens), 1))
    c = sel.config
    return {
        "swap_ab": c.swap_ab,
        "pingpong": c.pingpong,
        "mma_tiler_mnk": tuple(c.mma_tiler_mnk),
        "cluster_shape_mnk": tuple(c.cluster_shape_mnk),
        "fp8_accum_mode": c.accum_mode,
        "token_back_mode": c.token_back_mode,
    }


def is_valid(knobs: Dict[str, Any], *, apply_topk_in_fc1: bool = True) -> bool:
    """``True`` if ``knobs`` is a compilable SM90 FP8 MegaMoE combo.

    Mirrors the kernel ctor / config ``__post_init__`` rules; unspecified
    knobs fall back to the kernel defaults, so a partial dict is fine.
    """
    swap_ab = bool(knobs.get("swap_ab", False))
    pingpong = bool(knobs.get("pingpong", False))
    m, n, k = knobs.get("mma_tiler_mnk", (64, 128, 128))
    cm, cn, ck = knobs.get("cluster_shape_mnk", (1, 1, 1))
    accum = knobs.get("fp8_accum_mode", "1xacc")
    in_kernel = knobs.get("in_kernel_fc2_reduce", False)

    if swap_ab:
        if m not in (128, 256) or n not in (16, 32, 64, 128):
            return False
        if pingpong and m != 128:
            return False
    else:
        if m != 64 or n not in (128, 256):
            return False
        if pingpong and n != 128:
            return False
    if k % 128 != 0:
        return False
    if ck != 1 or (cm, cn) not in ((1, 1), (2, 1), (1, 2), (2, 2)):
        return False
    if accum not in ("1xacc", "2xacc"):
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


def with_knobs(config: Any, knobs: Optional[Dict[str, Any]]) -> Any:
    """Return a copy of ``config`` with ``knobs`` applied.

    Only knobs the config declares are set (unknown keys are silently
    dropped, mirroring the SM100 tree).  Passing ``knobs=None`` returns the
    config unchanged.  The returned config re-runs ``__post_init__``
    validation, so an invalid combination raises immediately.
    """
    if not knobs:
        return config
    fields = {f.name for f in dataclasses.fields(config)}
    overrides = {k: v for k, v in knobs.items() if k in fields}
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)


__all__ = [
    "CORRECTNESS_KNOBS",
    "GEOMETRY_KNOBS",
    "PERF_KNOBS",
    "default_knobs",
    "is_valid",
    "iter_candidates",
    "with_knobs",
]
