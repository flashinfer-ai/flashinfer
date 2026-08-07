# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel tuning knobs for the CuTeDSL MegaMoE frontends.

Exposes the ``Sm100MegaMoE{,Mxfp8}Kernel`` tuning surface (tile / cluster /
schedule knobs) so callers -- or an autotuner -- can enumerate valid tactics and
apply them to a frontend config.  The knob taxonomy and the validity predicate
mirror the kernel team's ``tester/solvers/inference_solver.py`` (kept in sync on
each ``src/`` drop; see that file's ``filter_invalid`` for the ctor/compile-time
rules each check stands in for).

Two knob classes:
  * **correctness knobs** change a code path / output, so an autotuner must keep
    the value it validated against (``in_kernel_fc2_reduce``, ``token_back_mode``,
    ``non_ubulk_fc2_store``, ``load_balance_mode``, ``mma_tiler_mnk``,
    ``cluster_shape_mnk``).  ``in_kernel_fc2_reduce`` additionally makes the
    output accumulation order nondeterministic (validate with a tolerance);
    :mod:`.autotune` sweeps it for NVFP4 because the sym-heap output serves
    both modes;
  * **perf knobs** do not change the output and are free to sweep for speed
    (``group_hint``, ``flag_batch``, ``epi_flag_batch``).

The shim configs differ slightly per dtype (NVFP4 uses ``token_back_mode``;
MXFP8 uses the ``token_back_by_dispatch`` bool and has no ``non_ubulk_fc2_store``),
so :func:`with_knobs` applies only the knobs a given config actually declares and
translates ``token_back_mode`` -> ``token_back_by_dispatch`` where needed.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Dict, Iterator, Optional, Tuple

# --- knob value-sets (mirror inference_solver._correctness_knobs / _perf_knobs) ---

CORRECTNESS_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "in_kernel_fc2_reduce": (False, True),
    "token_back_mode": ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
    "non_ubulk_fc2_store": (True, False),
    "load_balance_mode": ("static", "atomic_counter"),
    "mma_tiler_mnk": (
        (128, 128, 256),
        (128, 256, 256),
        (256, 128, 256),
        (256, 256, 256),
        (256, 64, 256),
    ),
    "cluster_shape_mnk": ((1, 1, 1), (2, 1, 1), (4, 1, 1)),
}

# ``group_hint=None`` means "use max_active_clusters" (the occupancy hint).
PERF_KNOBS: Dict[str, Tuple[Any, ...]] = {
    "group_hint": (None, 64, 128, 256, 512),
    "flag_batch": (1, 2, 4, 8, 16),
    "epi_flag_batch": (
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 2),
        (4, 4),
        (8, 2),
        (8, 4),
        (2, 8),
        (4, 8),
        (16, 16),
    ),
}


# --- token-count heuristic: default perf/tile tactic per compile-time size ----
# Keyed on the buffer's ``num_max_tokens`` (the kernel compiles once for that
# size, so the tile/cluster/schedule are fixed at compile).  Sets only the
# perf/tile knobs; ``in_kernel_fc2_reduce`` and ``combine_format`` stay owned by
# the config / caller.
#
# NVFP4 profiles from measured winners (online autotune, 4x GB200, 256 experts,
# top-8, hidden 7168, inter 2048, 2026-07-14).  The dominant axis is
# ``token_back_mode``: ``epi_warps`` wins at small batch but falls off a cliff
# mid-range (+18% at 512 tokens, +35% at 1024 -- every dispatch-warp candidate
# beat every epi_warps candidate there); tile/flag_batch are second-order.
_SMALL_TOKEN_KNOBS: Dict[str, Any] = {  # < 512 tokens (winner at 8)
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "epi_warps",
    "load_balance_mode": "atomic_counter",
}

_MID_TOKEN_KNOBS: Dict[str, Any] = {  # 512..1023 (winner at 512)
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "reuse_dispatch_warps",
    "load_balance_mode": "atomic_counter",
}

_MID_LARGE_TOKEN_KNOBS: Dict[str, Any] = {  # 1024..2047 (winner at 1024)
    "mma_tiler_mnk": (256, 256, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "standalone_warps",
    "load_balance_mode": "atomic_counter",
}

_LARGE_TOKEN_KNOBS: Dict[str, Any] = {  # >= 2048 (validated at 2048)
    "mma_tiler_mnk": (256, 256, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 8,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "reuse_dispatch_warps",
    "load_balance_mode": "atomic_counter",
}

# MXFP8 small-batch profile.  Re-derived 2026-07-15 on 4x GB200 (256 experts,
# top-8, hidden 7168, inter 2048) via the online autotuner on the CORRECTED
# K-major nvfp4 weight layout era (the mxfp8 path itself was unaffected by
# the layout fix; the profiles were simply re-measured): flag_batch=4 +
# epi_warps confirmed at 8 tokens (384.0 vs 384.4 us — tie with fb8).
# The MXFP8 kernel's mma tile is fixed at (256, 256), so no tile knob here.
_MXFP8_TOKEN_KNOBS: Dict[str, Any] = {
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "epi_warps",
    "load_balance_mode": "atomic_counter",
}

# MXFP8 large-batch profile (>= 2048 tokens).  Re-derived 2026-07-15 (same
# geometry, online autotuner, kernel-mode p50): fb4 + reuse_dispatch_warps
# wins 1010.6 vs 1181.6 us for the old epi_warps profile at 2048 tokens
# (-14.5%), superseding the 2026-07-14 "dispatch-warp is ~5% slower for
# MXFP8" reading, which was taken alongside fb8 in the same candidate (the
# token-back mode, not the flag batch, is the dominant axis here — mirrors
# the NVFP4 mid/large profiles).  Deterministic (no ikr in the mxfp8 space).
_MXFP8_LARGE_TOKEN_KNOBS: Dict[str, Any] = {
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "flag_batch": 4,
    "epi_flag_batch": (2, 4),
    "token_back_mode": "reuse_dispatch_warps",
    "load_balance_mode": "atomic_counter",
}


def default_knobs(num_tokens: int, *, dtype: str = "nvfp4") -> Dict[str, Any]:
    """Default perf/tile knobs for a compile-time token count (buffer size).

    NVFP4: four measured profiles keyed on token count -- <512 small-batch
    latency (128-wide N tile, ``epi_warps``), 512..1023 mid
    (``reuse_dispatch_warps``), 1024..2047 mid-large (256-wide N tile,
    ``standalone_warps``), >=2048 large throughput (``flag_batch=8``,
    ``reuse_dispatch_warps``).  See the profile dicts above for measurement
    provenance.

    ``dtype="mxfp8"`` -> two measured profiles: <2048 tokens
    ``_MXFP8_TOKEN_KNOBS`` (epi_warps), >=2048 ``_MXFP8_LARGE_TOKEN_KNOBS``
    (reuse_dispatch_warps, -14.5% at 2048 — re-derived 2026-07-15).  No
    ``mma_tiler_mnk``: the MXFP8 kernel hard-requires
    ``mma_tiler (M, N) = (256, 256)``.

    NVFP4 profiles were re-validated 2026-07-15 on the corrected K-major
    weight layout: the online autotuner confirmed all four non-ikr defaults
    within run noise (64/512/1024 tokens: <=1.2%); the only wins were
    ikr candidates at >=2048 (-4..5%), which stay OPT-IN because ikr makes
    the output accumulation order nondeterministic (autotune keeps it when
    it measures fastest; pin ``in_kernel_fc2_reduce=False`` for
    bit-reproducibility).

    Returns a fresh dict each call.
    """
    if dtype == "mxfp8":
        if num_tokens >= 2048:
            return dict(_MXFP8_LARGE_TOKEN_KNOBS)
        return dict(_MXFP8_TOKEN_KNOBS)
    if num_tokens < 512:
        return dict(_SMALL_TOKEN_KNOBS)
    if num_tokens < 1024:
        return dict(_MID_TOKEN_KNOBS)
    if num_tokens < 2048:
        return dict(_MID_LARGE_TOKEN_KNOBS)
    return dict(_LARGE_TOKEN_KNOBS)


def is_valid(knobs: Dict[str, Any], *, combine_format: str = "bf16") -> bool:
    """``True`` if ``knobs`` is a compilable ``Sm100MegaMoEKernel`` combo.

    Mirrors ``inference_solver.filter_invalid`` (negated).  ``combine_format`` is
    the cross-rank combine wire format (``"bf16"`` for the default moe_ep path,
    or ``"16e2m1xbf16"`` / ``"32e4m3xe8m0"`` when quantized).  Unspecified knobs
    fall back to the kernel defaults, so a partial dict is fine.
    """
    combine_quantized = combine_format != "bf16"
    in_kernel = knobs.get("in_kernel_fc2_reduce", False)
    token_back_mode = knobs.get("token_back_mode", "epi_warps")
    non_ubulk = knobs.get("non_ubulk_fc2_store", True)
    mma_tiler = knobs.get("mma_tiler_mnk", (128, 128, 256))
    cluster = knobs.get("cluster_shape_mnk", (1, 1, 1))

    # quantized combine uses the explicit topk-reduce path; no in-kernel REDG.
    if combine_quantized and in_kernel:
        return False
    # quantized combine wires are only wired for dispatch-warp token-back
    # (mirrors the NVFP4 config validation).
    if combine_quantized and token_back_mode != "reuse_dispatch_warps":
        return False
    # FP4 combine data cannot use the UBLK fc2 store (sub-byte scalar deref).
    if combine_format == "16e2m1xbf16" and not non_ubulk:
        return False
    # dispatch-warp token-back requires STG (non-UBLK) fc2 store.
    if token_back_mode != "epi_warps" and not non_ubulk:
        return False
    # a 2-CTA MMA tile (M == 256) needs an even cluster M.
    if mma_tiler[0] == 256 and cluster[0] % 2 != 0:
        return False
    return True


def iter_candidates(
    *,
    include_correctness: bool = False,
    combine_format: str = "bf16",
    base: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield valid knob dicts (cross-product), each merged onto ``base``.

    ``include_correctness=False`` (default) sweeps only the perf knobs (output
    invariant); set ``True`` to also enumerate the correctness knobs.  Illegal
    combos (per :func:`is_valid`) are skipped.
    """
    space = dict(PERF_KNOBS)
    if include_correctness:
        space = {**CORRECTNESS_KNOBS, **space}
    names = list(space)
    for values in itertools.product(*(space[n] for n in names)):
        knobs = dict(base or {})
        knobs.update(zip(names, values, strict=False))
        if is_valid(knobs, combine_format=combine_format):
            yield knobs


def with_knobs(config: Any, knobs: Optional[Dict[str, Any]]) -> Any:
    """Return a copy of ``config`` with ``knobs`` applied (dtype-aware).

    Only knobs the config declares are set; ``token_back_mode`` is translated to
    the MXFP8 config's ``token_back_by_dispatch`` bool.  ``config`` is any of the
    frontend config dataclasses (NVFP4 / MXFP8).  Passing ``knobs=None`` returns
    the config unchanged.
    """
    if not knobs:
        return config
    fields = {f.name for f in dataclasses.fields(config)}
    overrides: Dict[str, Any] = {}
    for key, value in knobs.items():
        if key == "token_back_mode" and "token_back_by_dispatch" in fields:
            overrides["token_back_by_dispatch"] = value != "epi_warps"
        elif key in fields:
            overrides[key] = value
        # silently drop knobs this dtype's config does not expose
    # a 256-M tile implies 2-CTA instrs; keep the derived flag consistent.
    if "mma_tiler_mnk" in overrides and "use_2cta_instrs" in fields:
        overrides.setdefault("use_2cta_instrs", overrides["mma_tiler_mnk"][0] == 256)
    return dataclasses.replace(config, **overrides)


__all__ = [
    "CORRECTNESS_KNOBS",
    "PERF_KNOBS",
    "default_knobs",
    "is_valid",
    "iter_candidates",
    "with_knobs",
]
