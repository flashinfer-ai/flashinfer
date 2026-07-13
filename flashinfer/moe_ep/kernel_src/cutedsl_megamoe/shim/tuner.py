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
    ``cluster_shape_mnk``);
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
        knobs.update(zip(names, values))
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
    "is_valid",
    "iter_candidates",
    "with_knobs",
]
