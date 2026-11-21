# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dispatcher of ``support.row.v1``: the row class's configuration space (``schedule.all_valid_configs``) at one row
per CTA group, a closed-form pick with constants fitted on the regret sweeps of both parts, no table. Outside every
kernel cache closure: an edit here
recompiles nothing.

The support kernel stages the CTA's share of the row in shared memory once (through the loader's predicated register
loads: ``_DrawLoader.stage_from_regs``) and reads it from there in every pass -- the trees, the four radix sweeps, the
masked sum and the row outputs -- so a row longer than the staging budget takes a cluster of CTAs, each with its share of
the row (the histograms are reduced across the cluster per pass). The sweeps are instruction-bound and want warps; the
scans are per warp.
"""

from __future__ import annotations

import functools

from .schedule import (
    BLOCK,
    LANES,
    MAX_CLUSTER,
    SMEM_BUDGET,
    LSEConfig,
    max_vec_elems,
    padded_blocks,
)

#: The regimes, by how many rows share the device. Few rows (``M * SMALL_ROWS_MULT <= sm_count``): the latency regime --
#: a row spread over many CTAs with a small share each. Rows filling the device about once (``M <= sm_count``): a cluster
#: per row sized so that the row-CTAs fill the SMs. Rows filling the device many times over: the fewest CTAs per row
#: that keep the share under a large target (fewer cluster round trips per pass).
SMALL_ROWS_MULT = 8
#: The regimes, by how many rows share the device. Few rows (``M * SMALL_ROWS_MULT <= sm_count``): the latency regime --
#: a row spread over many CTAs with a small share each. Rows filling the device about once (``M <= sm_count``): a cluster
#: per row sized so that the row-CTAs fill the SMs. Rows filling the device many times over: the fewest CTAs per row
#: that keep the share under a large target (fewer cluster round trips per pass).
SMALL_ROWS_MULT = 8
#: The staged share of a row per CTA the rule aims at in each regime (the cluster grows until the share is under it).
SHARE_TARGET_SMALL = 16 * 1024
SHARE_TARGET_MID = 64 * 1024
SHARE_TARGET_LARGE = 128 * 1024
#: The largest cluster per regime: 16 CTAs for a few rows, 8 otherwise -- at 64 rows and beyond a 16-CTA cluster
#: schedules onto the GPCs half as well (512 threads x 16 CTAs measured 2x slower than x 8 at N = 152064 f32 on Rubin NVL8
#: and the B200, interleaved A/B).
CLUSTER_MAX_SMALL = MAX_CLUSTER
CLUSTER_MAX_MID = 8
CLUSTER_MAX_LARGE = 8
#: Threads per row-CTA: 256 while the rows do not fill the device more than once (more CTAs in flight beat wider
#: sweeps: the B200 prefers them at every share, Rubin from 128 KB on); beyond, 512 from a 96 KB share on (the radix
#: sweeps are instruction-bound and want warps once the share is long), 256 below it.
THREADS_SMALL = 256
THREADS_MID = 256
T512_SHARE_LARGE = 96 * 1024


def support_config(N: int, M: int, elem_bytes: int, sm_count: int) -> LSEConfig:
    """The default of ``support.row.v1`` for ``M`` rows of vocabulary ``N``: one row per CTA group, the row's share staged
    in shared memory through the predicated register loads (the cp.async form timed no better), the cluster and the
    thread count by the regime rules above -- a closed form of ``(N, M, elem_bytes, sm_count)`` fitted on the regret
    sweeps of the restricted space (every member timed at every grid point: ``bench_support_regret.py`` /
    ``fit_support_rule.py``, the sandbox's bench/): a joint fit over Rubin NVL8 (SM107) and the B200 (SM100) -- geomean
    regret 1.029 (81 % of the points within 5 %) and 1.070 (50 %), from 1.19 / 30 % (Rubin) for the first rule. Every pick is a member of ``all_valid_configs`` at one row per CTA, so it
    never changes bits."""
    return _rule(int(N), int(M), int(elem_bytes), int(sm_count))


@functools.lru_cache(maxsize=65536)
def _rule(N: int, M: int, elem_bytes: int, sm: int) -> LSEConfig:
    """The cached body of :func:`support_config` (its arguments as plain ints)."""
    Kp = padded_blocks(N)
    row_bytes = Kp * BLOCK * elem_bytes  # the staged row: the Kp bucket's, not N's
    few = sm >= M * SMALL_ROWS_MULT
    once = sm >= M
    if few:
        target, cmax = SHARE_TARGET_SMALL, CLUSTER_MAX_SMALL
    elif once:
        target, cmax = SHARE_TARGET_MID, CLUSTER_MAX_MID
    else:
        target, cmax = SHARE_TARGET_LARGE, CLUSTER_MAX_LARGE
    C = 1
    while cmax > C and row_bytes // C > target:
        C *= 2
    while (
        C < MAX_CLUSTER and row_bytes // C > SMEM_BUDGET
    ):  # the staging budget binds last
        C *= 2
    while C > 1 and Kp % C:
        C //= 2
    share = row_bytes // C
    if (
        share > SMEM_BUDGET
    ):  # no member stages this row: the budget binds at the largest cluster
        raise ValueError(
            f"no configuration of support.row.v1 stages a {row_bytes}-byte row within the {SMEM_BUDGET}-byte budget at "
            f"a cluster of {MAX_CLUSTER} CTAs ({share} bytes per CTA)"
        )
    if few:
        T = THREADS_SMALL
    elif once:
        T = THREADS_MID
    else:
        T = 512 if share >= T512_SHARE_LARGE else 256
    while T > LANES and (Kp // C) % (T // LANES):
        T //= 2
    vec = max_vec_elems(N, elem_bytes)
    cfg = LSEConfig(
        T, 1, vec, "regs", C
    )  # the kernel stages the share itself (stage_from_regs)
    cfg.validate(N, elem_bytes)
    return cfg
