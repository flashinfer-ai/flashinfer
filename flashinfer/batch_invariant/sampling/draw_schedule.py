# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dispatcher of ``sample.icdf.v1``: the row class's configuration space (``schedule.all_valid_configs``), a closed-form
pick, no table.

The draw kernel runs the row class's passes and then a crossing pass that unrolls per element of a thread's share, so a
long register-resident share spills (the register form at 256 elements per thread ran 8-10x slower on SM100); the kernel
stages such a share in shared memory (``draw_kernel.STAGE_ABOVE_ELEMS``) and this rule never picks one. Outside every
kernel cache closure: an edit here recompiles nothing.
"""

from __future__ import annotations

import functools

from .schedule import (
    LANES,
    LSEConfig,
    all_valid_configs,
    default_config,
    max_vec_elems,
    padded_blocks,
)

#: The throughput regime's target share per thread (the crossing pass unrolls per element).
DRAW_ELEMS_PER_THREAD = 128
#: The largest register-resident share the rule picks: the kernel stages a longer one in shared memory
#: (``draw_kernel.STAGE_ABOVE_ELEMS``, the same number; the rule stays at or below it where a member allows).
REGS_SHARE_MAX = 128
#: The largest share per thread the rule picks at all: the crossing pass unrolls per element, and a member above this
#: (the 1024-element re-read members of the odd long vocabularies) compiles for minutes; such members stay in the space
#: for the suites, never in a pick.
SHARE_MAX = 512
#: The re-read member's share above which the throughput regime takes twice the threads per row (up to 1024): at
#: N = 152064 (594 blocks padded to 1024) the 512-element member is 1.3-1.8x slower than the 1024-thread 256-element one
#: for logits on both parts and for weights on Rubin NVL8 (SM107); on the B200 (SM100) the weights' draw prefers the
#: 512-element member by 6-15 %, and the joint geomean over both parts favours 256 at every measured point. At 128256
#: (256 elements at 512 threads) the 512-thread member wins on both parts, so the pick there is unchanged.
REREAD_SHARE_MAX = 256
#: The throughput regime's threads per row: from this many ...
DRAW_THREADS_MIN = 256
#: ... up to this many, doubled while the share per thread exceeds the target.
DRAW_THREADS_MAX = 512
#: The throughput regime's vector width (at most; the wider loads lose on the re-read form's share).
DRAW_VEC_MAX = 4


def _one_row(cfg: LSEConfig) -> LSEConfig:
    """The same member at one row per CTA (the index gather's requirement; a member whenever ``cfg`` is)."""
    if cfg.rows_per_cta == 1:
        return cfg
    return LSEConfig(cfg.threads_per_row, 1, cfg.vec_elems, cfg.stage, cfg.cluster)


def draw_config(N: int, M: int, elem_bytes: int, sm_count: int) -> LSEConfig:
    """The default of ``sample.icdf.v1`` for ``M`` output rows of vocabulary ``N``: the latency regime (fewer rows than
    SMs) takes the row class's two-pass pick (``default_config(..., row_output=False)``: a cluster of CTAs per row, ~256
    input bytes per thread) at one row per CTA -- at twice the threads per row, same cluster, while its register share
    exceeds ``REGS_SHARE_MAX`` elements and that member exists (the 1-byte-vector rows of an odd long vocabulary); the
    throughput regime re-reads the row block by block (``reread``) with 256 to 512 threads per row, so a thread's share
    stays around ``DRAW_ELEMS_PER_THREAD`` elements and the per-element crossing pass does not spill (empirical constants
    from the B200 search; up to 1024 threads while the share exceeds ``REREAD_SHARE_MAX``, measured on both parts), falling
    back to the two-pass pick for the short rows whose re-read member is not enumerated. No pick has a share above
    ``SHARE_MAX`` elements (where neither regime's candidate is within it, the member with the smallest share). Every pick
    is a member of ``all_valid_configs`` at one row per CTA, so it never changes bits. When even the lightest member of
    the space exceeds ``SHARE_MAX`` (fp32 rows beyond 2^21: a 4096-element share compiles for minutes with tens of GB)
    the rule raises ``ValueError`` and the entry reports the vocabulary as unsupported."""
    return _rule(int(N), int(M), int(elem_bytes), int(sm_count))


@functools.lru_cache(maxsize=65536)
def _rule(N: int, M: int, elem_bytes: int, sm: int) -> LSEConfig:
    """The cached body of :func:`draw_config` (its arguments as plain ints)."""
    space = all_valid_configs(N, elem_bytes)
    if not space:
        raise ValueError(
            f"no configuration of the class covers a vocabulary of {N} with {elem_bytes}-byte elements"
        )
    members = set(space)
    two_pass = _one_row(default_config(N, M, elem_bytes, sm, row_output=False))
    if sm > M:
        cand = two_pass
        while (
            cand.stage == "regs"
            and cand.elems_per_thread(N) > REGS_SHARE_MAX
            and cand.threads_per_row < 1024
        ):
            nxt = LSEConfig(
                cand.threads_per_row * 2, 1, cand.vec_elems, "regs", cand.cluster
            )
            if nxt not in members:
                break
            cand = nxt
        if cand in members and cand.elems_per_thread(N) <= SHARE_MAX:
            return cand
    else:
        Kp = padded_blocks(N)
        T = DRAW_THREADS_MIN
        while T < DRAW_THREADS_MAX and N // T > DRAW_ELEMS_PER_THREAD:
            T *= 2
        while T > LANES and Kp % (T // LANES):
            T //= 2
        vec = min(DRAW_VEC_MAX, max_vec_elems(N, elem_bytes))
        cand = LSEConfig(T, 1, vec, "reread", 1)
        while (
            cand.threads_per_row < 1024 and cand.elems_per_thread(N) > REREAD_SHARE_MAX
        ):
            nxt = LSEConfig(cand.threads_per_row * 2, 1, vec, "reread", 1)
            if not nxt.is_valid(N):
                break
            cand = nxt
        if cand in members and cand.elems_per_thread(N) <= SHARE_MAX:
            return cand
        if two_pass in members and two_pass.elems_per_thread(N) <= SHARE_MAX:
            return two_pass
    # the member with the fewest elements per thread, then the fewest CTAs, then the most threads -- refused above the
    # share bound (such a member compiles for minutes with tens of GB: the crossing pass unrolls per element)
    best = min(
        (_one_row(c) for c in space),
        key=lambda c: (c.elems_per_thread(N), c.cluster, -c.threads_per_row),
    )
    if best.elems_per_thread(N) > SHARE_MAX:
        raise ValueError(
            f"no configuration of the class draws a vocabulary of {N} with {elem_bytes}-byte elements within {SHARE_MAX} "
            f"elements per thread (the lightest member has {best.elems_per_thread(N)})"
        )
    return best
