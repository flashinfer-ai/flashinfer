# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch of the GEMM classes: the chunk rule (a function of the weight shape and the format only) and the heuristic
dispatcher.

Every record of a configuration space computes the class (R3), so the choice here never reaches the bits; it only
moves time. The choice is the heuristic's best record of the space (``heuristic.rank``); ``rank_topk`` is the autotuner's
tactic list.
"""

from __future__ import annotations

from typing import List, Optional

from .generator import CLUSTERS, SWIZZLES, Knobs, enumerate_space  # noqa: F401  (re-exported)
from .spec import FP8_BLOCKSCALE_FMTS, chunk_unit

GemmConfig = Knobs


def all_valid_configs(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    in_fmt: str = "bf16",
) -> List[GemmConfig]:
    """The configuration space for this shape, layout and class: the generator's knob product filtered by the kernel's
    own ``can_implement``: the set R3 is claimed over. The GPU configuration-agreement tests check the dispatcher's
    candidates and sampled records of it against each other bit for bit."""
    return enumerate_space(M, N, K, chunks=chunks, b_major=b_major, in_fmt=in_fmt)


#: How many records of a ranking are kept per (shape, layout, format, part): the autotuner's candidate count. A cached
#: entry holds that prefix, not the whole ranked space (M changes almost every call in serving).
RANK_CACHE_TOPK = 10
SPLIT_MIN_CTAS = 256  # below this many N tiles at 32 wide, a K split is considered
SPLIT_MAX_CHUNKS = 128
SPLIT_MIN_KC = 256  # elements of K per chunk at least


def chunks_for(N: int, K: int, in_fmt: str = "bf16") -> int:
    """Fixed-chunk split count of a weight shape in a format. A function of (N, K, format) ONLY: the same bits for every
    M and on every part. Budget: the fp32 partial round trip (8 * C * M * N bytes) stays below a quarter of the bf16 A
    traffic (2 * M * K bytes), i.e. C <= K / (16 N) (the same budget for the fp8 classes); K / C must be a whole number
    of the format's MMA steps (16 bf16 / 32 fp8 elements, so every chunk is exactly a ``kchunk`` fold of the
    specification) and >= SPLIT_MIN_KC; and no split at all when N alone gives enough CTAs. Because the rule takes no
    M, a narrow weight runs the chunked class at every M, large prefills included (the fp32 partial round trip is then
    a bounded fraction of the operand traffic). An empty weight (N = 0) has no split, as the Lean mirror's natural-number
    division gives. Mirrored by ``BIK.GEMM.chunksFor`` / ``chunksForFp8`` in Lean (checked by the generated conformance
    cases)."""
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        return 1  # the block-scale class has no split variant: its block fold already partitions K
    step = chunk_unit(
        in_fmt
    )  # the format's MMA step; for the block-scaled classes also a whole scale-factor atom column
    if N <= 0 or -(-N // 32) >= SPLIT_MIN_CTAS:
        return 1
    budget = K // (16 * N)
    C = 1
    while (
        min(budget, SPLIT_MAX_CHUNKS) >= C * 2
        and K % (step * C * 2) == 0
        and K // (C * 2) >= SPLIT_MIN_KC
    ):
        C *= 2
    return C


#: The sm_107 instructions' classes of the MX formats (the sm_100 / sm_103 instructions' classes elsewhere).
MX_SM107_CLASS = {"mxfp8": "mxfp8_k64", "mxfp4": "mxfp4_k128"}


def mx_class_for(base: str, family: Optional[str]) -> str:
    """The MX class a part runs for an MX format: the sm_107 instruction's class (64 elements per step for mxfp8,
    ``mxfp8_k64``; 128 for mxfp4, ``mxfp4_k128``) on the sm_107 parts, the sm_100 instruction's class (``mxfp8`` /
    ``mxfp4``) elsewhere. A function of the architecture family only (the fingerprint is scoped to the architecture),
    never of the shape."""
    return MX_SM107_CLASS.get(base, base) if family == "sm_107" else base


def fp4_class_for(N: int, K: int, family: Optional[str]) -> str:
    """The NVFP4 class a part runs for a weight shape: on the sm_107 parts the sm_107 instruction's class (128 elements per
    step, ``nvfp4_k128``) for every weight shape whose chunk rule at that step is unsplit, the sm_100 instruction's class
    (``nvfp4``) for the split shapes; ``nvfp4`` elsewhere. A function of (N, K) and the architecture family only, never of M
    (a token's bits may not depend on the batch), so a weight that takes the wider step at prefill takes it at decode too."""
    if family == "sm_107" and chunks_for(N, K, "nvfp4_k128") == 1:
        return "nvfp4_k128"
    return "nvfp4"


def rank_topk(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    k: int = 10,
    family: Optional[str] = None,
    in_fmt: str = "bf16",
    nominal: bool = False,
) -> List[GemmConfig]:
    """The dispatcher's ranked list (what FlashInfer's autotuner profiles under ``batch_invariant=True``): `heuristic.rank`'s
    order (the heuristic's best first) with the raster twin of each of the best ceil(k/2) records inserted after it,
    distinct records, at most ``k``; ``default_config(...) == rank_topk(...)[0]``. Every entry
    is a member of the class's enumerated space, so an autotuner may pick among them freely without changing a single bit.
    ``sm_count`` and ``family`` name the part (the operand device's); ``nominal`` ranks at the family's nominal table (the
    generated Lean cases), never a device's queried numbers. Empty when nothing implements the shape."""
    import dataclasses

    from .blocks.operands import can_implement
    from .heuristic import rank

    ranked: List[GemmConfig] = [
        rec
        for rec, _t in rank(
            M,
            N,
            K,
            chunks,
            b_major,
            2,
            sm_count,
            family,
            in_fmt=in_fmt,
            nominal=nominal,
            topk=max(k, RANK_CACHE_TOPK),
        )
    ]  # the best max(k, RANK_CACHE_TOPK) records, distinct by construction (the enumerated space has no duplicates)
    seen: set = set()
    out: List[GemmConfig] = []
    # the best ceil(k/2) records, each followed by its raster twin when the space has it: the one knob the estimate does
    # not price is hedged for free by the autotuner; then the rank order fills the list. The raster order is a knob of
    # every enumerated record, so the twin is a member exactly when the body accepts it (`can_implement`); the truncated
    # ranking is not consulted for membership.
    for rec in ranked[: -(-k // 2)]:
        twin = dataclasses.replace(
            rec, raster_along="n" if rec.raster_along == "m" else "m"
        )
        for cand in (rec, twin):
            if cand not in seen and (
                cand is rec or can_implement(cand, M, N, K, b_major, chunks, in_fmt)
            ):
                out.append(cand)
                seen.add(cand)
    for rec in ranked:
        if len(out) >= k:
            break
        if rec not in seen:
            out.append(rec)
            seen.add(rec)
    return out[:k]


def default_config(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    in_fmt: str = "bf16",
    nominal: bool = False,
) -> GemmConfig:
    """The dispatcher's pick: `rank_topk(...)[0]`, the heuristic's best record of the class's space for the shape, the layout,
    the format and the part. May depend on anything: every record is canonical."""
    top = rank_topk(M, N, K, chunks, b_major, sm_count, 1, family, in_fmt, nominal)
    if not top:
        raise ValueError(
            f"no record of the class implements {(M, N, K)} ({in_fmt}) in layout {b_major} with {chunks} chunks"
        )
    return top[0]
