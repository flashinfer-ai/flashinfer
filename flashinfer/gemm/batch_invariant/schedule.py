# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch of the bf16 GEMM: the chunk rule (a function of the weight shape only) and the cost-model dispatcher.

Every record of the configuration space computes the class (R3), so the choice here never reaches the bits; it only
moves time. The choice is the cost model's argmin over the space (``cost_model.rank``); ``rank_topk`` is the autotuner's
tactic list.
"""

from __future__ import annotations

from typing import List, Optional

from .generator import CLUSTERS, SWIZZLES, Knobs, enumerate_space  # noqa: F401  (re-exported)
from .spec import MMA_K

GemmConfig = Knobs


def all_valid_configs(
    M: int, N: int, K: int, chunks: int = 1, b_major: str = "n"
) -> List[GemmConfig]:
    """The configuration space for this shape and class: the generator's knob product filtered by the kernel's own
    ``can_implement``: the set R3 is claimed over. The GPU configuration-agreement tests check the dispatcher's candidates
    and sampled records of it against each other bit for bit."""
    return enumerate_space(M, N, K, chunks=chunks, b_major=b_major)


SPLIT_MIN_CTAS = 256  # below this many N tiles at 32 wide, a K split is considered
SPLIT_MAX_CHUNKS = 128
SPLIT_MIN_KC = 256  # elements of K per chunk at least


def chunks_for(N: int, K: int) -> int:
    """Fixed-chunk split count of a weight shape. A function of (N, K) ONLY: the same bits for every M and on every
    part. Budget: the fp32 partial round trip (8 * C * M * N bytes) stays below a quarter of the A traffic
    (2 * M * K bytes), i.e. C <= K / (16 N); K / C must be a whole number of 16-wide MMA steps (so every chunk is
    exactly a ``kchunk`` fold of the specification) and >= SPLIT_MIN_KC; and no split at all when N alone gives
    enough CTAs. Because the rule takes no M, a narrow weight runs the chunked class at every M, large prefills included
    (the fp32 partial round trip is then a bounded fraction of the operand traffic). An empty weight (N = 0) has no
    split, as the Lean mirror's natural-number division gives. Mirrored by ``BIK.GEMM.chunksFor`` in Lean (checked by
    the generated conformance cases)."""
    if N <= 0 or -(-N // 32) >= SPLIT_MIN_CTAS:
        return 1
    budget = K // (16 * N)
    C = 1
    while (
        min(budget, SPLIT_MAX_CHUNKS) >= C * 2
        and K % (MMA_K * C * 2) == 0
        and K // (C * 2) >= SPLIT_MIN_KC
    ):
        C *= 2
    return C


def rank_topk(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    k: int = 10,
    family: Optional[str] = None,
) -> List[GemmConfig]:
    """The dispatcher's ranked list (what FlashInfer's autotuner profiles under ``batch_invariant=True``): `cost_model.rank`'s
    order (the model's argmin first), distinct records, at most ``k``; ``default_config(...) == rank_topk(...)[0]``. Every entry
    is a member of the class's enumerated space, so an autotuner may pick among them freely without changing a single bit.
    ``sm_count`` and ``family`` name the part (the operand device's). Empty when nothing implements the shape."""
    from .cost_model import rank

    out: List[GemmConfig] = []
    for rec, _t in rank(M, N, K, chunks, b_major, 2, sm_count, family):
        if len(out) >= k:
            break
        if rec not in out:
            out.append(rec)
    return out


def default_config(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
) -> GemmConfig:
    """The dispatcher's pick: `rank_topk(...)[0]`, the cost model's argmin over the class's space for the shape, the layout and
    the part. May depend on anything: every record is canonical."""
    top = rank_topk(M, N, K, chunks, b_major, sm_count, 1, family)
    if not top:
        raise ValueError(
            f"no record of the class implements {(M, N, K)} in layout {b_major} with {chunks} chunks"
        )
    return top[0]
