# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 grouped GEMMs' dispatcher: the dense fp8 class's ranking heuristic at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`): a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model at the dense shape whose tiles, M tail and operand bytes are the grouped
launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by side), for the fp8 format
(``in_fmt="e4m3"``: the format's MMA rate and operand bytes), mapped onto the grouped space and filtered by the grouped fp8
body's acceptance. Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.schedule import rank_topk
from .blocks.fp8_epilogue import GEMM2_CHAIN, Fp8Chain
from .schedule import TOP_K, _grouped, active_experts, grouped_knobs, rows_per_expert

IN_FMT = "e4m3"


def gemm1_chain(act: str = "silu", params=(), gated: bool = True) -> Fp8Chain:
    """GEMM1 of the W8A8 class: the gated or the ungated activation's fused quantization."""
    return Fp8Chain("glu" if gated else "act", act, "e4m3", tuple(params), True)


def all_valid_configs_fp8(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN, clusters=None
) -> List[Knobs]:
    """The configuration space the grouped fp8 guarantee is claimed over, for a weight shape and a chain."""
    from .blocks.fp8_compile import can_implement_grouped_fp8

    return [
        k
        for k in grouped_knobs(clusters)
        if can_implement_grouped_fp8(k, N, K, E, chain)
    ]


def rank_grouped_fp8(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Fp8Chain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
) -> List[Knobs]:
    """The ranked grouped records for one fp8 GEMM: the dense fp8 ranking at ``(M = rows, N = n_active * N)`` mapped onto the
    grouped space, in rank order, de-duplicated, at least ``k`` records when the space has them."""
    from .blocks.fp8_compile import can_implement_grouped_fp8

    out: List[Knobs] = []
    seen = set()
    ranked = rank_topk(
        rows,
        (n_active or E) * N,
        K,
        1,
        "k",
        sm_count,
        max(k, 10),
        family,
        IN_FMT,
        nominal,
    )
    for cand in [_grouped(c) for c in ranked] + grouped_knobs():
        if cand in seen or not can_implement_grouped_fp8(cand, N, K, E, chain):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= k:
            break
    return out


def default_configs_fp8(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    gated: bool = True,
    nominal: bool = False,
    chain1: Optional[Fp8Chain] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I ungated), config for GEMM2 (N = H, K = I))``: the first record of each ranked
    list. ``chain1`` names GEMM1's chain (the activation's; its acceptance is the chain kind's)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    c1chain = (
        chain1
        if chain1 is not None
        else gemm1_chain("silu" if gated else "relu2", (), gated)
    )
    n1 = 2 * I if gated else I
    c1 = rank_grouped_fp8(rows, n1, H, E, c1chain, sm_count, family, 1, nominal, n_act)
    c2 = rank_grouped_fp8(
        rows, H, I, E, GEMM2_CHAIN, sm_count, family, 1, nominal, n_act
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped fp8 record implements (H={H}, I={I}, E={E}, gated={gated})"
        )
    return c1[0], c2[0]
