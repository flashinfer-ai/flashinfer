# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 grouped GEMMs' dispatcher: the shared ranking contract of :mod:`.schedule` at the fp8 per-tensor classes' chains.

As for the bf16 classes: a grouped GEMM is the dense class under a different scheduler, so its records are ranked by the
dense dispatcher's model at the dense shape whose tiles, M tail and operand bytes are the grouped launch's (``M`` = the rows
of one active expert, ``N`` = the active experts' weight rows side by side), for the fp8 format (the chain's ``in_fmt``:
the format's MMA rate and operand bytes, the e4m3 output of GEMM1's chain one byte, the W8A8 chain's node count in the
exposed-epilogue term), over the grouped and the swapped record spaces of the format, filtered by the bodies' acceptance.
The functions here name the fp8 classes' entry points on :mod:`.schedule`'s functions (one ranking, one record cost, one
memo). Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import Knobs
from .blocks.epilogue import GEMM2_CHAIN, Fp8Chain
from .schedule import (
    TOP_K,
    active_experts,
    all_valid_configs,
    all_valid_configs_bf16,
    all_valid_configs_swapped,
    chain_nodes,
    rank_grouped,
    rows_per_expert,
    swapped_knobs,
)

IN_FMT = "e4m3"


def gemm1_chain(act: str = "silu", params=(), gated: bool = True) -> Fp8Chain:
    """GEMM1 of the W8A8 class: the gated or the ungated activation's fused quantization."""
    return Fp8Chain("glu" if gated else "act", act, "e4m3", tuple(params), True)


def all_valid_configs_fp8(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN, clusters=None
) -> List[Knobs]:
    """The grouped record space the fp8 guarantee is claimed over, for a weight shape and a chain."""
    return all_valid_configs(N, K, E, chain, clusters)


def swapped_knobs_fp8() -> List[Knobs]:
    """The swapped record space of the fp8 classes (:mod:`.blocks.swap`): the swapped 1-CTA records at 128 or 64 weight rows
    x 16 / 32 tokens (``kind::f8f6f4`` takes N >= 16), no cluster, swizzle 1. Disjoint from the grouped space (the swapped
    flag); the body reads the same dense rows through a TMA box."""
    return swapped_knobs(IN_FMT)


def all_valid_configs_swapped_fp8(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN
) -> List[Knobs]:
    """The swapped records the swapped body accepts for a weight shape and a chain of the fp8 classes."""
    return all_valid_configs_swapped(N, K, E, chain)


def all_valid_configs_fp8_both(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN, clusters=None
) -> List[Knobs]:
    """Both record spaces of the fp8 classes for a weight shape and a chain: the grouped records, then the swapped ones."""
    return all_valid_configs_bf16(N, K, E, chain, clusters)


def chain_nodes_fp8(chain: Fp8Chain) -> int:
    """The node count of a W8A8 chain's term (:func:`.schedule.chain_nodes`): the exposed-epilogue term's cost unit."""
    return chain_nodes(chain)


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
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked records for one fp8 GEMM over both record spaces: :func:`.schedule.rank_grouped` at the chain's format."""
    return rank_grouped(
        rows, N, K, E, chain, sm_count, family, k, nominal, n_active, swap, IN_FMT
    )


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
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I ungated), config for GEMM2 (N = H, K = I))``: the first record of each ranked
    list over both record spaces (``swap``: None, or True / False to keep only / exclude the swapped records). ``chain1`` names
    GEMM1's chain (the activation's; its acceptance is the chain kind's)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    c1chain = (
        chain1
        if chain1 is not None
        else gemm1_chain("silu" if gated else "relu2", (), gated)
    )
    n1 = 2 * I if gated else I
    c1 = rank_grouped_fp8(
        rows, n1, H, E, c1chain, sm_count, family, 1, nominal, n_act, swap
    )
    c2 = rank_grouped_fp8(
        rows, H, I, E, GEMM2_CHAIN, sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped fp8 record implements (H={H}, I={I}, E={E}, gated={gated})"
        )
    return c1[0], c2[0]
