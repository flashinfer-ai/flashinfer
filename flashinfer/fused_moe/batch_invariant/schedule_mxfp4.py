# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 MoE classes' grouped-GEMM dispatcher: the NVFP4 classes' ranking (:mod:`.schedule_nvfp4`, the dense 4-bit
class's heuristic at the grouped launch's shape over the grouped and the swapped record spaces) at the MXFP4 instruction
class, with this class's chain term (the bf16 chain on the raw accumulators: no row scale, no block quantization)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import Knobs
from .blocks.epilogue import Mxfp4Chain
from .schedule import TOP_K, active_experts, rows_per_expert
from .schedule_nvfp4 import rank_grouped_nvfp4


def chain_nodes_mx(chain: Mxfp4Chain) -> int:
    """The node count of the chain's term per output element (the exposed epilogue's cost unit): the bf16 class's gated /
    ungated chain on the raw accumulators, or GEMM2's conversion."""
    from . import mterm as term
    from . import spec as S

    alg = term.TermAlgebra()
    g, u = alg.leaf("g", 0, 0), alg.leaf("u", 0, 0)
    if chain.pair:
        t = S.glu_elem(alg, g, u, chain.act, chain.out_fmt, chain.params)
    elif chain.kind == "act":
        t = S.act_elem(alg, u, chain.act, chain.out_fmt, chain.params)
    else:
        t = alg.cvt_out(u, chain.out_fmt)
    return term.size(t)


def rank_grouped_mxfp4(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Mxfp4Chain,
    in_fmt: str,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
    gather: Optional[bool] = False,
) -> List[Knobs]:
    """The ranked records for one GEMM over both record spaces (:func:`.schedule_nvfp4.rank_grouped_nvfp4` at the MXFP4
    instruction class with this class's chain term)."""
    return rank_grouped_nvfp4(
        rows,
        N,
        K,
        E,
        chain,
        in_fmt,
        sm_count,
        family,
        k,
        nominal,
        n_active,
        swap,
        gather,
    )


def default_configs_mxfp4(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    in_fmts: Tuple[str, str],
    chain1: Mxfp4Chain,
    chain2: Mxfp4Chain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
    gather: Optional[bool] = False,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list
    (``gather`` as :func:`.schedule_nvfp4.default_configs_nvfp4`)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_mxfp4(
        rows,
        n1,
        H,
        E,
        chain1,
        in_fmts[0],
        sm_count,
        family,
        1,
        nominal,
        n_act,
        swap,
        gather,
    )
    c2 = rank_grouped_mxfp4(
        rows, H, I, E, chain2, in_fmts[1], sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped block-scaled record implements (H={H}, I={I}, E={E}, {in_fmts})"
        )
    return c1[0], c2[0]
