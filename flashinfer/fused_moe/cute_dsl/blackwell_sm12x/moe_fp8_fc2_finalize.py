# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public entry for the fp8 fused fc2_down + finalize arm: tile choice, compile cache, and its gates."""

import functools

import cutlass.cute as cute
import torch
from cutlass.base_dsl.common import DSLUserCodeError

from ....utils import ceil_div
from ._moe_utils.moe_epilogue import EpiMethod, scatter_supports
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder, dsl_targets_sm12x
from .kernel_moe_fp8_fc2_finalize import (
    GRAN_K,
    GRAN_N,
    SCATTER_EPIS,
    CuteDslSm120MoeFp8Fc2Finalize,
    make_args,
    make_cfg,
)
from ._moe_utils.heuristic import select_plain_bm_64_or_128

TILES = ((128, GRAN_N), (64, GRAN_N), (32, GRAN_N))
BK = GRAN_K
EPIS = SCATTER_EPIS
DEFAULT_EPI = EpiMethod.WG_SCATTER


def resolve_stage(tile, epi=DEFAULT_EPI):
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, epi=epi), tuple(tile)
    )


class CuteDslSm120GroupedFp8Fc2FinalizeOp:
    TILES = TILES

    def __init__(
        self, n: int, k: int, tile, epi: EpiMethod = DEFAULT_EPI, enable_pdl=False
    ):
        if not self.can_implement(n=n, k=k, tile=tile, epi=epi):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                f"epi={epi}"
            )
        self.n, self.k, self.tile, self.epi, self.enable_pdl = (
            n,
            k,
            tuple(tile),
            epi,
            enable_pdl,
        )
        self.ab_stage = resolve_stage(self.tile, epi)
        self.tactic = (self.tile[0], self.tile[1], epi)
        self.cfg = make_cfg(self.tile, self.ab_stage, epi=epi, enable_pdl=enable_pdl)

    @classmethod
    def is_constructible(cls, tile, epi: EpiMethod = DEFAULT_EPI) -> bool:
        if not dsl_targets_sm12x():
            return False
        try:
            CuteDslSm120MoeFp8Fc2Finalize(
                make_cfg(tuple(tile), resolve_stage(tuple(tile), epi), epi=epi), 1
            )
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(
        cls, *, n: int, k: int, tile, epi: EpiMethod = DEFAULT_EPI
    ) -> bool:
        t = tuple(tile)
        if (
            t[:2] not in cls.TILES
            or t[2] != BK
            or epi not in EPIS
            or not scatter_supports(n)
        ):
            return False
        if n <= 0 or k <= 0 or k % t[2] != 0:
            return False
        return cls.is_constructible(t, epi)

    def build(self, grid_x: int):
        return CuteDslSm120MoeFp8Fc2Finalize(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedFp8Fc2FinalizeOp.TILES
    for epi in EPIS
    if CuteDslSm120GroupedFp8Fc2FinalizeOp.is_constructible((bm, bn, BK), epi)
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, BK), epi


_COMPILED: dict = {}


def compiled_kernel(
    sample_args,
    *,
    op: CuteDslSm120GroupedFp8Fc2FinalizeOp,
    grid_x: int,
    sm_version: str,
):
    key = (op.tactic, op.enable_pdl, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, num_experts: int, num_sms: int):
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    if m_per_expert <= 32:
        return (32, GRAN_N, GRAN_K)
    return (
        select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms),
        GRAN_N,
        GRAN_K,
    )


def _check_a_scale_granularity(a_scale, k: int) -> None:
    want = ceil_div(k, GRAN_K)
    got = int(a_scale.shape[0])
    if got != want:
        raise ValueError(f"a_scale K extent {got} != {want} implied by k={k}")


def _check_b_scale_granularity(b_scale, k: int) -> None:
    want = ceil_div(k, GRAN_K)
    got = int(b_scale.shape[1])
    if got != want:
        raise ValueError(f"b_scale K extent {got} != {want} implied by k={k}")


def cute_dsl_sm12x_fc2_finalize_fp8(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    src_token,
    pair_scales,
    num_tokens: int,
    tile=None,
    epi: EpiMethod = DEFAULT_EPI,
    enable_pdl: bool = False,
):
    n, k = int(b_q.shape[1]), int(b_q.shape[2])
    _check_a_scale_granularity(a_scale, k)
    _check_b_scale_granularity(b_scale, k)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    if tile is None:
        tile = select_tile(
            total_rows=int(a_q.shape[0]),
            n=n,
            num_experts=int(b_q.shape[0]),
            num_sms=grid_x,
        )
    op = CuteDslSm120GroupedFp8Fc2FinalizeOp(n, k, tuple(tile), epi, enable_pdl)
    out = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=a_q.device)
    args = make_args(
        a_q,
        a_scale,
        b_q,
        b_scale,
        out,
        src_token.to(torch.int32),
        pair_scales.to(torch.float32),
        m_indptr.to(torch.int32),
    )
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return out
