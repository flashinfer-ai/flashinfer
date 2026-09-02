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
"""mx fc1_gate_up + SiLU + mxfp8 quant on CuteDSL: what one compiled kernel is specialized on."""

import functools

import cutlass.cute as cute
import torch
from cutlass.base_dsl.common import DSLUserCodeError

from ....tllm_enums import (
    DEFAULT_SITU_BETA as SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA as SITU_LINEAR_BETA,
    ActivationType,
)
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN, UE8M0_PACK_NUM
from ....utils import ceil_div
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset
from ._moe_utils.moe_epilogue import EpiMethod
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder, dsl_targets_sm12x
from .kernel_moe_mxfp8_mxfp4_fc1_act_q1 import (
    GRANK_A,
    GRANK_B,
    QUANT_EPIS,
    CuteDslSm120MoeMxfp8Mxfp4Fc1ActQ1,
    is_swapab,
    make_args,
    make_cfg,
)
from ._moe_utils.heuristic import select_fc1_act_tile

BK = GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF
EPIS = QUANT_EPIS
DEFAULT_EPI = EpiMethod.WG_S2R_QUANT


def resolve_stage(tile, epi=DEFAULT_EPI):
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, epi=epi, activation=ActivationType.Swiglu),
        tuple(tile),
    )


def out_sf_shape(m: int, n: int, num_experts: int):
    na = ceil_div(n, GRANK_A * UE8M0_PACK_NUM)
    return (na, compute_padded_offset(m, num_experts, SF_M_ALIGN))


class CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op:
    OUT_DTYPES = (torch.float8_e4m3fn,)
    TILES = ((64, 128), (32, 128), (8, 128))
    ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situ)

    def __init__(
        self,
        *,
        n: int,
        k: int,
        tile,
        out_dtype: torch.dtype,
        activation: ActivationType,
        epi: EpiMethod = DEFAULT_EPI,
        enable_pdl=False,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    ):
        if not self.can_implement(
            n=n, k=k, tile=tile, out_dtype=out_dtype, activation=activation, epi=epi
        ):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                f"epi={epi} activation={activation} out_dtype={out_dtype}"
            )
        self.n, self.k, self.tile, self.out_dtype = n, k, tuple(tile), out_dtype
        self.activation, self.epi, self.enable_pdl = activation, epi, enable_pdl
        self.situ_beta, self.situ_linear_beta = situ_beta, situ_linear_beta
        self.ab_stage = resolve_stage(self.tile, epi)
        self.tactic = (self.tile[0], self.tile[1], epi)
        self.cfg = make_cfg(
            self.tile,
            self.ab_stage,
            epi=epi,
            activation=activation,
            enable_pdl=enable_pdl,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (bm, bn) in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op.TILES and bk == BK

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return (
            n > 0
            and k > 0
            and k % tile[2] == 0
            and n % SF_M_ALIGN == 0
            and n % tile[1] == 0
            and n % BK == 0
        )

    @classmethod
    def is_constructible(
        cls, tile, epi=DEFAULT_EPI, activation=ActivationType.Swiglu
    ) -> bool:
        if not dsl_targets_sm12x():
            return False
        try:
            stage = resolve_stage(tuple(tile), epi)
            CuteDslSm120MoeMxfp8Mxfp4Fc1ActQ1(
                make_cfg(tuple(tile), stage, epi=epi, activation=activation), 1
            )
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(
        cls,
        *,
        n: int,
        k: int,
        tile,
        out_dtype,
        activation=ActivationType.Swiglu,
        epi: EpiMethod = DEFAULT_EPI,
    ) -> bool:
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and activation in cls.ACTIVATIONS
            and epi in epi_tactics(tile)
            and cls.is_valid_alignment(n, k, tile)
            and cls.is_constructible(tile, epi, activation)
        )

    def build(self, grid_x: int) -> CuteDslSm120MoeMxfp8Mxfp4Fc1ActQ1:
        return CuteDslSm120MoeMxfp8Mxfp4Fc1ActQ1(self.cfg, grid_x)


def epi_tactics(tile):
    return (EpiMethod.WG_QUANT_SWAP,) if is_swapab(tile) else EPIS


TACTICS = tuple(
    (bm, bn, e)
    for bm, bn in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op.TILES
    for e in epi_tactics((bm, bn, BK))
    if CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op.is_constructible((bm, bn, BK), e)
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, BK), epi


@functools.lru_cache(maxsize=None)
def _op(
    n: int,
    k: int,
    tile,
    out_dtype,
    activation,
    epi,
    enable_pdl=False,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
) -> CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op:
    return CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op(
        n=n,
        k=k,
        tile=tile,
        out_dtype=out_dtype,
        activation=activation,
        epi=epi,
        enable_pdl=enable_pdl,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op, grid_x: int, sm_version: str):
    key = (
        op.tactic,
        op.activation,
        op.out_dtype,
        op.enable_pdl,
        op.situ_beta,
        op.situ_linear_beta,
        grid_x,
        sm_version,
    )
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    return select_fc1_act_tile(
        total_rows=total_rows,
        n=n,
        num_experts=num_experts,
        num_sms=num_sms,
        tiles=CuteDslSm120GroupedMxfp8Mxfp4Fc1ActQ1Op.TILES,
        gran_k=BK,
    )


def _check_a_scale_granularity(a_scale, k: int) -> None:
    want = ceil_div(k, GRANK_A * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)
    got = int(a_scale.shape[0])
    if got != want:
        raise ValueError(f"a_scale K extent {got} != {want} implied by k={k}")


def _check_b_scale_granularity(b_scale, k: int) -> None:
    want = ceil_div(k, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)
    got = int(b_scale.shape[1])
    if got != want:
        raise ValueError(f"b_scale K extent {got} != {want} implied by k={k}")


def cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    tile=None,
    epi: EpiMethod = DEFAULT_EPI,
    enable_pdl: bool = False,
    *,
    activation: ActivationType = ActivationType.Swiglu,
    situ_beta: float = SITU_BETA,
    situ_linear_beta: float = SITU_LINEAR_BETA,
):
    m, n, k = int(a_q.shape[0]), int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
    _check_a_scale_granularity(a_scale, k)
    _check_b_scale_granularity(b_scale, k)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    num_experts = int(b_q.shape[0])
    if tile is None:
        tile = select_tile(
            total_rows=m, n=n, k=k, num_experts=num_experts, num_sms=grid_x
        )
    op = _op(
        n,
        k,
        tuple(tile),
        out_dtype,
        activation,
        epi,
        enable_pdl,
        situ_beta,
        situ_linear_beta,
    )
    q = torch.empty(m, n, dtype=out_dtype, device=a_q.device)
    sf = torch.zeros(out_sf_shape(m, n, num_experts), dtype=torch.int32, device=a_q.device)
    args = make_args(a_q, a_scale, b_q, b_scale, q, sf.view(torch.uint8), m_indptr.to(torch.int32))
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return q, sf
