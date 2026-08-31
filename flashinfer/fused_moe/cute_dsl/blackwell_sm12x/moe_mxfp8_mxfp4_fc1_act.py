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
"""mx fused fc1_gate_up + SiLU on CuteDSL: what one compiled kernel is specialized on, and the entry."""

import functools
import os

import cutlass.cute as cute
import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from cutlass.base_dsl.common import DSLUserCodeError

from ....tllm_enums import (
    DEFAULT_SITU_BETA as SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA as SITU_LINEAR_BETA,
    ActivationType,
)
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN
from ....utils import ceil_div
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.moe_epilogue import EpiMethod
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder, dsl_targets_sm12x
from .kernel_moe_mxfp8_mxfp4_fc1_act import (
    GRANK_A,
    GRANK_B,
    CuteDslSm120MoeMxfp8Mxfp4Fc1Act,
    is_swapab,
    make_args,
    make_cfg,
)
from ._moe_utils.heuristic import select_fc1_act_tile

BK = GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF


def epi_tactics(tile):
    return (EpiMethod.DIRECT_STG,) if is_swapab(tile) else (EpiMethod.R2G_WG,)


def resolve_stage(tile, epi):
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, epi=epi, activation=ActivationType.Swiglu),
        tuple(tile),
    )


def resolve_stage_store(tile):
    best = None
    for epi in epi_tactics(tile):
        try:
            stage = resolve_stage(tile, epi)
        except (AssertionError, ValueError, DSLUserCodeError):
            continue
        if best is None or stage > best[0]:
            best = (stage, epi)
    if best is None:
        raise ValueError("no store method fits tile %s" % (tuple(tile),))
    return best


class CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp:
    OUT_DTYPES = (torch.bfloat16,)
    TILES = ((128, 64), (64, 128), (32, 128), (8, 128))
    ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situ)

    def __init__(
        self,
        *,
        n: int,
        k: int,
        tile,
        out_dtype: torch.dtype,
        activation: ActivationType,
        epi=None,
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
        self.activation, self.enable_pdl = activation, enable_pdl
        self.situ_beta, self.situ_linear_beta = situ_beta, situ_linear_beta
        if epi is None:
            self.ab_stage, self.epi = resolve_stage_store(self.tile)
        else:
            self.ab_stage, self.epi = resolve_stage(self.tile, epi), epi
        self.tactic = (self.tile[0], self.tile[1], self.epi)
        self.cfg = make_cfg(
            self.tile,
            self.ab_stage,
            epi=self.epi,
            activation=activation,
            enable_pdl=enable_pdl,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (bm, bn) in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.TILES and bk == BK

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return n > 0 and k > 0 and k % tile[2] == 0 and n % SF_M_ALIGN == 0

    @classmethod
    def is_constructible(cls, tile, epi=None, activation=ActivationType.Swiglu) -> bool:
        if not dsl_targets_sm12x():
            return False
        try:
            if epi is None:
                stage, epi = resolve_stage_store(tuple(tile))
            else:
                stage = resolve_stage(tuple(tile), epi)
            CuteDslSm120MoeMxfp8Mxfp4Fc1Act(
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
        epi=None,
    ) -> bool:
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and activation in cls.ACTIVATIONS
            and (epi is None or epi in epi_tactics(tile))
            and cls.is_valid_alignment(n, k, tile)
            and cls.is_constructible(tile, epi, activation)
        )

    def build(self, grid_x: int) -> CuteDslSm120MoeMxfp8Mxfp4Fc1Act:
        return CuteDslSm120MoeMxfp8Mxfp4Fc1Act(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.TILES
    for epi in epi_tactics((bm, bn, BK))
    if CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.is_constructible((bm, bn, BK), epi)
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
    epi=None,
    enable_pdl=False,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
) -> CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp:
    return CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp(
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
        tiles=CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.TILES,
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


def cute_dsl_sm12x_fc1_act_mxfp8_mxfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    out_dtype: torch.dtype = torch.bfloat16,
    tile=None,
    epi=None,
    tune=None,
    enable_pdl: bool = False,
    *,
    activation: ActivationType = ActivationType.Swiglu,
    situ_beta: float = SITU_BETA,
    situ_linear_beta: float = SITU_LINEAR_BETA,
) -> torch.Tensor:
    n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
    _check_a_scale_granularity(a_scale, k)
    _check_b_scale_granularity(b_scale, k)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    if tile is None and tune is not False:
        chosen = None
        if MOE_AUTOTUNE_ENABLED():
            with autotune():
                _, tactic = AutoTuner.get().choose_one(
                    "cute_dsl_sm12x_fc1_act_mxfp8_mxfp4",
                    [
                        _Fc1ActRunner(
                            out_dtype,
                            activation,
                            enable_pdl,
                            situ_beta,
                            situ_linear_beta,
                        )
                    ],
                    TuningConfig(),
                    [a_q, a_scale, b_q, b_scale, m_indptr],
                )
            if tactic != -1:
                chosen = tactic
        if chosen is not None:
            tile, epi = split_tactic(chosen)
    if tile is None:
        tile = select_tile(
            total_rows=int(a_q.shape[0]),
            n=n,
            k=k,
            num_experts=int(b_q.shape[0]),
            num_sms=grid_x,
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
    out = torch.zeros(int(a_q.shape[0]), n, dtype=out_dtype, device=a_q.device)
    args = make_args(a_q, a_scale, b_q, b_scale, out, m_indptr.to(torch.int32))
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return out


class _Fc1ActRunner(TunableRunner):
    def __init__(
        self,
        out_dtype,
        activation,
        enable_pdl=False,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    ):
        self.out_dtype = out_dtype
        self._out = None
        self.activation, self.enable_pdl = activation, enable_pdl
        self.situ_beta, self.situ_linear_beta = situ_beta, situ_linear_beta

    def __hash__(self) -> int:
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile):
        return self.valid_tactics(inputs)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if self._out is None:
            self._out = self.alloc_out(inputs)
        if do_preparation:
            return self._out
        self.launch(inputs, self._out, None if tactic == -1 else tactic)
        return self._out

    def valid_tactics(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
        return [
            t
            for t in TACTICS
            if CuteDslSm120GroupedMxfp8Mxfp4Fc1ActOp.can_implement(
                n=n,
                k=k,
                tile=split_tactic(t)[0],
                out_dtype=self.out_dtype,
                activation=self.activation,
                epi=split_tactic(t)[1],
            )
        ]

    def alloc_out(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n = int(b_q.shape[1]) // 2
        return torch.zeros(
            int(a_q.shape[0]), n, dtype=self.out_dtype, device=a_q.device
        )

    def launch(self, inputs, out, tactic):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
        props = torch.cuda.get_device_properties(a_q.device)
        grid_x = props.multi_processor_count
        epi = None
        if tactic is None:
            tile = select_tile(
                total_rows=int(a_q.shape[0]),
                n=n,
                k=k,
                num_experts=int(b_q.shape[0]),
                num_sms=grid_x,
            )
        else:
            tile, epi = split_tactic(tactic)
        op = _op(
            n,
            k,
            tuple(tile),
            self.out_dtype,
            self.activation,
            epi,
            self.enable_pdl,
            self.situ_beta,
            self.situ_linear_beta,
        )
        args = make_args(a_q, a_scale, b_q, b_scale, out, m_indptr.to(torch.int32))
        sm_version = "sm_{}{}".format(props.major, props.minor)
        compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)

    def get_cache_key_extras(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        return (
            int(a_q.shape[0]),
            int(b_q.shape[0]),
            int(b_q.shape[1]),
            int(b_q.shape[2]) * 2,
            str(self.out_dtype),
            self.activation,
            self.enable_pdl,
            self.situ_beta,
            self.situ_linear_beta,
            torch.cuda.get_device_capability(),
        )


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
