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
"""Public entry for the fused fc2_down + finalize arm: tile choice, compile cache, and its gates."""

import functools
import os

import cutlass.cute as cute
import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from cutlass.base_dsl.common import DSLUserCodeError

from ....utils import ceil_div
from ._moe_utils.moe_epilogue import EpiMethod, scatter_supports
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder, dsl_targets_sm12x
from .kernel_moe_mxfp8_mxfp4_fc2_finalize import (
    GRANK_A,
    GRANK_B,
    SCATTER_EPIS,
    CuteDslSm120MoeMxfp8Mxfp4Fc2Finalize,
    make_args,
    make_cfg,
)

TILES = ((128, 128), (64, 128), (32, 128))
BK = GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF
EPIS = SCATTER_EPIS
DEFAULT_EPI = EpiMethod.WG_SCATTER


def resolve_stage(tile, epi: EpiMethod = DEFAULT_EPI):
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, epi=epi), tuple(tile)
    )


class CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp:
    TILES = TILES

    def __init__(
        self, n: int, k: int, tile, epi: EpiMethod = DEFAULT_EPI, enable_pdl=False
    ):
        if not self.can_implement(n=n, k=k, tile=tile, epi=epi):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                f"epi={epi}"
            )
        self.n, self.k, self.tile = n, k, tuple(tile)
        self.epi, self.enable_pdl = epi, enable_pdl
        self.ab_stage = resolve_stage(self.tile, epi)
        self.tactic = (self.tile[0], self.tile[1], epi)
        self.cfg = make_cfg(self.tile, self.ab_stage, epi=epi, enable_pdl=enable_pdl)

    @classmethod
    def is_constructible(cls, tile, epi: EpiMethod = DEFAULT_EPI) -> bool:
        if not dsl_targets_sm12x():
            return False
        try:
            CuteDslSm120MoeMxfp8Mxfp4Fc2Finalize(
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
        return CuteDslSm120MoeMxfp8Mxfp4Fc2Finalize(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp.TILES
    for epi in EPIS
    if CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp.is_constructible((bm, bn, BK), epi)
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, BK), epi


_COMPILED: dict = {}


def compiled_kernel(
    sample_args,
    *,
    op: CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp,
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
        return (32, 128, BK)
    return (128, 128, BK)


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


def cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
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
    tune=None,
    enable_pdl: bool = False,
    *,
    out: torch.Tensor | None = None,
):
    n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
    assert not enable_pdl or out is not None, "PDL requires caller-owned out"
    _check_a_scale_granularity(a_scale, k)
    _check_b_scale_granularity(b_scale, k)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    if tile is None and tune is not False:
        chosen = None
        if MOE_AUTOTUNE_ENABLED():
            with autotune():
                _, tactic = AutoTuner.get().choose_one(
                    "cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4",
                    [_Fc2FinalizeRunner(enable_pdl)],
                    TuningConfig(),
                    [
                        a_q,
                        a_scale,
                        b_q,
                        b_scale,
                        m_indptr,
                        src_token,
                        pair_scales,
                        num_tokens,
                    ],
                )
            if tactic != -1:
                chosen = tactic
        if chosen is not None:
            tile, epi = split_tactic(chosen)
    if tile is None:
        tile = select_tile(
            total_rows=int(a_q.shape[0]),
            n=n,
            num_experts=int(b_q.shape[0]),
            num_sms=grid_x,
        )
    op = CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(n, k, tuple(tile), epi, enable_pdl)
    if out is None:
        out = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=a_q.device)
    if (
        out.shape != (num_tokens, n)
        or out.dtype is not torch.bfloat16
        or out.device != a_q.device
        or not out.is_contiguous()
    ):
        raise ValueError("out does not match the FC2 output contract")
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


class _Fc2FinalizeRunner(TunableRunner):
    def __init__(self, enable_pdl=False):
        self.out_dtype = torch.bfloat16
        self._out = None
        self.enable_pdl = enable_pdl

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
        a_q, a_scale, b_q, b_scale, m_indptr, src_token, pair_scales, num_tokens = (
            inputs
        )
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        return [
            t
            for t in TACTICS
            if CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp.can_implement(
                n=n, k=k, tile=split_tactic(t)[0], epi=split_tactic(t)[1]
            )
        ]

    def alloc_out(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr, src_token, pair_scales, num_tokens = (
            inputs
        )
        n = int(b_q.shape[1])
        return torch.zeros(num_tokens, n, dtype=self.out_dtype, device=a_q.device)

    def launch(self, inputs, out, tactic):
        a_q, a_scale, b_q, b_scale, m_indptr, src_token, pair_scales, num_tokens = (
            inputs
        )
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        props = torch.cuda.get_device_properties(a_q.device)
        grid_x = props.multi_processor_count
        if tactic is None:
            tile, epi = (
                select_tile(
                    total_rows=int(a_q.shape[0]),
                    n=n,
                    num_experts=int(b_q.shape[0]),
                    num_sms=grid_x,
                ),
                DEFAULT_EPI,
            )
        else:
            tile, epi = split_tactic(tactic)
        op = CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(
            n, k, tuple(tile), epi, self.enable_pdl
        )
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
        sm_version = "sm_{}{}".format(props.major, props.minor)
        compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)

    def get_cache_key_extras(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr, src_token, pair_scales, num_tokens = (
            inputs
        )
        return (
            int(a_q.shape[0]),
            int(b_q.shape[0]),
            int(b_q.shape[1]),
            int(b_q.shape[2]) * 2,
            int(num_tokens),
            self.enable_pdl,
            torch.cuda.get_device_capability(),
        )


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
