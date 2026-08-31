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
"""act-MXFP8 x weight-MXFP4 arm of grouped_mm: compile-time configuration carrier and static feasibility."""

import functools
import os

import cutlass.cute as cute
import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from cutlass.base_dsl.common import DSLUserCodeError

from ....utils import ceil_div

from .kernel_moe_mxfp8_mxfp4_gemm import (
    GRANK_A,
    GRANK_B,
    CuteDslSm120MoeMxfp8Mxfp4Grouped,
    is_swapab,
    make_args,
    make_cfg,
)
from ._moe_utils.sm12x_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ._moe_utils.moe_epilogue import EpiMethod
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder
from ._moe_utils.heuristic import select_plain_bm_64_or_128

FALLBACK_TILE = (128, 128, 128)

EPI_ORDER = (EpiMethod.STAGED_R2G, EpiMethod.R2G_WG)


def epi_tactics(tile):
    return (EpiMethod.DIRECT_STG,) if is_swapab(tile) else EPI_ORDER


def resolve_stage(tile, epi):
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, epi=epi), tuple(tile)
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
        raise ValueError(
            f"no store method yields a viable ab_stage for tile {tuple(tile)}"
        )
    return best


class CuteDslSm120GroupedMxfp8Mxfp4Op:
    OUT_DTYPES = (torch.bfloat16,)
    TILES = ((128, 128), (64, 128), (32, 128), (8, 128))

    def __init__(
        self,
        *,
        n: int,
        k: int,
        tile,
        out_dtype: torch.dtype,
        epi=None,
        enable_pdl=False,
    ):
        if not self.can_implement(n=n, k=k, tile=tile, out_dtype=out_dtype, epi=epi):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                f"out_dtype={out_dtype} epi={epi}"
            )
        self.n, self.k, self.tile, self.out_dtype = n, k, tuple(tile), out_dtype
        self.enable_pdl = enable_pdl
        if epi is None:
            self.ab_stage, self.epi = resolve_stage_store(self.tile)
        else:
            self.ab_stage, self.epi = resolve_stage(self.tile, epi), epi
        self.tactic = (self.tile[0], self.tile[1], self.epi)
        self.cfg = make_cfg(
            self.tile, self.ab_stage, epi=self.epi, enable_pdl=enable_pdl
        )

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedMxfp8Mxfp4Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (
            (bm, bn) in CuteDslSm120GroupedMxfp8Mxfp4Op.TILES
            and bk == GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF
        )

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return n > 0 and k > 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(cls, tile, epi=None) -> bool:
        try:
            if epi is None:
                stage, epi = resolve_stage_store(tuple(tile))
            else:
                stage = resolve_stage(tuple(tile), epi)
            CuteDslSm120MoeMxfp8Mxfp4Grouped(make_cfg(tuple(tile), stage, epi=epi), 1)
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(cls, *, n: int, k: int, tile, out_dtype, epi=None) -> bool:
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and cls.is_valid_alignment(n, k, tile)
            and cls.is_constructible(tile, epi)
        )

    def build(self, grid_x: int) -> CuteDslSm120MoeMxfp8Mxfp4Grouped:
        return CuteDslSm120MoeMxfp8Mxfp4Grouped(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedMxfp8Mxfp4Op.TILES
    for epi in epi_tactics((bm, bn, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF))
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF), epi


@functools.lru_cache(maxsize=None)
def _op(
    n: int, k: int, tile, out_dtype, epi=None, enable_pdl=False
) -> CuteDslSm120GroupedMxfp8Mxfp4Op:
    return CuteDslSm120GroupedMxfp8Mxfp4Op(
        n=n, k=k, tile=tile, out_dtype=out_dtype, epi=epi, enable_pdl=enable_pdl
    )


_COMPILED: dict = {}


def compiled_kernel(
    sample_args, *, op: CuteDslSm120GroupedMxfp8Mxfp4Op, grid_x: int, sm_version: str
):
    key = (op.tactic, op.out_dtype, op.enable_pdl, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    by_bm = dict(CuteDslSm120GroupedMxfp8Mxfp4Op.TILES)
    smallest = min(by_bm)

    if m_per_expert <= 12:
        bm = smallest
    elif m_per_expert <= 32:
        bm = 32
    else:
        bm = select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms)
    bm = max(bm, smallest)
    return (bm, by_bm[bm], GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)


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


def cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4(
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
) -> torch.Tensor:
    n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
    _check_a_scale_granularity(a_scale, k)
    _check_b_scale_granularity(b_scale, k)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    if tile is None and tune is not False:
        gate = should_autotune(a_q, b_q) if tune is None else True
        chosen = None
        if gate and MOE_AUTOTUNE_ENABLED():
            with autotune():
                _, tactic = AutoTuner.get().choose_one(
                    "cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4",
                    [_Mxfp8Mxfp4Runner(out_dtype, enable_pdl)],
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
    op = _op(n, k, tuple(tile), out_dtype, epi, enable_pdl)
    out = torch.zeros(int(a_q.shape[0]), n, dtype=out_dtype, device=a_q.device)
    args = make_args(a_q, a_scale, b_q, b_scale, out, m_indptr.to(torch.int32))
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return out


class _Mxfp8Mxfp4Runner(TunableRunner):
    def __init__(self, out_dtype, enable_pdl=False):
        self.out_dtype = out_dtype
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
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        return [
            t
            for t in TACTICS
            if CuteDslSm120GroupedMxfp8Mxfp4Op.can_implement(
                n=n,
                k=k,
                tile=split_tactic(t)[0],
                out_dtype=self.out_dtype,
                epi=split_tactic(t)[1],
            )
        ]

    def alloc_out(self, inputs):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n = int(b_q.shape[1])
        return torch.zeros(
            int(a_q.shape[0]), n, dtype=self.out_dtype, device=a_q.device
        )

    def launch(self, inputs, out, tactic):
        a_q, a_scale, b_q, b_scale, m_indptr = inputs
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        props = torch.cuda.get_device_properties(a_q.device)
        grid_x = props.multi_processor_count
        if tactic is None:
            tile, epi = (
                select_tile(
                    total_rows=int(a_q.shape[0]),
                    n=n,
                    k=k,
                    num_experts=int(b_q.shape[0]),
                    num_sms=grid_x,
                ),
                None,
            )
        else:
            tile, epi = split_tactic(tactic)
        op = _op(n, k, tuple(tile), self.out_dtype, epi, self.enable_pdl)
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
            self.enable_pdl,
            torch.cuda.get_device_capability(),
        )


def should_autotune(a_q, b_q) -> bool:
    num_experts, physical_n, k = (
        int(b_q.shape[0]),
        int(b_q.shape[1]),
        int(b_q.shape[2]) * 2,
    )
    m_per_expert = int(a_q.shape[0]) // num_experts if num_experts > 0 else 0
    return num_experts > 0 and m_per_expert > 0 and k > 2048 and physical_n % 32 == 0


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
