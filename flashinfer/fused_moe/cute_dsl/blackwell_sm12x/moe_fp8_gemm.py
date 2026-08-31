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
"""fp8 x fp8 moe_gemm on CuteDSL: what one compiled kernel is specialized on, and the entry."""

import functools
import os

import cutlass.cute as cute
import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from cutlass.base_dsl.common import DSLUserCodeError

from ....utils import ceil_div
from ._moe_utils.moe_epilogue import EpiMethod
from ._moe_utils.moe_kernel_builder import Sm120GemmBuilder
from .kernel_moe_fp8_gemm import (
    GRAN_K,
    GRAN_N,
    CuteDslSm120MoeFp8Grouped,
    is_swapab,
    make_args,
    make_cfg,
)
from ._moe_utils.heuristic import select_plain_bm_64_or_128

FALLBACK_TILE = (128, 128, 128)

EPI_ORDER = (EpiMethod.R2G_WG, EpiMethod.STAGED_R2G)


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
            f"no ab_stage fits smem for tile {tuple(tile)} under any store method"
        )
    return best


class CuteDslSm120GroupedFp8Op:
    OUT_DTYPES = (torch.bfloat16,)
    TILES = ((128, GRAN_N), (64, GRAN_N), (32, GRAN_N), (8, GRAN_N))

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
                f"epi={epi} out_dtype={out_dtype}"
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
        return out_dtype in CuteDslSm120GroupedFp8Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (bm, bn) in CuteDslSm120GroupedFp8Op.TILES and bk == GRAN_K

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
            CuteDslSm120MoeFp8Grouped(make_cfg(tuple(tile), stage, epi=epi), 1)
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

    def build(self, grid_x: int) -> CuteDslSm120MoeFp8Grouped:
        return CuteDslSm120MoeFp8Grouped(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedFp8Op.TILES
    for epi in epi_tactics((bm, bn, GRAN_K))
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, GRAN_K), epi


@functools.lru_cache(maxsize=None)
def _op(
    n: int, k: int, tile, out_dtype, epi=None, enable_pdl=False
) -> CuteDslSm120GroupedFp8Op:
    return CuteDslSm120GroupedFp8Op(
        n=n, k=k, tile=tile, out_dtype=out_dtype, epi=epi, enable_pdl=enable_pdl
    )


_COMPILED: dict = {}


def compiled_kernel(
    sample_args, *, op: CuteDslSm120GroupedFp8Op, grid_x: int, sm_version: str
):
    key = (op.tactic, op.out_dtype, op.enable_pdl, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    by_bm = dict(CuteDslSm120GroupedFp8Op.TILES)
    smallest = min(by_bm)

    def tile_count(bm: int, bn: int) -> int:
        return num_experts * ceil_div(m_per_expert, bm) * ceil_div(n, bn)

    if n % 128 == 0 and (
        m_per_expert <= 8
        or (tile_count(32, 128) < num_sms // 2 and tile_count(8, 128) <= num_sms)
    ):
        bm = smallest
    elif m_per_expert <= 32:
        bm = 32
    elif k <= 2048:
        bm = 64 if m_per_expert < 192 else 128
    else:
        bm = select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms)
    bm = max(bm, smallest)
    return (bm, by_bm[bm], GRAN_K)


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


def cute_dsl_sm12x_moe_gemm_fp8(
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
    n, k = int(b_q.shape[1]), int(b_q.shape[2])
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
                    "cute_dsl_sm12x_moe_gemm_fp8",
                    [_Fp8Runner(out_dtype, enable_pdl)],
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


class _Fp8Runner(TunableRunner):
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
        n, k = int(b_q.shape[1]), int(b_q.shape[2])
        return [
            t
            for t in TACTICS
            if CuteDslSm120GroupedFp8Op.can_implement(
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
        n, k = int(b_q.shape[1]), int(b_q.shape[2])
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
            int(b_q.shape[2]),
            str(self.out_dtype),
            self.enable_pdl,
            torch.cuda.get_device_capability(),
        )


def should_autotune(a_q, b_q) -> bool:
    num_experts, physical_n, k = int(b_q.shape[0]), int(b_q.shape[1]), int(b_q.shape[2])
    m_per_expert = int(a_q.shape[0]) // num_experts if num_experts > 0 else 0
    return num_experts > 0 and m_per_expert > 0 and k > 2048 and physical_n % 32 == 0


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
