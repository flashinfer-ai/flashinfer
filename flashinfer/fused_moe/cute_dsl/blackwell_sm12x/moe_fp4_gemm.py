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
"""NVFP4 token-packed grouped GEMM entry and static feasibility."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.common import DSLUserCodeError

from ....utils import ceil_div
from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import Sm12xGemmConfig
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset
from .kernel_moe_fp4_gemm import (
    CuteDslSm120MoeNvfp4Grouped,
    is_swapab,
    make_args,
    make_cfg,
)

FALLBACK_TILE = (128, 128, 128)
PLAIN_TILE_K = 128
EPI_ORDER = (EpiMethod.R2G_WG,)


def epi_tactics(tile):
    if is_swapab(tile):
        return (EpiMethod.DIRECT_STG,)
    return tuple(epi for epi in EPI_ORDER if EPI_CONFIGS[epi].supports_tile(tile))


def resolve_stage(tile, epi):
    return Sm12xGemmConfig.max_ab_stage(
        functools.partial(make_cfg, epi=epi), tuple(tile)
    )


def resolve_stage_store(tile, output_n):
    best = None
    for epi in epi_tactics(tile):
        if not EPI_CONFIGS[epi].can_implement(tile, output_n, cutlass.BFloat16.width):
            continue
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


class CuteDslSm120GroupedNvfp4Op:
    OUT_DTYPES = (torch.bfloat16,)
    TILES = ((128, 128), (64, 128), (32, 128), (8, 128))

    def __init__(self, *, n: int, k: int, tile, out_dtype: torch.dtype, epi=None):
        if not self.can_implement(n=n, k=k, tile=tile, out_dtype=out_dtype, epi=epi):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} "
                f"tile={tuple(tile)} out_dtype={out_dtype} epi={epi}"
            )
        self.n, self.k, self.tile, self.out_dtype = n, k, tuple(tile), out_dtype
        if epi is None:
            self.ab_stage, self.epi = resolve_stage_store(self.tile, n)
        else:
            self.ab_stage, self.epi = resolve_stage(self.tile, epi), epi
        self.tactic = (self.tile[0], self.tile[1], self.epi)
        self.cfg = make_cfg(self.tile, self.ab_stage, epi=self.epi)

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedNvfp4Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (bm, bn) in CuteDslSm120GroupedNvfp4Op.TILES and bk == PLAIN_TILE_K

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return n > 0 and k > 0 and n % tile[1] == 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(cls, tile, epi=None) -> bool:
        if epi is None:
            return any(
                cls.is_constructible(tile, candidate) for candidate in epi_tactics(tile)
            )
        if epi not in epi_tactics(tile):
            return False
        try:
            if not EPI_CONFIGS[epi].supports_tile(tile):
                return False
            stage = resolve_stage(tuple(tile), epi)
            CuteDslSm120MoeNvfp4Grouped(make_cfg(tuple(tile), stage, epi=epi), 1)
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(cls, *, n: int, k: int, tile, out_dtype, epi=None) -> bool:
        candidates = epi_tactics(tile) if epi is None else (epi,)
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and cls.is_valid_alignment(n, k, tile)
            and any(
                candidate in epi_tactics(tile)
                and EPI_CONFIGS[candidate].can_implement(
                    tile, n, cutlass.BFloat16.width
                )
                and cls.is_constructible(tile, candidate)
                for candidate in candidates
            )
        )

    def build(self, grid_x: int) -> CuteDslSm120MoeNvfp4Grouped:
        return CuteDslSm120MoeNvfp4Grouped(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedNvfp4Op.TILES
    for epi in epi_tactics((bm, bn, PLAIN_TILE_K))
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, PLAIN_TILE_K), epi


@functools.lru_cache(maxsize=None)
def _op(n: int, k: int, tile, out_dtype, epi=None) -> CuteDslSm120GroupedNvfp4Op:
    return CuteDslSm120GroupedNvfp4Op(n=n, k=k, tile=tile, out_dtype=out_dtype, epi=epi)


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op, grid_x: int, sm_version: str):
    key = (op.tactic, op.out_dtype, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def _validate_inputs(a_q, a_scale, b_q, b_scale, m_indptr, alpha):
    tensors = (a_q, a_scale, b_q, b_scale, m_indptr, alpha)
    if any(t.device != a_q.device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if a_q.dtype != torch.uint8 or b_q.dtype != torch.uint8:
        raise TypeError("a_q and b_q must be uint8-packed FP4")
    if a_q.ndim != 2 or b_q.ndim != 3:
        raise ValueError("a_q must be 2D and b_q must be 3D")
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("a_scale and b_scale must be E4M3 bytes from fp4_quantize")
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    if alpha.dtype != torch.float32 or tuple(alpha.shape) != (1,):
        raise TypeError("alpha must be float32[1]")
    if int(m_indptr.shape[0]) != int(b_q.shape[0]) + 1:
        raise ValueError("m_indptr length must equal num_experts + 1")
    if int(a_q.shape[1]) != int(b_q.shape[2]):
        raise ValueError("a_q and b_q K extents must match")
    m, packed_k = int(a_q.shape[0]), int(a_q.shape[1])
    e, n = int(b_q.shape[0]), int(b_q.shape[1])
    k = packed_k * 2
    want_a_scale = (compute_padded_offset(m, e, 128), ceil_div(k, 16))
    want_b_scale = (e * n, ceil_div(k, 16))
    if tuple(a_scale.shape) != want_a_scale:
        raise ValueError(f"a_scale shape must be {want_a_scale}")
    if tuple(b_scale.shape) != want_b_scale:
        raise ValueError(f"b_scale shape must be {want_b_scale}")


def cute_dsl_sm12x_moe_gemm_nvfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    alpha,
    out_dtype: torch.dtype = torch.bfloat16,
    tile=None,
    epi=None,
) -> torch.Tensor:
    _validate_inputs(a_q, a_scale, b_q, b_scale, m_indptr, alpha)
    n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
    tile = FALLBACK_TILE if tile is None else tuple(tile)
    op = _op(n, k, tile, out_dtype, epi)
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    out = torch.zeros(int(a_q.shape[0]), n, dtype=out_dtype, device=a_q.device)
    args = make_args(
        a_q, a_scale, b_q, b_scale, out, m_indptr, alpha, op.cfg.epi, op.cfg.TILE
    )
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return out
