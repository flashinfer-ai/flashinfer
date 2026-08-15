# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""fp8 x fp8 arm of grouped_mm: compile-time configuration carrier and static feasibility."""

import functools

import cutlass.cute as cute
import torch

from ..kernels.cutedsl.sm120_moe.moe_gemm_fp8 import (
    GRAN_K, GRAN_N, CutedslSm120MoeFp8Grouped, make_args, make_cfg, is_swapab)
from ..kernels.cutedsl.sm120_moe.core.sm120_gemm_builder import (
    Sm120GemmBuilder,
    StoreMethod,
)
from ..kernels.cutedsl.sm120_moe.core._common import ceil_div
from ..cutedsl_sm120_mxfp8_mxfp4._arm import select_plain_bm_64_or_128

FALLBACK_TILE = (128, 128, 128)

# Tie-break when no tactic is named; wg leads on smem-free depth, measured on one card/shape (exp_12).
STORE_ORDER = (StoreMethod.R2G_WG, StoreMethod.STAGED_R2G)


def store_tactics(tile):
    """The store methods a tile admits. A swapped tile stages nothing: its epilogue is the"""
    return (StoreMethod.DIRECT_STG,) if is_swapab(tile) else STORE_ORDER


def resolve_stage(tile, store):
    """Deepest A/B ring this tile affords under one store method."""
    return Sm120GemmBuilder.max_ab_stage(functools.partial(make_cfg, store=store), tuple(tile))


def resolve_stage_store(tile):
    """The default when no tactic is named: the deepest ring any store method affords."""
    best = None
    for store in store_tactics(tile):
        try:
            stage = resolve_stage(tile, store)
        except (AssertionError, ValueError):
            continue
        if best is None or stage > best[0]:
            best = (stage, store)
    if best is None:
        raise ValueError(f"no ab_stage fits smem for tile {tuple(tile)} under any store method")
    return best


class CutedslSm120GroupedFp8Op:
    """Holds the configuration one compiled kernel is specialized on, and answers whether it exists."""

    OUT_DTYPES = (torch.bfloat16,)
    # Measured, not assumed: the tests sweep TACTICS, so every value here has a case behind it.
    BM_SUPPORTED = (128, 64, 32, 8)

    def __init__(self, *, n: int, k: int, tile, out_dtype: torch.dtype, store=None):
        if not self.can_implement(n=n, k=k, tile=tile, out_dtype=out_dtype, store=store):
            raise TypeError(f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                            f"store={store} out_dtype={out_dtype}")
        self.n, self.k, self.tile, self.out_dtype = n, k, tuple(tile), out_dtype
        if store is None:
            self.ab_stage, self.store = resolve_stage_store(self.tile)
        else:
            self.ab_stage, self.store = resolve_stage(self.tile, store), store
        self.cfg = make_cfg(self.tile, self.ab_stage, store=self.store)

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        """StagedR2GStoreConfig is built with bf16; the MMA operand pair is fixed e4m3 x e4m3."""
        return out_dtype in CutedslSm120GroupedFp8Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        """bn and bk are the scale granularity itself -- the kernel asserts both as identities."""
        bm, bn, bk = tile
        return bm in CutedslSm120GroupedFp8Op.BM_SUPPORTED and bn == GRAN_N and bk == GRAN_K

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        """K rides whole k-tiles; N only needs the TMA box, since the epilogue predicates its residue."""
        return n > 0 and k > 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(cls, tile, store=None) -> bool:
        """Build the configuration and the kernel object, which is where the rest of the invariants live."""
        try:
            if store is None:
                stage, store = resolve_stage_store(tuple(tile))
            else:
                stage = resolve_stage(tuple(tile), store)
            CutedslSm120MoeFp8Grouped(make_cfg(tuple(tile), stage, store=store), 1)
        except (AssertionError, ValueError):
            return False
        return True

    @classmethod
    def can_implement(cls, *, n: int, k: int, tile, out_dtype, store=None) -> bool:
        return (cls.is_valid_dtypes(out_dtype) and cls.is_valid_tile(tile)
                and cls.is_valid_alignment(n, k, tile) and cls.is_constructible(tile, store))

    def build(self, grid_x: int) -> CutedslSm120MoeFp8Grouped:
        return CutedslSm120MoeFp8Grouped(self.cfg, grid_x)


# The verified domain, expressed once: the tests sweep exactly what can_implement accepts.
TACTICS = tuple((bm, GRAN_N, store.name)
                for bm in CutedslSm120GroupedFp8Op.BM_SUPPORTED
                for store in store_tactics((bm, GRAN_N, GRAN_K)))


def split_tactic(tactic):
    """A tactic back into the pair the entry point takes."""
    bm, bn, store = tactic
    return (bm, bn, GRAN_K), StoreMethod[store]

# Building the configuration walks the whole smem cost model, so it is cached per compiled kernel.
@functools.lru_cache(maxsize=None)
def _op(n: int, k: int, tile, out_dtype, store=None) -> CutedslSm120GroupedFp8Op:
    return CutedslSm120GroupedFp8Op(n=n, k=k, tile=tile, out_dtype=out_dtype, store=store)


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op: CutedslSm120GroupedFp8Op, grid_x: int, sm_version: str):
    """One compiled kernel per (problem-invariant, tile, device) key; ``sample_args`` only shape the trace."""
    key = (op.n, op.k, op.tile, op.ab_stage, op.store, op.out_dtype, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    """Pick bm from the problem shape. Ported from flashinfer's C++ runner (da1ac9fe), which chose it"""
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    smallest = min(CutedslSm120GroupedFp8Op.BM_SUPPORTED)

    def tile_count(bm: int, bn: int) -> int:
        return num_experts * ceil_div(m_per_expert, bm) * ceil_div(n, bn)

    if n % 128 == 0 and (m_per_expert <= 8
                         or (tile_count(32, 128) < num_sms // 2
                             and tile_count(8, 128) <= num_sms)):
        bm = smallest                      # upstream: 8 (swap-AB)
    elif m_per_expert <= 32:
        bm = 32
    elif k <= 2048:
        bm = 64 if m_per_expert < 192 else 128
    else:
        bm = select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms)
    return (max(bm, smallest), GRAN_N, GRAN_K)
