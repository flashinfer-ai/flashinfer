# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""act-MXFP8 x weight-MXFP4 arm: compile-time configuration carrier, tactic table and static feasibility."""

import functools

import cutlass.cute as cute
import torch

from ..kernels.cutedsl.sm120_moe.core._common import ceil_div
from ..kernels.cutedsl.sm120_moe.moe_gemm_mxfp8_mxfp4 import (
    GRANK_A, GRANK_B, CutedslSm120MoeMxfp8Mxfp4Grouped, is_swapab, make_args, make_cfg)
from ..kernels.cutedsl.sm120_moe.core.sm120_blockscaled_layout import Sm120SfConfigMxfp8Mxfp4
from ..kernels.cutedsl.sm120_moe.core.sm120_gemm_builder import Sm120GemmBuilder, StoreMethod

PLAIN_TILE_OVERHEAD = 48


def select_plain_bm_64_or_128(m_per_expert: int, n: int, num_experts: int, num_sms: int) -> int:
    """Cost is waves x (bm + overhead), not tile count."""
    def cost(bm: int) -> int:
        num_tiles = num_experts * ceil_div(m_per_expert, bm) * ceil_div(n, 128)
        return ceil_div(num_tiles, num_sms) * (bm + PLAIN_TILE_OVERHEAD)

    return 64 if cost(64) < cost(128) else 128

FALLBACK_TILE = (128, 128, 128)

# Tie-break when no tactic is named; staged leads here across two cases, widening with M (exp_15).
STORE_ORDER = (StoreMethod.STAGED_R2G, StoreMethod.R2G_WG)


def store_tactics(tile):
    """The store methods a tile admits. A swapped tile stages nothing: its epilogue is the"""
    return (StoreMethod.DIRECT_STG,) if is_swapab(tile) else STORE_ORDER


def resolve_stage(tile, store, grank_a=GRANK_A):
    """Deepest A/B ring this tile affords under one store method."""
    return Sm120GemmBuilder.max_ab_stage(
        functools.partial(make_cfg, store=store, grank_a=grank_a), tuple(tile))


def resolve_stage_store(tile, grank_a=GRANK_A):
    """The default when no tactic is named: the deepest ring any store method affords."""
    best = None
    for store in store_tactics(tile):
        try:
            stage = resolve_stage(tile, store, grank_a)
        except (AssertionError, ValueError):
            continue
        if best is None or stage > best[0]:
            best = (stage, store)
    if best is None:
        raise ValueError(f'no store method yields a viable ab_stage for tile {tuple(tile)}')
    return best


class CutedslSm120GroupedMxfp8Mxfp4Op:
    """Holds the configuration one compiled kernel is specialized on, and answers whether it exists."""

    OUT_DTYPES = (torch.bfloat16,)
    # Measured, not assumed: exp_09 swept these bit-exact; an unmeasured tile is a silent wrong answer.
    BM_SUPPORTED = (128, 64, 32, 8)
    BN_SUPPORTED = (128,)

    def __init__(self, *, n: int, k: int, tile, out_dtype: torch.dtype, store=None,
                 grank_a: int = GRANK_A):
        if not self.can_implement(n=n, k=k, tile=tile, out_dtype=out_dtype, store=store,
                                  grank_a=grank_a):
            raise TypeError(f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                            f"out_dtype={out_dtype} store={store} grank_a={grank_a}")
        self.n, self.k, self.tile, self.out_dtype = n, k, tuple(tile), out_dtype
        self.grank_a = grank_a
        if store is None:
            self.ab_stage, self.store = resolve_stage_store(self.tile, grank_a)
        else:
            self.ab_stage, self.store = resolve_stage(self.tile, store, grank_a), store
        self.cfg = make_cfg(self.tile, self.ab_stage, store=self.store, grank_a=grank_a)

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        """R2GWgStoreConfig is built with bf16; the MMA operand pair is fixed act-e4m3 x weight-e2m1."""
        return out_dtype in CutedslSm120GroupedMxfp8Mxfp4Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        """bk is the SFB pack width, not a free axis: one int32 pack must cover exactly one k-tile."""
        bm, bn, bk = tile
        return (bm in CutedslSm120GroupedMxfp8Mxfp4Op.BM_SUPPORTED
                and bn in CutedslSm120GroupedMxfp8Mxfp4Op.BN_SUPPORTED
                and bk == GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        """K rides whole k-tiles; N only needs the TMA box, since the epilogue predicates its residue."""
        return n > 0 and k > 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(cls, tile, store=None, grank_a: int = GRANK_A) -> bool:
        """Build the configuration and the kernel object, which is where the rest of the invariants live."""
        try:
            if store is None:
                stage, store = resolve_stage_store(tuple(tile), grank_a)
            else:
                stage = resolve_stage(tuple(tile), store, grank_a)
            CutedslSm120MoeMxfp8Mxfp4Grouped(
                make_cfg(tuple(tile), stage, store=store, grank_a=grank_a), 1)
        except (AssertionError, ValueError):
            return False
        return True

    @staticmethod
    def is_valid_granularity(grank_a: int) -> bool:
        """A-side SF granularity: the two the same-family entry admits, nesting with the fixed B side."""
        return grank_a in (128, 32)

    @classmethod
    def can_implement(cls, *, n: int, k: int, tile, out_dtype, store=None,
                      grank_a: int = GRANK_A) -> bool:
        return (cls.is_valid_dtypes(out_dtype) and cls.is_valid_tile(tile)
                and cls.is_valid_granularity(grank_a)
                and cls.is_valid_alignment(n, k, tile)
                and cls.is_constructible(tile, store, grank_a))

    def build(self, grid_x: int) -> CutedslSm120MoeMxfp8Mxfp4Grouped:
        return CutedslSm120MoeMxfp8Mxfp4Grouped(self.cfg, grid_x)


# The verified domain, expressed once: the tests sweep exactly what can_implement accepts.
TACTICS = tuple((bm, bn, store.name)
                for bm in CutedslSm120GroupedMxfp8Mxfp4Op.BM_SUPPORTED
                for bn in CutedslSm120GroupedMxfp8Mxfp4Op.BN_SUPPORTED
                for store in store_tactics((bm, bn, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)))


def split_tactic(tactic):
    """A tactic back into the pair the entry point takes."""
    bm, bn, store = tactic
    return (bm, bn, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF), StoreMethod[store]

# Building the configuration walks the whole smem cost model, so it is cached per compiled kernel.
@functools.lru_cache(maxsize=None)
def _op(n: int, k: int, tile, out_dtype, store=None,
        grank_a: int = GRANK_A) -> CutedslSm120GroupedMxfp8Mxfp4Op:
    return CutedslSm120GroupedMxfp8Mxfp4Op(n=n, k=k, tile=tile, out_dtype=out_dtype, store=store,
                                           grank_a=grank_a)


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op: CutedslSm120GroupedMxfp8Mxfp4Op, grid_x: int, sm_version: str):
    """One compiled kernel per (problem-invariant, tile, device) key; ``sample_args`` only shape the trace."""
    key = (op.n, op.k, op.tile, op.ab_stage, op.store, op.out_dtype, op.grank_a, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    """Pick bm from the problem shape. Ported from flashinfer's C++ runner (da1ac9fe)."""
    m_per_expert = total_rows // num_experts if num_experts > 0 else 0
    smallest = min(CutedslSm120GroupedMxfp8Mxfp4Op.BM_SUPPORTED)
    bn = CutedslSm120GroupedMxfp8Mxfp4Op.BN_SUPPORTED[0]

    if m_per_expert <= 12:
        bm = smallest                      # 8: the swap-AB tile
    elif m_per_expert <= 32:
        bm = 32
    else:
        # no k<=2048 shortcut here: measured 41% slower than the plain choice on this kernel
        bm = select_plain_bm_64_or_128(m_per_expert, n, num_experts, num_sms)
    return (max(bm, smallest), bn, GRANK_B * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)


def _check_a_scale_granularity(a_scale, k: int, grank_a: int) -> None:
    """The kernel rebuilds a_scale's K extent from config and never reads it, so mismatch is silent."""
    want = ceil_div(k, grank_a * Sm120SfConfigMxfp8Mxfp4.PACK_NSF)
    got = int(a_scale.shape[0])
    if got != want:
        raise ValueError(f"a_scale K extent {got} != {want} implied by k={k}, grank_a={grank_a}")
