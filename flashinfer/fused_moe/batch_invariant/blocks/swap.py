# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The swapped body of the MoE classes: ``C^T`` per expert, with the expert's weight rows on the MMA's M side (a 128- or
64-row tile streamed by TMA with the expert as the batch coordinate) and the group's tokens on the N side as a **narrow
token tile** (8, 16 or 32 rows: the swapped records of :mod:`..schedule`; ``kind::f8f6f4`` takes 16 or 32), read as a TMA
box from the dense permuted activations at the group's row (``x_perm`` for GEMM1, the dense intermediate for GEMM2) -- the
same dense rows the grouped body reads, so no second packing. One body for the bf16, the fp8 per-tensor, the NVFP4 and
the MXFP8 classes, as :mod:`.kernel`: the operand block (:mod:`.operands`, keyed by the chain's operand format) and the epilogue
block (the chain) are resolved at trace time. Rows past the group's end (the next group's rows or the buffer's tail) are
computed and never stored: the MMA is token-local on N. One TMA pipeline carries the weight tile and the token box of one
k-tile per stage, two swizzle atoms of K per stage (the grouped records' step), the dense ``mma_loop`` folds them, and the
transposed group-masked epilogue stores token ``j`` of the tile as output row ``row0 + j``.

The block-scaled formats add the scale factors: the weight tile's 128x4 scale-factor atom streams with the weights on the TMA
(the expert as the batch coordinate), the token tile's scale-factor rows are gathered by one ``cp.async`` warp from the padded
positions of the dense packing into rows ``0 .. token_tile - 1`` of the stage's atom (the narrow swapped records read the
atom from its first rows: no sub-tile addressing), and the MMA loop copies both sets to TMEM per k-tile; the group tables
come in one scan on the scale-factor warp while the epilogue warps issue the TMEM allocation, and the epilogue's per-token
scale is per column: ``s_j = mul(alpha_e, pts[pad(row0 + j)])``.

The same class: the tcgen05 step is symmetric in its operands -- ``D[m, n] += sum_k A[m, k] B[n, k]`` folds the same K
sequence per output element whichever operand carries the token -- so every output element's fold is the dense class's
fold on the same operands (the per-expert agreement with the grouped body is the test). The gated chain's pairing moves
from two registers of one thread to two lanes of one warp (rows ``16q + r`` (up) and ``16q + 8 + r`` (gate) of the
interleaved weight are TMEM lanes ``l`` and ``l + 8``: a ``shfl.sync.down 8`` of the gate accumulators); the term per
element is unchanged. Everything else is placement: which CTA owns which (expert, token tile, weight tile), the TMA
pipeline, the store pattern.

Per tile the operand bytes are the weight tile plus ``token_tile`` rows of tokens, so at a few rows per active expert the
GEMM traffic is the weights read once; the dispatcher ranks the swapped records with the grouped ones (:func:`..schedule.rank_grouped`).

Operands: ``rows (R, K, 1)`` K-major (the dense permuted rows; ``R >= Tk``: with ``R >= Tk + token_tile`` the last tiles'
boxes stay inside the buffer, else they are zero-filled past it; the byte-packed ``(R, K / 2, 1)`` of the 4-bit formats),
``w (N, K, E)`` K-major, ``out (Tk, Nout)`` row-major (``Nout = N`` plain and ungated, ``N / 2`` gated; bf16 / f16 / f32,
the uint8 bytes of e4m3 for the fp8 classes' GEMM1, the packed codes of the static NVFP4 class's GEMM1), ``counts (E,)``
int32; for a chain with scale operands ``alpha (E,)``, ``sa (1,)``, ``si (1,)`` fp32 as for the grouped body
(:meth:`SwappedGroupedGemm.with_scales`); for the block-scaled chains ``sfa`` the flat scale factors of the rows at the padded
positions, ``sfw`` the experts' scale buffers, ``alpha (E,)``, ``pts`` (the per-token scales at the padded positions, a
multiple of 128 entries) and the static class's ``out_sf`` / ``aux`` (:meth:`SwappedGroupedGemm.block_scaled` /
:meth:`SwappedGroupedGemm.block_scaled_static`); the MX chains carry no further operand
(:meth:`SwappedGroupedGemm.block_scaled_plain`; the fused requantization's codes and scale bytes through
:meth:`SwappedGroupedGemm.block_scaled_quant`, the stage two atoms of K wide when the class's rule says so).
The gather form (``gather=True``, :meth:`SwappedGroupedGemm.gathered`): the token tile's rows gathered by the TMA's gather4
mode from the unpermuted ``x (T, K)`` through the plan's ``src`` (row ``p`` reads ``x[src[p] // k_div]``; rows past the
group's end repeat its last row and are never stored), one swizzle atom of K per stage (the gather4 box is the atom): GEMM1
without the gather launch and the permuted tensor, the same bytes in the stage, the same fold. Its predecessor is the plan's
launch, so every warp waits for it before the group tables.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, List, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Int32, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cvt_fp8 import (
    cvt_f32_to_e4m3,
    cvt_f32x2_to_fp8x2,
    cvt_f32x8_to_e2m1x8,
)
from ....gemm.batch_invariant.blocks.blockscaled import s2t_copy_and_partition
from ....gemm.batch_invariant.blocks.mainloop import mma_loop
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE, BAR_TMEM_ALLOC
from ....gemm.batch_invariant.generator import TILES_N_NARROW, Knobs
from ....gemm.batch_invariant.spec import MMA_K_OF, SF_ATOM_KBLOCKS
from .. import spec_nvfp4 as Q
from .epilogue import (
    BAR_EPI_JOIN,
    GROUP,
    GroupedChain,
    GroupedEpilogueBlock,
    MoeF32,
    fragment_offsets,
    sf_byte_offset,
)
from .operands import (
    PipelineCpAsyncUmma,
    operands_for,
    swap_sf_producer_loop,
    xch_smem_bytes,
)
from .scheduler import (
    MAX_E,
    GroupScheduler,
    build_group_tables,
    build_group_tables_dual,
)
from .. import pdl

EPILOGUE_WARPS = (
    0,
    1,
    2,
    3,
)  # one warpgroup: the weight rows of the tile (one per thread, or the (r, r + 8) pair of the 64-row tile)
MMA_WARP = 4
TMA_WARP = 5
THREADS = 32 * (TMA_WARP + 1)
#: Token columns of one epilogue subtile: the TMEM load's width per warpgroup.
EPI_SUBTILE_N = 8
#: Epilogue warpgroups at most (each takes one subtile of the token tile).
EPI_WARPGROUPS_MAX = 4
#: The in-kernel plan of the decode form (the scale-factor warp ranks the routing's slots from the ids itself, no plan launch):
#: the slots one warp holds in registers and the experts its membership masks cover.
PLAN_MAX_SLOTS = 128
PLAN_MAX_E = 256
PLAN_SMEM_BYTES = 4 * (MAX_E + 4 * PLAN_MAX_E + PLAN_MAX_SLOTS)


def epi_warpgroups(token_tile: int, chain=None) -> int:
    """The swapped epilogue's warpgroups at a token tile: one per ``EPI_SUBTILE_N`` columns, at most ``EPI_WARPGROUPS_MAX``
    (a placement: every output element's chain is computed by one thread from the same accumulator value, in the same fp32
    sequence, whichever warpgroup owns its subtile), for the shared epilogue and for a chain's own swapped loop alike (the
    quantising MX chains' block reduction spans a warp's lanes, orthogonal to the token columns the warpgroups split)."""
    return max(1, min(EPI_WARPGROUPS_MAX, token_tile // EPI_SUBTILE_N))


def epilogue_warps(token_tile: int, chain=None):
    return tuple(range(4 * epi_warpgroups(token_tile, chain)))


WEIGHT_TILE = (
    128  # weight rows per tile (the MMA's M: the 1-CTA instruction's 128 rows)
)
WEIGHT_TILES = (
    WEIGHT_TILE,
    64,
)  # the weight tiles: 128 rows, or 64 (the two-row TMEM layout: twice the CTAs of a small weight)
#: The narrowest token tile of each operand format's instruction: ``kind::f16`` admits N >= 8, ``kind::f8f6f4`` N >= 16, the
#: block-scaled ``kind::mxf4nvf4`` / ``kind::mxf8f6f4`` N >= 8.
MIN_TOKEN_TILE: Dict[str, int] = {"bf16": 8, "e4m3": 16}
#: The 64-row weight tile takes token tiles from 16 (a 64 x 8 tile holds too little work per CTA).
MIN_TOKEN_TILE_64 = 16


def weight_tiles_of(in_fmt: str = "bf16") -> Tuple[int, ...]:
    """The weight tiles of a format's swapped records: 128 or 64 rows for the dense formats; 128 for the block-scaled ones
    (the weight tile is one scale-factor atom)."""
    return WEIGHT_TILES if in_fmt in MIN_TOKEN_TILE else (WEIGHT_TILE,)


@functools.lru_cache(maxsize=None)
def token_tiles_of(weight_tile: int, in_fmt: str = "bf16") -> Tuple[int, ...]:
    """The token tiles a weight tile takes for an operand format: the narrow dense tiles from the format's narrowest, the
    64-row tile from 16 (memoised: the record predicate runs at every launch)."""
    lo = max(
        MIN_TOKEN_TILE.get(in_fmt, 8), MIN_TOKEN_TILE_64 if weight_tile == 64 else 0
    )
    return tuple(t for t in TILES_N_NARROW if t >= lo)


TOKEN_TILES = TILES_N_NARROW  # every swapped record's token tile is one of the narrow dense tiles (8, 16, 32)


def swap_knobs(
    token_tile: int = None,
    raster: str = "m",
    weight_tile: int = WEIGHT_TILE,
    in_fmt: str = "bf16",
) -> Knobs:
    """The classes' swapped record with ``token_tile`` tokens (the format's narrowest by default) on the MMA's N side and
    ``weight_tile`` (128 or 64) weight rows on its M side, 1-CTA, no cluster."""
    tiles = token_tiles_of(weight_tile, in_fmt)
    token_tile = tiles[0] if token_tile is None else token_tile
    assert token_tile in tiles, (token_tile, weight_tile, in_fmt)
    return Knobs(
        (weight_tile, token_tile), (1, 1), False, True, raster, 1, swap_ab=True
    )


def swapped_knobs(in_fmt: str = "bf16") -> List[Knobs]:
    """The swapped record space of a format's classes: one record per (weight tile, token tile) -- 128 x 8 / 16 / 32 and
    64 x 16 / 32 for bf16; 128 x 16 / 32 and 64 x 16 / 32 for e4m3; 128 x 8 / 16 / 32 for the block-scaled formats."""
    return [
        swap_knobs(tn, weight_tile=wt, in_fmt=in_fmt)
        for wt in weight_tiles_of(in_fmt)
        for tn in token_tiles_of(wt, in_fmt)
    ]


def is_swapped_record(knobs: Knobs, in_fmt: str = "bf16") -> bool:
    """A record the swapped body takes for the format: the swapped 1-CTA record at one of the weight tiles and its narrow
    token tiles, no cluster, swizzle 1, no in-cluster split, no in-CTA chunk fold, the TMA store flag of every record."""
    return (
        knobs.swap_ab
        and knobs.tile_m in weight_tiles_of(in_fmt)
        and knobs.tile_n in token_tiles_of(knobs.tile_m, in_fmt)
        and knobs.cluster_shape_mn == (1, 1)
        and not knobs.use_2cta_instrs
        and knobs.swizzle_size == 1
        and knobs.cluster_k == 1
        and not knobs.cta_chunks
        and knobs.use_tma_store
    )


def plan_knobs_of(knobs: Knobs) -> Knobs:
    """The dense twin that sizes the static plan (tile, cluster, stages): the swapped flag names the body, not a tile shape."""
    return dataclasses.replace(knobs, swap_ab=False)


class SwappedGroupedGemm:
    """``out[p, n] = chain(sum_k rows[p, k] w[n, k, e(p)])`` for every dense row ``p`` of every group (with the fp8 classes'
    chains ``alpha = mul(sa[0], alpha[e(p)])`` scales the accumulators; with the block-scaled chains the block scales are applied
    inside the instruction and the W4A4 per-token scale per column by the chain), the weights as the MMA's A and the group's
    rows as a narrow TMA box on its B side. The persistent grid (``max_active_clusters``, in CTAs) is a dynamic launch argument:
    which CTA computes a tile, never the bits. The chain's operand format selects the operand block and the record space; a
    chain with scale operands launches through :meth:`with_scales`, the block-scaled chains through :meth:`block_scaled` /
    :meth:`block_scaled_static`; the gather form belongs to the bf16 classes."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Union[str, GroupedEpilogueBlock],
        gather: bool = False,
        sf_src: str = "128x4",
        *,
        k_block: int = 0,
        plan_in_kernel: bool = False,
    ):
        """``gather``: the gather form -- the bf16 classes' by the API's choice, the block-scaled classes' by the record
        (``knobs.gather_a``); ``sf_src``: the block-scaled gather form's scale source, the class's flat 128x4 buffer of the token
        rows or the pre-quantized packs' row-major bytes (``"linear"``); ``k_block``: K elements per smem stage of a
        block-scaled chain's body (0: the format's one swizzle atom of K; a multiple of the instruction's K step: several
        atoms, the MXFP8 classes' two-atom decode stage), a placement of the same fold; ``plan_in_kernel``: the decode form
        of the block-scaled gather record -- the scale-factor warp builds the group tables and the dense row map from the
        routing's ids itself (``PLAN_MAX_SLOTS`` slots, ``PLAN_MAX_E`` experts), the plan launch gone; the first CTA stores
        ``counts`` and ``dest`` for the kernels after it. The same integers as the plan kernel's."""
        if isinstance(chain, str):
            chain = GroupedChain.legacy(chain, "bf16")
        assert isinstance(chain, GroupedEpilogueBlock), chain
        self.in_fmt = chain.in_fmt
        self.operands = operands_for(self.in_fmt)  # the body reads the dense packing
        assert is_swapped_record(knobs, self.in_fmt), (
            "the swapped body takes the narrow swapped records of the format (128 or 64 weight rows x its narrow token "
            "tiles, 1-CTA)",
            knobs,
            self.in_fmt,
        )
        gather = gather or knobs.gather_a
        assert not gather or self.in_fmt == "bf16" or self.operands.block_scaled, (
            "the gather form is the bf16 classes' (the API's choice) and the block-scaled classes' (the record's flag)",
            knobs,
            self.in_fmt,
        )
        assert not knobs.gather_a or self.operands.block_scaled, (
            "the gather record is the block-scaled classes'",
            knobs,
        )
        # the block-scaled gather form reads the token rows of the one-format classes (the fp4 pairs' 4-bit rows, the MXFP8
        # classes' 8-bit rows); the mixed operand block's 8-bit tokens against its 4-bit weights take the box form
        assert not gather or not self.operands.mixed, (
            "the block-scaled gather form is the one-format classes'; the mixed classes read the token box",
            knobs,
            self.in_fmt,
        )
        assert sf_src in ("128x4", "linear"), sf_src
        self.knobs, self.chain, self.gather, self.sf_src = knobs, chain, gather, sf_src
        self.token_tile, self.weight_tile = knobs.tile_n, knobs.tile_m
        assert k_block == 0 or (
            self.operands.block_scaled and k_block % MMA_K_OF[self.in_fmt] == 0
        ), ("the k-block is the block-scaled bodies' whole instruction steps", k_block)
        self.k_block = k_block
        assert not (gather and k_block), (
            "the gather form's stage is one swizzle atom of K (the gather4 box is the atom): no k-block",
            knobs,
            k_block,
        )
        assert not plan_in_kernel or (gather and self.operands.block_scaled), (
            "the in-kernel plan is the block-scaled gather record's (the decode form)",
            knobs,
        )
        self.plan_in_kernel = plan_in_kernel
        # the warp roles: the epilogue warpgroup, (the block-scaled formats' scale-factor warp,) the MMA warp, the TMA warp
        bs = self.operands.block_scaled
        self.epi_wgs = epi_warpgroups(self.token_tile, chain)
        self.epilogue_warps = epilogue_warps(self.token_tile, chain)
        self.sf_warp = len(self.epilogue_warps) if bs else None
        self.mma_warp = len(self.epilogue_warps) + (1 if bs else 0)
        self.tma_warp = self.mma_warp + 1
        self.threads = 32 * (self.tma_warp + 1)

    @cute.jit
    def __call__(
        self,
        rows: cute.Tensor,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The TMA-box form of the chains without scale operands: ``rows (R, K, 1)`` the dense permuted rows."""
        assert not self.gather and not self.chain.operands
        assert not self.operands.block_scaled
        self._launch(rows, w, out, counts, max_active_clusters, stream, None, Int32(1))

    @cute.jit
    def with_scales(
        self,
        rows: cute.Tensor,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        sa: cute.Tensor,
        si: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The TMA-box form of the chains with scale operands (the fp8 per-tensor classes): ``alpha (E,)`` the per-expert
        weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier."""
        assert not self.gather and self.chain.scale_operands
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            alpha=alpha,
            sa=sa,
            si=si,
        )

    @cute.jit
    def gathered(
        self,
        x: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form: ``x (T, K)`` the unpermuted rows, ``src (>= Tk,)`` int32 the plan's slot of every expanded row."""
        assert self.gather
        self._launch(x, w, out, counts, max_active_clusters, stream, src, k_div)

    @cute.jit
    def block_scaled(
        self,
        rows: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The block-scaled chains with the W4A4 row scale: ``rows`` the dense rows (at least the token tile past the live rows),
        ``sfa`` their scale factors at the padded positions, ``w`` / ``sfw`` the experts and their scale buffers, ``pts`` the
        per-token scales at the padded positions."""
        assert self.operands.block_scaled and not self.chain.needs_aux
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            alpha=alpha,
            sfa=sfa,
            sfw=sfw,
            pts=pts,
        )

    @cute.jit
    def block_scaled_gathered(
        self,
        x: cute.Tensor,
        sfx: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form of :meth:`block_scaled`: ``x (T, K/2, 1)`` the unpermuted packed token rows, ``sfx`` their flat scale
        factors, ``pts (T,)`` their per-token scales (one entry: a unit scale), ``src (>= Tk,)`` int32 the plan's slot of every
        dense row, ``k_div`` the slots per token; ``out (Tk, Nout)`` sizes the dense rows."""
        assert self.gather and self.operands.block_scaled and not self.chain.needs_aux
        self._launch(
            x,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            src,
            k_div,
            alpha=alpha,
            sfa=sfx,
            sfw=sfw,
            pts=pts,
        )

    @cute.jit
    def block_scaled_plain_gathered(
        self,
        x: cute.Tensor,
        sfx: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form of :meth:`block_scaled_plain` (the MXFP4 classes)."""
        assert self.gather and self.operands.block_scaled and not self.chain.operands
        self._launch(
            x, w, out, counts, max_active_clusters, stream, src, k_div, sfa=sfx, sfw=sfw
        )

    @cute.jit
    def block_scaled_quant_gathered(
        self,
        x: cute.Tensor,
        sfx: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form of :meth:`block_scaled_quant` (the MXFP8 classes' fused requantization): ``x`` the unpermuted token
        rows, ``sfx`` their scale bytes (the pack's row-major layout or the flat 128x4 buffer), ``src`` the plan's slot of every
        dense row."""
        assert self.gather and self.operands.block_scaled and not self.chain.operands
        assert self.chain.quant, self.chain
        self._launch(
            x,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            src,
            k_div,
            sfa=sfx,
            sfw=sfw,
            out_sf=out_sf,
        )

    @cute.jit
    def block_scaled_quant_gathered_plan(
        self,
        x: cute.Tensor,
        sfx: cute.Tensor,
        ids: cute.Tensor,
        n_slots: Int32,
        k_div: Int32,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        dest: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """:meth:`block_scaled_quant_gathered` with the plan built in the kernel: ``ids [>= n_slots]`` the routing's expert ids in
        slot order (an id outside the expert range is a dropped slot; slot ``i`` is token ``i // k_div``), ``counts [E]`` and
        ``dest [n_slots]`` written by the first CTA (the plan kernel's integers) for the kernels after this one."""
        assert self.gather and self.plan_in_kernel and self.operands.block_scaled
        assert self.chain.quant and not self.chain.operands, self.chain
        self._launch(
            x,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            ids,
            k_div,
            sfa=sfx,
            sfw=sfw,
            out_sf=out_sf,
            ids=ids,
            n_slots=n_slots,
            dest=dest,
        )

    @cute.jit
    def block_scaled_static_gathered(
        self,
        x: cute.Tensor,
        sfx: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        aux: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form of :meth:`block_scaled_static` (the intermediate's scale bytes still go to the padded positions)."""
        assert self.gather and self.operands.block_scaled and self.chain.needs_aux
        self._launch(
            x,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            src,
            k_div,
            alpha=alpha,
            sfa=sfx,
            sfw=sfw,
            pts=pts,
            out_sf=out_sf,
            aux=aux,
        )

    @cute.jit
    def block_scaled_plain(
        self,
        rows: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The block-scaled chains without further operands (the MX classes, MXFP8 and MXFP4): ``rows`` the dense rows, ``sfa`` their scale
        factors at the padded positions, ``w`` / ``sfw`` the experts and their scale buffers."""
        assert self.operands.block_scaled and not self.chain.operands
        assert not self.chain.quant, self.chain
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            sfa=sfa,
            sfw=sfw,
        )

    @cute.jit
    def block_scaled_quant(
        self,
        rows: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The block-scaled chains without further operands whose epilogue quantizes (the MXFP8 classes' fused
        requantization): ``out`` the codes of the dense rows, ``out_sf`` the flat 128x4 scale bytes at the padded positions."""
        assert self.operands.block_scaled and not self.chain.operands
        assert self.chain.quant, self.chain
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            sfa=sfa,
            sfw=sfw,
            out_sf=out_sf,
        )

    @cute.jit
    def block_scaled_static(
        self,
        rows: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        aux: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The static class's chains (as :meth:`.kernel.GroupedBlockGemm.block_scaled_static`): ``out`` the packed codes,
        ``out_sf`` the flat 128x4 scale bytes at the padded positions, ``aux[e]`` the expert's ``gs2`` (GEMM1) or ``inv2``
        (GEMM2, ``out_sf`` unread)."""
        assert self.operands.block_scaled and self.chain.needs_aux
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            alpha=alpha,
            sfa=sfa,
            sfw=sfw,
            pts=pts,
            out_sf=out_sf,
            aux=aux,
        )

    def _launch(
        self,
        rows,
        w,
        out,
        counts,
        max_active_clusters,
        stream,
        src,
        k_div,
        *,
        alpha=None,
        sa=None,
        si=None,
        sfa=None,
        sfw=None,
        pts=None,
        out_sf=None,
        aux=None,
        ids=None,
        n_slots=None,
        dest=None,
    ):
        ops = self.operands
        if cutlass.const_expr(not ops.mixed and rows.element_type != w.element_type):
            raise TypeError(
                f"rows and w must have one type: {rows.element_type} != {w.element_type}"
            )
        # the gather form's slab is one swizzle atom of K wide (the gather4 box is the atom): it keeps the format's default stage
        body = "gather" if self.gather else "swapped"
        sfw_t = None
        self.xch_layout = self.chain.smem_layout(1, token_tile=self.token_tile)
        if cutlass.const_expr(ops.block_scaled):
            rows = ops.view_tokens(
                rows
            )  # (R, K, 1): the dense token rows, the MMA's B operand
            w = ops.view_weights(w)  # (N, K, E): the MMA's A operand (M = N, K, L = E)
            sfw_t = ops.sf_view_experts(sfw, w.shape[0], w.shape[1], w.shape[2])
            # the block-scaled plan takes the swapped record itself (its knob rule admits the narrow token tiles with the
            # swapped flag); the A ring (the weights) ends inside the sm_100 instructions' descriptor window; the stage holds
            # ``k_block`` elements of K when given (the two-atom stage), the format's atom otherwise
            st = ops.plan(
                self.knobs,
                lambda st: (
                    ops.table_bytes
                    + xch_smem_bytes(self.xch_layout)
                    + (PLAN_SMEM_BYTES if self.plan_in_kernel else 0)
                ),
                body,
                ops.plan_cls.UMMA_SMEM_ADDR_MAX,
                k_blocks=(self.k_block // MMA_K_OF[self.in_fmt])
                if self.k_block
                else None,
            )
            self.block = st.block
            assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
        else:
            assert self.xch_layout is None, self.chain
            # the weights are the plan's A operand, the token rows its B operand
            st = ops.plan(plan_knobs_of(self.knobs), w, rows, ops.table_bytes, body)
            self.block = ops.mainloop(body)
        self.st = st
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        assert (
            st.cta_tile_shape_mnk[0] == self.weight_tile
            and st.cta_tile_shape_mnk[1] == self.token_tile
        ), st.cta_tile_shape_mnk
        # the weights as the A operand: (M = N, K, L = E), TMA with the expert as the batch coordinate
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.select(
            st.a_smem_layout_staged,
            mode=list(range(cute.rank(st.a_smem_layout_staged) - 1)),
        )
        # `internal_type`: the TMA's unpacking format of a sub-byte weight into the 8-bit smem container (the mixed sm_100
        # class's e2m1 weight on the twin plan's A side); None for the one-format plans and the dense formats
        atom_w, tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            a_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=getattr(st, "A_TMA_INTERNAL", None),
        )
        b_one = cute.select(
            st.b_smem_layout_staged,
            mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
        )
        tile_k = st.cta_tile_shape_mnk[2]
        atom_sfw = tensor_sfw = None
        sf_bytes = 0
        if cutlass.const_expr(ops.block_scaled):
            sfa_one = cute.select(
                st.sfa_smem_layout_staged,
                mode=list(range(cute.rank(st.sfa_smem_layout_staged) - 1)),
            )
            atom_sfw, tensor_sfw = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                sfw_t,
                sfa_one,
                st.mma_tiler,
                st.tiled_mma,
                st.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            sf_bytes = cute.size_in_bytes(st.sf_dtype, sfa_one)
            self.sf_stage_bytes = ops.sf_stage_bytes(st)
            self.sf_groups = ops.sf_groups(st)
        if cutlass.const_expr(self.gather):
            # the token rows gathered by the TMA's gather4 mode over the 2-D x: a (token_tile, k-tile) box whose rows come
            # from the index tensor, written to the swizzled stage as the dense box writes it (the atom carries the swizzle);
            # the block-scaled classes' packed rows through their element view's 2-D form
            rows2 = rows
            if cutlass.const_expr(ops.block_scaled):
                rows2 = cute.make_tensor(
                    rows.iterator,
                    cute.make_layout(
                        (rows.shape[0], rows.shape[1]),
                        stride=(rows.layout.stride[0], 1),
                    ),
                )
            slab = cute.make_composed_layout(
                st.b_smem_layout_staged.inner,
                0,
                cute.make_layout((self.token_tile, tile_k), stride=(tile_k, 1)),
            )
            gI = cute.make_tensor(
                src.iterator,
                cute.make_layout((src.shape[0], rows2.shape[1]), stride=(1, 0)),
            )
            atom_tok, tensor_tok = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensor2DGather4G2SOp(tcgen05.CtaGroup.ONE),
                rows2,
                slab,
                (self.token_tile, tile_k),
                gmem_coord_tensor=gI,
            )
            m_src = src
        else:
            # the token rows as the B operand: a box of token_tile rows x one k-tile, re-addressed per tile at the group's row
            b_op = utils.sm100.cluster_shape_to_tma_atom_B(
                st.cluster_shape_mn, st.tiled_mma.thr_id
            )
            atom_tok, tensor_tok = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                rows,
                b_one,
                st.mma_tiler,
                st.tiled_mma,
                st.cluster_layout_vmnk.shape,
                internal_type=getattr(st, "B_TMA_INTERNAL", None),
            )
            m_src = counts  # unread
        self.num_tma_load_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
            + sf_bytes
        )
        if cutlass.const_expr(ops.block_scaled):
            self.kernel(
                st.tiled_mma,
                atom_w,
                tensor_w,
                atom_tok,
                tensor_tok,
                out,
                counts,
                m_src,
                k_div,
                st.cluster_layout_vmnk,
                st.a_smem_layout_staged,
                st.b_smem_layout_staged,
                st.epi_tile,
                alpha,
                sa,
                si,
                atom_sfw,
                tensor_sfw,
                sfa,
                st.sfa_smem_layout_staged,
                st.sfb_smem_layout_staged,
                pts,
                out_sf,
                aux,
                ids,
                n_slots,
                dest,
            ).launch(
                grid=(1, 1, max_active_clusters),
                block=[self.threads, 1, 1],
                cluster=(1, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            self.kernel(
                st.tiled_mma,
                atom_w,
                tensor_w,
                atom_tok,
                tensor_tok,
                out,
                counts,
                m_src,
                k_div,
                st.cluster_layout_vmnk,
                st.a_smem_layout_staged,
                st.b_smem_layout_staged,
                st.epi_tile,
                alpha,
                sa,
                si,
            ).launch(
                grid=(1, 1, max_active_clusters),
                block=[self.threads, 1, 1],
                cluster=(1, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW_mkl: cute.Tensor,
        tma_atom_tok: cute.CopyAtom,
        mTok_nkl: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        mSrc: cute.Tensor,
        k_div: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        mAlpha: cute.Tensor = None,
        mSa: cute.Tensor = None,
        mSi: cute.Tensor = None,
        tma_atom_sfw: Optional[cute.CopyAtom] = None,
        mSFW_mkl: Optional[cute.Tensor] = None,
        mSFA_flat: Optional[cute.Tensor] = None,
        sfa_smem_layout_staged: Optional[cute.Layout] = None,
        sfb_smem_layout_staged: Optional[cute.Layout] = None,
        mPts: Optional[cute.Tensor] = None,
        mOutSf: Optional[cute.Tensor] = None,
        mAux: Optional[cute.Tensor] = None,
        mIds: Optional[cute.Tensor] = None,
        n_slots: Optional[Int32] = None,
        mDest: Optional[cute.Tensor] = None,
    ):
        st = self.st
        ops = self.operands
        bs = ops.block_scaled
        tn, wt = self.token_tile, self.weight_tile
        mma_warp, tma_warp, sf_warp = self.mma_warp, self.tma_warp, self.sf_warp
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == tma_warp:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_tok)
            if cutlass.const_expr(bs):
                cpasync.prefetch_descriptor(tma_atom_sfw)
        # programmatic dependent launch: every warp waits for the preceding grid before the group tables (the tables read
        # ``counts``, the predecessor chain's writes); the producer waits once more before its first read of the
        # predecessor's output (the token rows), the epilogue before its first store; the trigger placement is the chain's
        # (``pdl.TRIGGER``: the early site right after the wait, the late site after the producer's tile loop).
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        bidx, _, _ = cute.arch.block_idx()
        is_leader_cta = (
            bidx % cute.size(tiled_mma.thr_id.shape) == 0
        )  # every CTA (the 1-CTA instruction)
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mW_mkl.shape[0]
        Tk = mOut.shape[0]
        K = mTok_nkl.shape[1]
        pcols = Int32(0)
        if cutlass.const_expr(bs):
            # scale-factor columns of the padded buffer
            pcols = (
                (K // st.sf_vec + SF_ATOM_KBLOCKS - 1) // SF_ATOM_KBLOCKS
            ) * SF_ATOM_KBLOCKS

        smem = cutlass.memory.SmemAllocator()
        if cutlass.const_expr(bs and self.plan_in_kernel):

            @cute.struct
            class SharedStorageBlockScaledPlan:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                sf_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_off_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                # the in-kernel plan: the histogram of the slots, the experts' slot membership masks, the dense row map
                plan_hist: cute.struct.MemRange[cutlass.Int32, MAX_E]
                plan_mask: cute.struct.MemRange[cutlass.Int32, 4 * PLAN_MAX_E]
                plan_src: cute.struct.MemRange[cutlass.Int32, PLAN_MAX_SLOTS]

            storage = smem.allocate(SharedStorageBlockScaledPlan)
        elif cutlass.const_expr(bs):

            @cute.struct
            class SharedStorageBlockScaled:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                sf_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_off_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

            storage = smem.allocate(SharedStorageBlockScaled)
        else:

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

            storage = smem.allocate(SharedStorage)
        # one TMA pipeline: the weight tile (and its scale-factor atom) and the token box of one k-tile per stage
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        sf_pipeline = None
        if cutlass.const_expr(bs):
            # the cp.async pipeline: the token tile's scale-factor rows (one warp of producers)
            sf_pipeline = PipelineCpAsyncUmma.create(
                barrier_storage=storage.sf_full_mbar_ptr.data_ptr(),
                num_stages=st.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                defer_sync=True,
            )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warps)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(self.epilogue_warps) + 1)
        )
        epi_join = pipeline.NamedBarrier(
            barrier_id=BAR_EPI_JOIN, num_threads=32 * len(self.epilogue_warps)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warps[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            **ops.tmem_alloc_kwargs(st),
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_off_pad = None
        if cutlass.const_expr(bs):
            s_off_pad = storage.group_off_pad.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_src = None
        if cutlass.const_expr(bs and self.plan_in_kernel):
            s_src = storage.plan_src.get_tensor(cute.make_layout((PLAN_MAX_SLOTS,)))
        src_t = s_src if cutlass.const_expr(bs and self.plan_in_kernel) else mSrc
        cute.arch.griddepcontrol_wait()  # the tables read the predecessor chain's counts (or its ids)
        pdl.trigger_early()  # the early placement's site: right after the wait, as in the grouped body
        if cutlass.const_expr(bs):
            # the three tables in one scan on the scale-factor warp (idle until its first tile) while the allocator warp issues
            # the TMEM allocation: the two longest legs of the prologue overlap instead of running back to back
            if warp_idx == sf_warp:
                if cutlass.const_expr(self.plan_in_kernel):
                    # the plan from the ids: the histogram by shared-memory atomics (integers: exact in every order), the
                    # tables from it, the rank of every slot = its group's offset + the slots of its expert before it
                    # (the expert's membership mask over the slots, each lane adding its own bit: a sum of distinct bits is
                    # their union); the dense row map for the producers; the first CTA stores counts and dest
                    lane = cute.arch.lane_idx()
                    s_hist = storage.plan_hist.get_tensor(cute.make_layout((MAX_E,)))
                    s_mask = storage.plan_mask.get_tensor(
                        cute.make_layout((4 * PLAN_MAX_E,))
                    )
                    for h in cutlass.range(lane, E, 32):
                        s_hist[h] = Int32(0)
                    for h in cutlass.range(lane, 4 * E, 32):
                        s_mask[h] = Int32(0)
                    for p in cutlass.range(lane, PLAN_MAX_SLOTS, 32):
                        s_src[p] = Int32(-1)
                    cute.arch.sync_warp()
                    ee = cute.make_rmem_tensor(
                        cute.make_layout((PLAN_MAX_SLOTS // 32,)), Int32
                    )
                    for q in cutlass.range_constexpr(PLAN_MAX_SLOTS // 32):
                        i = lane + Int32(32 * q)
                        e = Int32(-1)
                        if i < n_slots:
                            e = mIds[i]
                        if (e < Int32(0)) | (e >= E):
                            e = Int32(-1)  # a dropped slot: not counted, no position
                        ee[q] = e
                        if e >= Int32(0):
                            cute.arch.atomic_add(s_hist.iterator + e, Int32(1))
                            cute.arch.atomic_add(
                                s_mask.iterator + (e * Int32(4) + Int32(q)),
                                Int32(1) << lane,
                            )
                    cute.arch.sync_warp()
                    build_group_tables_dual(
                        s_hist, E, tn, s_off, s_pre, s_off_pad, ops.group_align
                    )
                    for q in cutlass.range_constexpr(PLAN_MAX_SLOTS // 32):
                        i = lane + Int32(32 * q)
                        e = ee[q]
                        if e >= Int32(0):
                            r = s_off[e]
                            for wq in cutlass.range_constexpr(PLAN_MAX_SLOTS // 32):
                                m = Uint32(s_mask[e * Int32(4) + Int32(wq)])
                                if cutlass.const_expr(wq < q):
                                    r = r + Int32(cute.arch.popc(m))
                                elif cutlass.const_expr(wq == q):
                                    r = r + Int32(
                                        cute.arch.popc(m & cute.arch.lanemask_lt())
                                    )
                            s_src[r] = i
                            if bidx == Int32(0):
                                mDest[i] = r
                        elif i < n_slots:
                            if bidx == Int32(0):
                                mDest[i] = Int32(-1)
                    if bidx == Int32(0):
                        for h in cutlass.range(lane, E, 32):
                            counts[h] = s_hist[h]
                    cute.arch.sync_warp()
                else:
                    build_group_tables_dual(
                        counts, E, tn, s_off, s_pre, s_off_pad, ops.group_align
                    )
            if warp_idx < len(self.epilogue_warps):
                tmem.allocate(st.num_tmem_alloc_cols)
        else:
            if warp_idx == self.epilogue_warps[0]:
                build_group_tables(
                    counts, E, tn, s_off, s_pre
                )  # dense offsets; the "m-block" is the token tile
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        # the operand rings at the plan's container types (the mixed sm_100 class's e2m1 weight occupies a byte per element)
        a_smem_dtype, b_smem_dtype = ops.smem_dtypes(st)
        sW = smem.allocate_tensor(
            element_type=a_smem_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sTok = smem.allocate_tensor(
            element_type=b_smem_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFW = sSFA = None
        if cutlass.const_expr(bs):
            sSFW = smem.allocate_tensor(
                element_type=st.sf_dtype,
                layout=sfa_smem_layout_staged,
                byte_alignment=128,
            )
            sSFA = smem.allocate_tensor(
                element_type=st.sf_dtype,
                layout=sfb_smem_layout_staged,
                byte_alignment=128,
            )
        sX = None
        if cutlass.const_expr(self.xch_layout is not None):
            # a quantising chain's exchange buffer (the fused requantization's half-block amax between the two warps of a block);
            # the layout is built here (a layout built at plan time belongs to the host region)
            sX = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=self.chain.smem_layout(1, token_tile=tn),
                byte_alignment=16,
            )
        gW_mkl = cute.local_tile(
            mW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        k_tile_cnt = cute.size(gW_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW_mkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sW, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )
        tWsSFW = tWgSFW = None
        tCtSFW_layout = tCtSFA_layout = None
        if cutlass.const_expr(bs):
            gSFW_mkl = cute.local_tile(
                mSFW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgSFW = thr_mma.partition_A(gSFW_mkl)
            tWsSFW, tWgSFW = cpasync.tma_partition(
                tma_atom_sfw,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sSFW, 0, 3),
                cute.group_modes(tCgSFW, 0, 3),
            )
            tWsSFW = cute.filter_zeros(tWsSFW)
            tWgSFW = cute.filter_zeros(tWgSFW)
        tCrW = tiled_mma.make_fragment_A(sW)
        tCrTok = tiled_mma.make_fragment_B(sTok)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        if cutlass.const_expr(bs):
            tCtSFW_layout = bs_layout.make_tmem_layout_sfa(
                tiled_mma,
                st.mma_tiler,
                st.sf_vec,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA_layout = bs_layout.make_tmem_layout_sfb(
                tiled_mma,
                st.mma_tiler,
                st.sf_vec,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        # the scheduler's "cluster rows" are the token tile, its "N tiles" the weight tiles
        if warp_idx == tma_warp:
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            if cutlass.const_expr(self.gather):
                swap_gather_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_w,
                    tma_atom_tok,
                    tWgW,
                    tWsW,
                    mTok_nkl,
                    src_t,
                    k_div,
                    sTok,
                    Tk,
                    tn,
                    st.cta_tile_shape_mnk[2],
                    tma_atom_sfw=tma_atom_sfw,
                    tWgSFW=tWgSFW,
                    tWsSFW=tWsSFW,
                )
            else:
                swap_tma_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_w,
                    tma_atom_tok,
                    tWgW,
                    tWsW,
                    mTok_nkl,
                    sTok,
                    thr_mma,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tma_atom_sfw=tma_atom_sfw,
                    tWgSFW=tWgSFW,
                    tWsSFW=tWsSFW,
                )
        if cutlass.const_expr(bs):
            if warp_idx == sf_warp:
                tile_sched = GroupScheduler.create(
                    s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
                )
                cute.arch.griddepcontrol_wait()  # the token scale factors are the predecessor's
                swap_sf_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    sf_pipeline,
                    mSFA_flat,
                    sSFA,
                    tidx - 32 * sf_warp,
                    Tk,
                    K,
                    pcols,
                    s_off,
                    s_off_pad,
                    tn,
                    self.sf_groups,
                    self.sf_stage_bytes,
                    src_t if cutlass.const_expr(self.gather) else None,
                    k_div,
                    self.sf_src == "linear",
                )
        if warp_idx == mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            if cutlass.const_expr(bs):
                tCtSFW = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + st.num_acc_tmem_cols, dtype=st.sf_dtype),
                    tCtSFW_layout,
                )
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + st.num_acc_tmem_cols + st.num_sfa_tmem_cols,
                        dtype=st.sf_dtype,
                    ),
                    tCtSFA_layout,
                )
                swap_mma_loop(
                    self.lowering,
                    st,
                    k_tile_cnt,
                    tile_sched,
                    tiled_mma,
                    ab_consumer,
                    sf_pipeline,
                    acc_pipeline,
                    tCrW,
                    tCrTok,
                    tCtAcc_base,
                    tCtSFW,
                    tCtSFA,
                    sSFW,
                    sSFA,
                )
            else:
                mma_loop(
                    self.lowering,
                    st.num_acc_stage,
                    k_tile_cnt,
                    is_leader_cta,
                    tile_sched,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    tCrW,
                    tCrTok,
                    tCtAcc_base,
                )
        if warp_idx < len(self.epilogue_warps):
            if cutlass.const_expr(not bs):
                tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            cute.arch.griddepcontrol_wait()  # the output rows were the predecessor chain's reads
            quant_loop = self.chain.swapped_loop()
            if cutlass.const_expr(quant_loop is not None):
                # a quantising chain's own epilogue in the transposed orientation (the MXFP8 classes' fused requantization:
                # the codes of the dense rows, the scale bytes at the padded positions)
                quant_loop(
                    st,
                    self.chain,
                    tidx % 128,
                    tile_sched,
                    tCtAcc_base,
                    thr_mma,
                    mOut,
                    mOutSf,
                    Tk,
                    N,
                    (wt, tn // self.epi_wgs)
                    if cutlass.const_expr(self.epi_wgs > 1)
                    else epi_tile,
                    acc_pipeline,
                    sX,
                    s_off,
                    s_off_pad,
                    wt,
                    subtile=tidx // 128
                    if cutlass.const_expr(self.epi_wgs > 1)
                    else None,
                )
            else:
                swapped_epilogue_loop(
                    st,
                    self.chain,
                    tidx % 128,
                    tile_sched,
                    tCtAcc_base,
                    thr_mma,
                    mOut,
                    Tk,
                    N,
                    (wt, tn // self.epi_wgs)
                    if cutlass.const_expr(self.epi_wgs > 1)
                    else epi_tile,
                    acc_pipeline,
                    subtile=tidx // 128
                    if cutlass.const_expr(self.epi_wgs > 1)
                    else None,
                    mAlpha=mAlpha,
                    mSa=mSa,
                    mSi=mSi,
                    mPts=mPts,
                    mAux=mAux,
                    mOutSf=mOutSf,
                    s_off=s_off,
                    s_off_pad=s_off_pad,
                    mSrc=src_t if cutlass.const_expr(self.gather and bs) else None,
                    k_div=k_div,
                )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: the epilogue warpgroup's warps join here after their last TMEM loads
            # (the block-scaled body's allocation was issued in the prologue's own region: the column count is given explicitly)
            epi_join.arrive_and_wait()
            if cutlass.const_expr(bs):
                tmem.free(tmem_ptr, st.num_tmem_alloc_cols)
            else:
                tmem.free(tmem_ptr)


@cute.jit
def swap_tma_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tma_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mTok_nkl: cute.Tensor,
    sTok: cute.Tensor,
    thr_mma,
    b_cta_coord,
    b_cta_layout,
    tma_atom_sfw: Optional[cute.CopyAtom] = None,
    tWgSFW: Optional[cute.Tensor] = None,
    tWsSFW: Optional[cute.Tensor] = None,
):
    """The TMA producer: per tile the expert's weight tile ``n_tile`` (128 or 64 rows; the expert as the batch coordinate; with
    the block-scaled formats its scale-factor atom too) and the token box re-addressed at the group's row ``row0``
    (``token_tile`` rows: the rows past the group's end are computed and never stored, the rows past the buffer are
    zero-filled); k-tiles ascending, one ring slot holds both. The weights are inputs of the whole chain, the token rows the
    predecessor's output: the first tile's weight stages (as many as the ring holds) are issued before the
    programmatic-dependent-launch wait and their token boxes right after it, so the ring fills while the predecessor drains;
    the stage's transaction barrier completes when every copy has landed, whatever their order."""
    has_sf = tma_atom_sfw is not None
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)  # a CTA without tiles issues nothing
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    tWgSFW_slice = None
    if cutlass.const_expr(has_sf):
        tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
    real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
    gTok = cute.local_tile(
        real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
    )
    tCgTok = thr_mma.partition_B(gTok)
    tTsTok, tTgTok = cpasync.tma_partition(
        tma_atom_tok,
        b_cta_coord,
        b_cta_layout,
        cute.group_modes(sTok, 0, 3),
        cute.group_modes(tCgTok, 0, 3),
    )
    tTgTok_slice = tTgTok[(None, 0, None, 0)]
    tma_producer.reset()
    tok_producer = (
        tma_producer.clone()
    )  # the same stages, walked again for the token boxes
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tma_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        if cutlass.const_expr(has_sf):
            cute.copy(
                tma_atom_sfw,
                tWgSFW_slice[(None, handle.count)],
                tWsSFW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
    cute.arch.griddepcontrol_wait()  # the token rows are the predecessor's
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tok_producer.current_handle()
        cute.copy(
            tma_atom_tok,
            tTgTok_slice[(None, handle.count)],
            tTsTok[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        tok_producer.advance()
    k_start = n_pre
    while work_tile.is_valid_tile:
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
        if cutlass.const_expr(has_sf):
            tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
        real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
        gTok = cute.local_tile(
            real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgTok = thr_mma.partition_B(gTok)
        tTsTok, tTgTok = cpasync.tma_partition(
            tma_atom_tok,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sTok, 0, 3),
            cute.group_modes(tCgTok, 0, 3),
        )
        tTgTok_slice = tTgTok[(None, 0, None, 0)]
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = tma_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = tma_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            if cutlass.const_expr(has_sf):
                cute.copy(
                    tma_atom_sfw,
                    tWgSFW_slice[(None, handle.count)],
                    tWsSFW[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            cute.copy(
                tma_atom_tok,
                tTgTok_slice[(None, handle.count)],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = tma_producer.try_acquire()
        tma_producer.reset()  # the next tile's k-tiles count from 0
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()  # the CTA's loads are all issued: the dependents' prologues may start
    tma_producer.tail()


@cute.jit
def swap_gather_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tma_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mX: cute.Tensor,
    mSrc: cute.Tensor,
    k_div: Int32,
    sTok: cute.Tensor,
    Tk: Int32,
    tn: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    tma_atom_sfw: Optional[cute.CopyAtom] = None,
    tWgSFW: Optional[cute.Tensor] = None,
    tWsSFW: Optional[cute.Tensor] = None,
):
    """The TMA producer of the gather form: per tile the expert's weight tile (128 or 64 rows; the expert as the batch
    coordinate; with the block-scaled formats its scale-factor atom too) and the token tile gathered by the TMA's gather4 mode from the unpermuted rows -- the tile's ``tn`` token indices (``src[p] //
    k_div`` for the group's rows, the group's last row past its end: a valid box whose extra rows are never stored) go into
    registers once per tile and every k-tile's gather issues from them; both copies count on the stage's transaction barrier.
    The kernel has waited for the predecessor (the plan's launch) before its tables, so the wait here returns at once; the
    loop keeps the box form's shape (the first tile's weight stages, the wait, the token gathers)."""
    has_sf = tma_atom_sfw is not None
    stage_elems = tn * tile_k
    sTok_plain = cute.make_tensor(
        sTok.iterator,
        cute.make_layout(
            (tn, tile_k, st.num_ab_stage), stride=(tile_k, 1, stage_elems)
        ),
    )  # the atom applies the swizzle
    sTok_plain = cute.group_modes(sTok_plain, 0, 2)
    idx = cute.make_rmem_tensor(cute.make_layout((tn,)), Int32)
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    tWgSFW_slice = None
    if cutlass.const_expr(has_sf):
        tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
    tma_producer.reset()
    tok_producer = tma_producer.clone()
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tma_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        if cutlass.const_expr(has_sf):
            cute.copy(
                tma_atom_sfw,
                tWgSFW_slice[(None, handle.count)],
                tWsSFW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
    cute.arch.griddepcontrol_wait()  # the plan's row map is the predecessor's
    row0 = work_tile.row0
    last = cutlass.max(cutlass.min(work_tile.row_end, Tk) - Int32(1), Int32(0))
    for j in cutlass.range_constexpr(tn):
        idx[j] = mSrc[cutlass.min(row0 + Int32(j), last)] // k_div
    gX = cute.local_tile(
        mX, (tn, tile_k), (0, None)
    )  # (tn, tile_k, n_kt): the K tiles of the gathered rows
    n_kt = cute.size(gX, mode=[2])
    gI = cute.make_tensor(
        idx.iterator, cute.make_layout((tn, tile_k, n_kt), stride=(1, 0, 0))
    )
    tTsTok, tTgX, tTgI = cpasync.tma_partition(
        tma_atom_tok,
        0,
        cute.make_layout(1),
        sTok_plain,
        [cute.group_modes(gX, 0, 2), cute.group_modes(gI, 0, 2)],
    )
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tok_producer.current_handle()
        cute.copy(
            tma_atom_tok,
            [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
            tTsTok[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        tok_producer.advance()
    k_start = n_pre
    while work_tile.is_valid_tile:
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
        if cutlass.const_expr(has_sf):
            tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
        row0 = work_tile.row0
        last = cutlass.max(cutlass.min(work_tile.row_end, Tk) - Int32(1), Int32(0))
        for j in cutlass.range_constexpr(tn):
            idx[j] = mSrc[cutlass.min(row0 + Int32(j), last)] // k_div
        gI = cute.make_tensor(
            idx.iterator, cute.make_layout((tn, tile_k, n_kt), stride=(1, 0, 0))
        )
        tTsTok, tTgX, tTgI = cpasync.tma_partition(
            tma_atom_tok,
            0,
            cute.make_layout(1),
            sTok_plain,
            [cute.group_modes(gX, 0, 2), cute.group_modes(gI, 0, 2)],
        )
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = tma_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = tma_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            if cutlass.const_expr(has_sf):
                cute.copy(
                    tma_atom_sfw,
                    tWgSFW_slice[(None, handle.count)],
                    tWsSFW[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            cute.copy(
                tma_atom_tok,
                [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = tma_producer.try_acquire()
        tma_producer.reset()
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    tma_producer.tail()


@cute.jit
def swap_mma_loop(
    lowering,
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tiled_mma: cute.TiledMma,
    tma_consumer,
    sf_pipeline,
    acc_pipeline,
    tCrW: cute.Tensor,
    tCrTok: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    tCtSFW: cute.Tensor,
    tCtSFA: cute.Tensor,
    sSFW: cute.Tensor,
    sSFA: cute.Tensor,
):
    """The block-scaled MMA loop with the weights, their scale factors and the token rows behind the TMA pipeline and the token
    scale factors behind the LDGSTS pipeline: both waited per k-tile, the per-block ``tcgen05.cp`` of both scale-factor sets,
    the lowering's inner steps with the SFA / SFB TMEM slices (the narrow token tile reads the SFB atom from its first rows: no
    window shift), ``ACCUMULATE`` off on the fold's first step only, both stages released by the commit."""
    assert cute.size(tCrW, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    tiled_copy_sfw, tCsSFW_s2t, tCtSFW_s2t = s2t_copy_and_partition(
        sSFW, tCtSFW, st.cta_group, st.S2T_BROADCAST
    )
    tiled_copy_sfa, tCsSFA_s2t, tCtSFA_s2t = s2t_copy_and_partition(
        sSFA, tCtSFA, st.cta_group, st.S2T_BROADCAST
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_acc_stage
    )
    sf_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
        tma_consumer.reset()
        sf_state.reset_count()
        peek_w = tma_consumer.try_wait()
        acc_pipeline.producer_acquire(acc_state)
        for k_tile in range(k_tile_cnt):
            handle = tma_consumer.wait_and_advance(peek_w)
            sf_pipeline.consumer_wait(sf_state)
            w_stage = (None, None, None, None, handle.index)
            sf_stage = (None, None, None, None, sf_state.index)
            cute.copy(tiled_copy_sfw, tCsSFW_s2t[w_stage], tCtSFW_s2t)
            cute.copy(tiled_copy_sfa, tCsSFA_s2t[sf_stage], tCtSFA_s2t)
            for j in lowering.inner_steps():
                tiled_mma.set(tcgen05.Field.ACCUMULATE, lowering.accumulate(k_tile, j))
                tiled_mma.set(tcgen05.Field.SFA, tCtSFW[(None, None, j)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFA[(None, None, j)].iterator)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrW[(None, None, j, handle.index)],
                    tCrTok[(None, None, j, handle.index)],
                    tCtAcc,
                )
            handle.release()
            sf_pipeline.consumer_release(sf_state)
            sf_state.advance()
            peek_w = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_w = tma_consumer.try_wait()
        acc_pipeline.producer_commit(acc_state)
        acc_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def _col_scale(
    chain: cutlass.Constexpr,
    alpha_e,
    mPts: cute.Tensor,
    row0: Int32,
    j: Int32,
    rows_left: Int32,
    shift: Int32,
    mSrc: cute.Tensor = None,
    k_div: Int32 = None,
):
    """The row scale of token column ``j`` of the tile: ``chain.row_scale`` at the padded position of dense row ``row0 + j``
    (``+ shift``: the group's padded offset less its dense one), or in the gather form (``mSrc`` given) at the token
    ``src[row0 + j] // k_div``; columns past the valid tokens read a valid row and store nothing."""
    p = row0 + cutlass.min(j, cutlass.max(rows_left - Int32(1), Int32(0)))
    if cutlass.const_expr(mSrc is not None):
        p = cutlass.min(cutlass.max(p, Int32(0)), mSrc.shape[0] - Int32(1))
        p = cutlass.max(mSrc[p], Int32(0)) // k_div
    else:
        p = cutlass.max(p, Int32(0)) + shift
    p = cutlass.min(cutlass.max(p, Int32(0)), mPts.shape[0] - Int32(1))
    return chain.row_scale(MoeF32, alpha_e, mPts, p)


@cute.jit
def swapped_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mAlpha: cute.Tensor = None,
    mSa: cute.Tensor = None,
    mSi: cute.Tensor = None,
    mPts: cute.Tensor = None,
    mAux: cute.Tensor = None,
    mOutSf: cute.Tensor = None,
    s_off: cute.Tensor = None,
    s_off_pad: cute.Tensor = None,
    subtile=None,
    mSrc: cute.Tensor = None,
    k_div: Int32 = None,
):
    """The transposed group-masked epilogue (``subtile``: this warpgroup's one token subtile of the tile when several
    warpgroups split the tile's columns, the accumulator released once every warp has arrived; None: one warpgroup takes
    every subtile): the accumulator tile is (weight rows, tokens); the 128-row TMEM layout gives
    thread ``t`` weight row ``t`` of the tile and its token columns (the record's N tile). Gated chain: the gate accumulators
    come from lane ``+8`` (``shfl.sync.down``); the up lanes (row % 16 < 8) compute the chain and store output column
    ``(n // 16) * 8 + n % 8``; plain and ungated chains: every lane stores column ``n``. Token ``j`` of the tile is dense row
    ``row0 + j``, stored iff ``< row_end``. Every output element's chain is computed by one thread from the same accumulator
    value in the same fp32 sequence as the grouped body's epilogue (one element per store: the token columns of a thread's
    weight row are not contiguous in the output, so this body's stores are scalars by construction). The chains with scale
    operands (``mAlpha`` / ``mSa`` / ``mSi``, as in the grouped epilogue) form ``alpha_e = mul(sa, alpha[e])`` per tile and
    read ``inv = si[0]`` as given; the e4m3 outputs are converted in pairs (two token columns of one weight row; each value's
    conversion is independent of its pairing) and stored as bytes. The block-scaled chains' per-token scale is per column,
    ``s_j = mul(alpha_e, pts[row0 + j + off_pad[e] - off[e]])`` (the row scales at the padded positions); the static class's
    GEMM1 (``out_fmt == "nvfp4"``) quantizes every live token column's 16-output block across the warp's output lanes (a block =
    the up lanes of one warp for the gated chain, 16 consecutive lanes for the ungated one): the block amax by a butterfly
    (exact in every association), the e4m3 scale and the encode scale on every lane, the codes paired into bytes by the even
    lanes (``mOut`` the packed-code buffer ``[rows, I/2]`` uint8, ``mOutSf`` the flat 128x4 scale bytes at the padded positions,
    ``mAux[e] = gs2[e]``); its GEMM2 (``scale == "w4a4s"``): ``s = mul(alpha[e], mAux[e])`` once per tile."""
    pair = chain.pair
    paired_cvt = chain.paired_cvt
    static_q = chain.static_quant
    per_row = chain.per_row_scale
    n_blocks = pcols = Int32(0)
    if cutlass.const_expr(static_q):
        N_out = N // Int32(2) if cutlass.const_expr(pair) else N
        n_blocks = N_out // Int32(Q.NVFP4_BLOCK)
        pcols = ((n_blocks + Int32(3)) // Int32(4)) * Int32(4)
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk, st.c_layout, st.c_dtype, ACC_DTYPE, epi_tile, False
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    idF = cute.make_identity_tensor(
        (st.cta_tile_shape_mnk[0], st.cta_tile_shape_mnk[1], 1)
    )  # the tile's own (weight row, token) coordinates
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF = thr_t2r.partition_D(cF_epi)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    nv = len(offs)
    tile_m = st.cta_tile_shape_mnk[0]
    # the thread's weight rows: one (the 128-row tile: lane = row, the gate row GROUP lanes below) or two (the 64-row tile's
    # lane split: rows r and r + GROUP of one thread, so the gated pair sits in the thread's own registers)
    row_offs = sorted({m for m, _ in offs})
    assert row_offs in ([0], [0, GROUP]), (
        "the swapped epilogue's TMEM layouts: one weight row per thread, or the (r, r + 8) pair",
        offs,
        frag.layout,
    )
    two_rows = len(row_offs) == 2
    assert not (static_q and two_rows), (
        "the static class's block quantization takes the 128-row layout (one weight row per thread)"
    )
    index = {o: v for v, o in enumerate(offs)}
    partner = [
        index.get((m + GROUP, n)) for m, n in offs
    ]  # the gate register of an up register (two_rows)
    ups = [
        v for v in range(nv) if offs[v][0] == 0
    ]  # the registers that hold an up row (all, with one row)
    assert not two_rows or all(partner[v] is not None for v in ups), offs
    cols = [
        n - offs[0][1] for _, n in offs
    ]  # the register's token column inside the subtile
    rows = [m for m, _ in offs]  # the register's weight row inside the thread's pair
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    inv = None
    if cutlass.const_expr(paired_cvt):
        inv = chain.inv(mSi)
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile, e = (
            work_tile.row0,
            work_tile.row_end,
            work_tile.n_tile,
            work_tile.e,
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        alpha_e = chain.tile_scale(MoeF32, mAlpha, mSa, mAux, e)
        shift = None
        if cutlass.const_expr(per_row):
            shift = s_off_pad[e] - s_off[e]  # the row scales at the padded positions
        gs6 = inv_gs = None
        if cutlass.const_expr(static_q):
            gs6, inv_gs = chain.static_consts(MoeF32, mAux, e)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        subtiles = (
            [subtile]
            if cutlass.const_expr(subtile is not None)
            else list(range(subtile_cnt))
        )
        for s in subtiles:
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            crd0 = tTR_cF[(None, None, None, s)][(0, 0, 0)]
            wrow = crd0[0]  # this thread's (first) weight row in the tile
            col0 = crd0[1] - Int32(offs[0][1])  # the subtile's first token column
            n_glob = n_tile * Int32(tile_m) + wrow
            # the gate accumulators of the up rows: lane + 8 (rows 16q + 8 + r sit 8 lanes below rows 16q + r in the 128-row
            # layout), or the thread's own second row (the 64-row layout)
            gate = None
            if cutlass.const_expr(pair):
                if cutlass.const_expr(two_rows):
                    gate = [
                        tTR_rAcc[partner[v]] if partner[v] is not None else tTR_rAcc[v]
                        for v in range(nv)
                    ]
                else:
                    gate = [
                        cute.arch.shuffle_sync_down(tTR_rAcc[v], offset=8)
                        for v in range(nv)
                    ]
                is_up = (wrow % Int32(2 * GROUP)) < Int32(GROUP)
                out_col = (
                    n_tile * Int32(tile_m // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = is_up & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            # the chain of a token column is computed only where the column is a live token of the group (a warp-uniform
            # branch: every lane of the warp holds the same column), so a tile with few live tokens pays few chains
            if cutlass.const_expr(static_q):
                # the static class's GEMM1: per live token column, the block of this warp's output lanes
                if cutlass.const_expr(pair):
                    bcol = n_tile * Int32(
                        tile_m // (2 * Q.NVFP4_BLOCK)
                    ) + wrow // Int32(2 * Q.NVFP4_BLOCK)  # 16 outputs = 32 weight rows
                    even = (wrow % Int32(GROUP)) % Int32(2) == Int32(0)
                    sf_lane = (wrow % Int32(32)) == Int32(0)
                else:
                    bcol = n_tile * Int32(tile_m // Q.NVFP4_BLOCK) + wrow // Int32(
                        Q.NVFP4_BLOCK
                    )
                    even = (wrow % Int32(2)) == Int32(0)
                    sf_lane = (wrow % Int32(Q.NVFP4_BLOCK)) == Int32(0)
                byte_col = out_col // Int32(2)
                for v in cutlass.range_constexpr(nv):
                    j = col0 + Int32(cols[v])
                    if j < rows_left:
                        sv = _col_scale(
                            chain, alpha_e, mPts, row0, j, rows_left, shift, mSrc, k_div
                        )
                        val = chain.elem(
                            MoeF32,
                            tTR_rAcc[v],
                            gate[v] if cutlass.const_expr(pair) else None,
                            sv,
                            inv,
                        )
                        amax = MoeF32.fabs(val)
                        for off in (
                            (1, 2, 4, 16) if cutlass.const_expr(pair) else (1, 2, 4, 8)
                        ):
                            amax = MoeF32.fmaxc(
                                amax, cute.arch.shuffle_sync_bfly(amax, offset=off)
                            )
                        sf_byte, sf_d = cvt_f32_to_e4m3(MoeF32.mul(amax, gs6))
                        inv_b = MoeF32.fmin(
                            MoeF32.div(MoeF32.const(1.0), MoeF32.mul(sf_d, inv_gs)),
                            MoeF32.const(3.4028234663852886e38),
                        )
                        y = MoeF32.mul(val, inv_b)
                        y_odd = cute.arch.shuffle_sync_down(
                            y, offset=1
                        )  # the neighbouring output column's value
                        word = cvt_f32x8_to_e2m1x8(
                            y, y_odd, y, y, y, y, y, y
                        )  # byte 0 = (this column, the next)
                        prow = row0 + j
                        if store_ok & even & (bcol < n_blocks):
                            mOut[(prow, byte_col)] = (word & Uint32(0xFF)).to(Uint8)
                        if sf_lane & (bcol < n_blocks):
                            mOutSf[sf_byte_offset(prow + shift, bcol, pcols)] = (
                                sf_byte.to(Uint8)
                            )
            elif cutlass.const_expr(paired_cvt):
                outs = ups if cutlass.const_expr(pair) else list(range(nv))
                assert len(outs) % 2 == 0, offs
                for q in cutlass.range_constexpr(0, len(outs), 2):
                    v0, v1 = outs[q], outs[q + 1]
                    j0 = col0 + Int32(cols[v0])
                    j1 = col0 + Int32(cols[v1])
                    if (j0 < rows_left) | (j1 < rows_left):
                        y0 = chain.elem(
                            MoeF32,
                            tTR_rAcc[v0],
                            gate[v0] if cutlass.const_expr(pair) else None,
                            alpha_e,
                            inv,
                        )
                        y1 = chain.elem(
                            MoeF32,
                            tTR_rAcc[v1],
                            gate[v1] if cutlass.const_expr(pair) else None,
                            alpha_e,
                            inv,
                        )
                        pr = cvt_f32x2_to_fp8x2(y0, y1, "e4m3")
                        if store_ok & (j0 < rows_left) & (n_glob + Int32(rows[v0]) < N):
                            mOut[(row0 + j0, out_col + Int32(rows[v0]))] = (
                                pr & 0xFF
                            ).to(Uint8)
                        if store_ok & (j1 < rows_left) & (n_glob + Int32(rows[v1]) < N):
                            mOut[(row0 + j1, out_col + Int32(rows[v1]))] = (pr >> 8).to(
                                Uint8
                            )
            else:
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(pair and rows[v] != 0):
                        continue  # a gate register (two_rows): folded into its up register's chain
                    j = col0 + Int32(cols[v])
                    if j < rows_left:
                        sv = alpha_e
                        if cutlass.const_expr(per_row):
                            sv = _col_scale(
                                chain,
                                alpha_e,
                                mPts,
                                row0,
                                j,
                                rows_left,
                                shift,
                                mSrc,
                                k_div,
                            )
                        val = chain.elem(
                            MoeF32,
                            tTR_rAcc[v],
                            gate[v] if cutlass.const_expr(pair) else None,
                            sv,
                            inv,
                        )
                        if store_ok & (n_glob + Int32(rows[v]) < N):
                            mOut[(row0 + j, out_col + Int32(rows[v]))] = val
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
