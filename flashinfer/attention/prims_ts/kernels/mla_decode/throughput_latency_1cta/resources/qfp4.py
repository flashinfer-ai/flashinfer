# Copyright (c) 2026 by FlashInfer team.
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

"""Resources of the NVFP4 token-sparse path (native FP4 QK).

``SmemQFp4Resource``: the softmax warps quantize the CTA's 64 Q heads to NVFP4
(block 16, E4M3 block scale) once per work tile, write the packed E2M1 rows into
a 128-row K-major tile (``tcgen05.mma.kind::mxf4nvf4`` needs M = 128: TMEM lane
32q + l holds head 16q + (l & 15), so the 16x32bx2 softmax reads are unchanged)
and write the Q block scales (SFA) as UTCCP source blocks.  The MMA warp copies
them into TMEM with ``tcgen05.cp``, consumes the SMEM descriptors and releases
the tile with ``tcgen05.commit`` at the end of the work tile.

Token block scales (SFB): the loader lands each K tile's 40-byte scale rows
token-major in the raw ring's scale slot (``cp.async``) and publishes them
through ``ScaleLandedResource``; the dequant warps write the ten K64 chunks
into TMEM with ``tcgen05.st`` (``SfbReadyResource``) and the MMA warp waits on
that before issuing BMM1.  The dequant warps convert V from the resident latent
slices and release the raw stage only after BMM1 completed (``Bmm1DoneResource``).

TMEM scale-factor layout (verified by ``mxf4_mma_probe.py``): 4 columns per
K64 chunk, 4-column aligned base; for the output block (M quadrant r, N quadrant
q) subpartition r reads SFA at column base + r and SFB at column base + q from
its own lanes 32r + i.  ``tcgen05.cp.32x128b.warpx4`` fills exactly that from a
512-byte INTERLEAVE block whose 16-byte line ``i`` holds the chunk words of rows
i, 32 + i, 64 + i, 96 + i (byte offset ``16 * (r % 32) + 4 * (r // 32)`` for
row r), broadcast to the four quadrants.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.layout import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    decode_gen_task_cache,
    smem_array,
)
from ...helpers.nvfp4 import (
    cp_async_wait_group,
    e4m3x4_to_f32x4,
    ld_shared_v2,
    tcgen05_fence_after_thread_sync,
    f32_to_e4m3_rounded,
    f32x8_to_e2m1x8,
    fence_proxy_async_shared,
    ld_global_v4,
    rcp_approx_f32,
    st_shared_u32,
    st_shared_v4,
)
from ...helpers.ops import (
    freeze_smem_descriptor,
    lane_idx_from_thread,
    tcgen05_st_32x32b_x8,
)
from ...helpers.stage import MlaStage
from ...helpers.tile import local_kv_tile_idx
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_q_for_stage,
    head_idx_for_stage,
)

from .common import MlaResource

# Q tile geometry (fixed by the 576/512 MLA head and the M128 instruction).
QFP4_ROWS = 128
QFP4_LATENT_BYTES = 512 // 2  # 256 B per row: two SW128 segments of 128 B
QFP4_ROPE_BYTES = 64  # four 16-byte groups [q_g | q_g] per row (SW64), see quantize_q
QFP4_SEG_BYTES = QFP4_ROWS * 128  # 16 KB per latent segment
QFP4_ROPE_SEG_BYTES = QFP4_ROWS * QFP4_ROPE_BYTES  # 8 KB
QFP4_TILE_BYTES = 2 * QFP4_SEG_BYTES + QFP4_ROPE_SEG_BYTES  # 40 KB
NUM_K64_CHUNKS = 10  # 512 latent + 64 rope (main) + 64 rope (residual)
SF_COLS_PER_CHUNK = 4
SF_BLOCK_BYTES = 512  # UTCCP 32x128b source block of one K64 chunk (128 rows x 4 B)
SF_BLOCKS_BYTES = NUM_K64_CHUNKS * SF_BLOCK_BYTES  # 5 KB per operand


@cute.jit
def sf_block_byte_offset(row, chunk: int) -> Int32:
    """SMEM byte offset of ``row``'s word of ``chunk`` inside a UTCCP block set."""
    return (
        Int32(chunk * SF_BLOCK_BYTES)
        + (row & Int32(31)) * Int32(16)
        + (row >> Int32(5)) * Int32(4)
    )


@cute.jit
def issue_sf_utccp(tmem_col_base, blocks_smem_addr) -> None:
    """Issue the ten ``tcgen05.cp`` copies of one SF block set (one elected thread)."""
    for c in cutlass.range_constexpr(NUM_K64_CHUNKS):
        desc = prims.Tcgen05SmemDesc.build(
            blocks_smem_addr + Int32(c * SF_BLOCK_BYTES),
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
        prims.tcgen05_cp(
            prims.Tcgen05CpShape.SHAPE_32X128B,
            prims.make_tmem_ptr(tmem_col_base + Int32(SF_COLS_PER_CHUNK * c), Int32),
            desc,
            group=prims.CTAGroup.CTA_1,
            multicast=prims.Tcgen05CpMulticast.WARPX4,
        )


# =====================================================================
# SmemQFp4Resource — Q quantized to NVFP4 (SMEM operand + SFA blocks)
# =====================================================================


@dataclass(kw_only=True)
class SmemQFp4Resource(MlaResource):
    """Per-work-tile NVFP4 Q operand produced by the softmax warps."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "q_desc_lat0_var",
            Int64,
            Int64(0),
            "Q latent segment 0 descriptor.",
            "q_desc_lat0",
        ),
        (
            "q_desc_lat1_var",
            Int64,
            Int64(0),
            "Q latent segment 1 descriptor.",
            "q_desc_lat1",
        ),
        (
            "q_desc_rope_var",
            Int64,
            Int64(0),
            "Q rope segment descriptor.",
            "q_desc_rope",
        ),
    )
    q_latent: object = None
    q_rope: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    _smem_q: object = None
    _sf_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _smem_sf: object = None
    q_desc_lat0_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    q_desc_lat1_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    q_desc_rope_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_smem_requirements(self):
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=QFP4_TILE_BYTES,
                alignment=1024,
            )
            self._sf_alloc = SmemAllocation(
                name=f"{self.name}_sf",
                size_bytes=SF_BLOCKS_BYTES,
                alignment=16,
            )
        return [self._alloc, self._sf_alloc]

    def get_tmem_requirements(self):
        # SFA (Q block scales, this resource) followed by SFB (token block
        # scales, copied per K tile by the MMA from the raw scale slot):
        # 4 columns per K64 chunk each.
        if self._tmem_alloc is None:
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}_sf",
                num_columns=2 * self.cfg.tmem_sf_cols,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self._smem_q = smem_array(context, self._alloc, cutlass.Uint8, QFP4_TILE_BYTES)
        self._smem_sf = smem_array(
            context, self._sf_alloc, cutlass.Uint8, SF_BLOCKS_BYTES
        )
        self._init_tmem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_quant_state(self, stage_info: StageInfo) -> None:
        """Softmax-side view initialization."""
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """MMA-side view initialization."""
        self._init_smem_state(stage_info)

    @cute.jit
    def _tile_byte_address(self, byte_offset) -> Int32:
        return self._smem_q.data_ptr(Int32(byte_offset)).toint(cutlass.Int32)

    @cute.jit
    def _sf_byte_address(self, byte_offset) -> Int32:
        return self._smem_sf.data_ptr(Int32(byte_offset)).toint(cutlass.Int32)

    @cute.jit
    def _quantize_q_rows(
        self,
        stage_info: StageInfo,
        *,
        lat_lo: cutlass.Constexpr[int],
        lat_hi: cutlass.Constexpr[int],
        do_rope: cutlass.Constexpr[bool],
    ):
        """Quantize latent blocks [lat_lo, lat_hi) (+ the rope groups) of this thread's row.

        Streaming: every pair of blocks is written as one 16-byte SMEM chunk
        and every four blocks as one SFA word as soon as they are ready, so the
        live set stays small.  The row is the task-local thread index (0..127;
        the second 128-thread half of the softmax task shares the rows and
        quantizes the other blocks).
        """
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        xt = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        row = xt & Int32(127)
        lane = xt & Int32(31)
        local_head = ((row >> Int32(5)) << Int32(4)) + (lane & Int32(15))
        head_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        # TMA-equivalent coordinates: flat query row = cta_idx_q * H + head.
        flat_row = cta_idx_q * Int32(cfg.num_heads_q) + head_idx + local_head
        q_pos = flat_row // Int32(cfg.num_heads_q)
        head = flat_row - q_pos * Int32(cfg.num_heads_q)
        lat = self.q_latent
        rope = self.q_rope
        lat_base = lat.iterator.toint() + (
            Int64(head) * Int64(lat.stride[0])
            + Int64(q_pos) * Int64(lat.stride[2])
            + Int64(batch_idx) * Int64(lat.stride[3])
        )
        rope_base = rope.iterator.toint() + (
            Int64(head) * Int64(rope.stride[0])
            + Int64(q_pos) * Int64(rope.stride[2])
            + Int64(batch_idx) * Int64(rope.stride[3])
        )
        inv6 = Float32(1.0 / 6.0)

        def quantize_block(words4):
            """16 E4M3 (four Int32 words) -> (scale byte, two E2M1 words)."""
            vals = []
            for w in words4:
                vals += list(e4m3x4_to_f32x4(w))
            amax = cute.math.max(vals[0], -vals[0], ftz=True)
            for v in vals[1:]:
                amax = cute.math.max(amax, cute.math.max(v, -v, ftz=True), ftz=True)
            sf_byte, sf_val = f32_to_e4m3_rounded(amax * inv6)
            inv = Float32(0.0) if sf_val == Float32(0.0) else rcp_approx_f32(sf_val)
            scaled = [v * inv for v in vals]
            lo = f32x8_to_e2m1x8(*scaled[0:8])
            hi = f32x8_to_e2m1x8(*scaled[8:16])
            return sf_byte, lo, hi

        def sf_word(b0, b1, b2, b3):
            return (
                (b0 & Int32(0xFF))
                | ((b1 & Int32(0xFF)) << Int32(8))
                | ((b2 & Int32(0xFF)) << Int32(16))
                | ((b3 & Int32(0xFF)) << Int32(24))
            )

        swz128 = row & Int32(7)
        swz64 = (row >> Int32(1)) & Int32(3)
        sf_base = self._sf_byte_address(0)
        # Latent blocks: block pairs land as one SW128 16-byte chunk, block
        # quads as one SFA word.
        prev = {}
        sf_win = {}
        for b in cutlass.range_constexpr(lat_lo, lat_hi):
            words4 = ld_global_v4(lat_base + Int64(16 * b))
            sf_byte, lo, hi = quantize_block(words4)
            sf_win[b] = sf_byte
            if cutlass.const_expr(b % 2 == 0):
                prev[0] = lo
                prev[1] = hi
            else:
                seg = b // 16
                c = (b % 16) // 2
                row_addr = self._tile_byte_address(seg * QFP4_SEG_BYTES) + row * Int32(
                    128
                )
                st_shared_v4(
                    row_addr + ((Int32(c) ^ swz128) * Int32(16)),
                    prev[0],
                    prev[1],
                    lo,
                    hi,
                )
            if cutlass.const_expr(b % 4 == 3):
                st_shared_u32(
                    sf_base + sf_block_byte_offset(row, b // 4),
                    sf_word(sf_win[b - 3], sf_win[b - 2], sf_win[b - 1], sf_win[b]),
                )
        if cutlass.const_expr(do_rope):
            rope_row_addr = self._tile_byte_address(2 * QFP4_SEG_BYTES) + row * Int32(
                64
            )
            rope_sf = {}
            for b in cutlass.range_constexpr(4):
                words4 = ld_global_v4(rope_base + Int64(16 * b))
                sf_byte, lo, hi = quantize_block(words4)
                rope_sf[b] = sf_byte
                # 16-byte chunk b = rope group b twice: [q_g (main) | q_g (residual)].
                st_shared_v4(
                    rope_row_addr + ((Int32(b) ^ swz64) * Int32(16)), lo, hi, lo, hi
                )
            # K64 chunk 8 = groups 0-1 (each scale twice), chunk 9 = groups 2-3.
            st_shared_u32(
                sf_base + sf_block_byte_offset(row, 8),
                sf_word(rope_sf[0], rope_sf[0], rope_sf[1], rope_sf[1]),
            )
            st_shared_u32(
                sf_base + sf_block_byte_offset(row, 9),
                sf_word(rope_sf[2], rope_sf[2], rope_sf[3], rope_sf[3]),
            )

    @producer_work
    @cute.jit
    def quantize_q(self, stage_info: StageInfo):
        """Quantize this CTA's Q heads to NVFP4 and write the SFA blocks (256 softmax threads).

        Thread ``t & 127`` (warp w = (t & 127) >> 5, lane l) owns row ``t & 127``
        of the 128-row operand = head ``16w + (l & 15)`` (rows l and l + 16 of a
        quadrant carry the same head; only lanes 0-15 of each TMEM quadrant are
        read by the softmax).  Per 16-element block: scale = e4m3(amax / 6),
        q = e2m1(x / scale) (RN, satfinite).  TRT-LLM packs the K rope per
        16-value group as [main 8 B | residual 8 B] with scale bytes
        [m0 r0 m1 r1 m2 r2 m3 r3], so K64 chunk 8 covers groups 0-1 and chunk 9
        groups 2-3, each group twice; Q mirrors that: group g appears twice in a
        row with its scale duplicated.
        """
        xt = decode_gen_task_cache(stage_info)[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        # The two 128-thread halves of the softmax task split the row: half 0
        # -> blocks 0-15 (SF chunks 0-3); half 1 -> blocks 16-31 + rope (SF
        # chunks 4-9).
        if xt < Int32(128):
            self._quantize_q_rows(stage_info, lat_lo=0, lat_hi=16, do_rope=False)
        else:
            self._quantize_q_rows(stage_info, lat_lo=16, lat_hi=32, do_rope=True)
        # Generic-proxy SMEM writes (Q tile, SF blocks) -> async proxy (UMMA
        # descriptors, tcgen05.cp) before the pipeline arrive.
        fence_proxy_async_shared()

    @consumer_work(returns=("q_desc_lat0", "q_desc_lat1", "q_desc_rope"))
    @cute.jit
    def q_desc(self, stage_info: StageInfo):
        """Stage SFA (tcgen05.cp) and publish the three Q operand descriptors."""
        task_cache = decode_gen_task_cache(stage_info)
        if prims.elect_sync():
            issue_sf_utccp(
                task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
                + Int32(self._tmem_alloc.offset),
                self._sf_byte_address(0),
            )
        desc_lat0 = freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                self._tile_byte_address(0),
                leading_byte_offset=QFP4_SEG_BYTES,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        )
        desc_lat1 = freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                self._tile_byte_address(QFP4_SEG_BYTES),
                leading_byte_offset=QFP4_SEG_BYTES,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        )
        desc_rope = freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                self._tile_byte_address(2 * QFP4_SEG_BYTES),
                leading_byte_offset=QFP4_ROPE_SEG_BYTES,
                stride_byte_offset=512,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
        )
        return desc_lat0, desc_lat1, desc_rope


# =====================================================================
# Bmm1DoneResource — "BMM1 of raw stage t % 2 completed" (UMMA -> dequant)
# =====================================================================


@dataclass(kw_only=True)
class Bmm1DoneResource(MlaResource):
    """Two-stage barrier-only pipeline: the MMA commits it right after the BMM1
    instructions of K tile ``t`` (stage ``t % 2``, like the raw ring); the
    dequant warps wait on it before releasing that raw stage.

    A parity spin on the single-stage S barrier is not enough here: with BMM1
    issued ahead of PV, BMM1(t + 1) can complete before the dequant warps check
    BMM1(t), and the S barrier's parity would already have flipped back.  Two
    stages indexed like the raw ring cannot alias, because BMM1(t + 2) needs
    raw stage ``t % 2`` refilled, which happens only after this release.
    """

    @producer_work
    @cute.jit
    def bmm1_issued(self, stage_info: StageInfo) -> None:
        """Producer side: the BMM1 instructions were issued; ``commit`` tracks them."""
        del stage_info

    @consumer_work
    @cute.jit
    def bmm1_completed(self, stage_info: StageInfo) -> None:
        """Consumer side: observed after ``wait``; nothing to do."""
        del stage_info


# =====================================================================
# ScaleLandedResource (loader -> dequant) and SfbReadyResource (dequant ->
# MMA): SFB written with tcgen05.st from token-major scale rows
# =====================================================================


@dataclass(kw_only=True)
class ScaleLandedResource(MlaResource):
    """Barrier-only pipeline, loader warps -> dequant warps, one stage per
    scale slot: "the token-major scale rows of tile ``u`` have landed".

    The loader publishes tiles 0 and 1 after HEAD (it waited for all of its
    HEAD copies) and tile ``t + 1`` at the end of the loop step that gathers
    tile ``t``: that step's tail waited for every older cp.async group
    (``wait_group 0``, which includes scales(t + 1) committed at the previous
    tail) before committing scales(t + 2) and rows(t + 3), so the publication
    only has to keep those two newest groups in flight.  The dequant warps
    consume it before writing SFB(t + 1) (``SfbReadyResource``).
    """

    @producer_work
    @cute.jit
    def scales_landed(
        self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]
    ) -> None:
        """Producer side (loader warps): complete this tile's scale copies."""
        del stage_info
        if cutlass.const_expr(section == MlaStage.Loop):
            # scales(t + 1) completed in this step's tail; only scales(t + 2)
            # and rows(t + 3) may still be in flight.
            cp_async_wait_group(2)
        # No fence.acq_rel.cta here: the warp-converged ``.release.cta``
        # arrive of the pipeline after ``cp.async.wait_group`` already
        # publishes the landed rows to the dequant warps.

    @consumer_work
    @cute.jit
    def landed_ack(self, stage_info: StageInfo) -> None:
        """Consumer side (dequant warps): observed; nothing to do."""
        del stage_info


@dataclass(kw_only=True)
class SfbReadyResource(MlaResource):
    """Single-stage barrier pipeline, dequant warps -> MMA warp: "SFB(u) is in
    TMEM".  The eight dequant warps (two per TMEM lane quadrant) write the ten
    K64 chunks of the token block scales with ``tcgen05.st.32x32b`` straight
    from the token-major scale rows: lane ``i`` of quadrant ``q`` holds, in
    the four columns of chunk ``c``, the chunk-``c`` word of tokens ``i``,
    ``32 + i``, ``64 + i``, ``96 + i`` (the replicated SF layout the MMA
    reads), warps 0-3 of the task write chunks 0-3 and 8-9, warps 4-7 chunks
    4-7.  WAR on the single SFB buffer holds because the writer already
    observed BMM1(u - 1) complete (``bmm1_done``); the MMA waits here instead
    of issuing the UTCCP copies.
    """

    raw_ref: object = None
    qfp4_ref: object = None

    @producer_work
    @cute.jit
    def write_sfb(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int],
        ahead: cutlass.Constexpr[int],
    ) -> None:
        """Producer side (dequant warps): SFB(tile + ahead) <- token-major rows."""
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        tile = local_kv_tile_idx(cfg, stage_info, inst, 0, section=section) + Int32(
            ahead
        )
        raw = self.raw_ref
        slot_base = raw.smem_byte_address(
            raw.scale_slot_byte_offset(tile % Int32(cfg.nvfp4_scale_slots))
        )
        row_bytes = Int32(cfg.nvfp4_scale_row_bytes)
        warp_in_task = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.dequant_warp_idx
        )
        lane = lane_idx_from_thread(cute.arch.thread_idx()[0])
        sfb_col = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self.qfp4_ref._tmem_alloc.offset)
            + Int32(cfg.tmem_sf_cols)
        )

        # Token rows of this lane: tokens lane, 32 + lane, 64 + lane, 96 + lane.
        row_addrs = []
        for qq in cutlass.range_constexpr(4):
            tok = Int32(32 * qq) + lane
            row_addrs.append(slot_base + tok * row_bytes)

        # x8 stores (two chunks each) keep at most eight staged words live, so
        # the write fits next to the dequant's register-staged V slices.
        def _store_pair(c0):
            r8 = [Int32(0)] * 8
            for qq in cutlass.range_constexpr(4):
                w0, w1 = ld_shared_v2(row_addrs[qq] + Int32(4 * c0))
                r8[0 * 4 + qq] = w0
                r8[1 * 4 + qq] = w1
            tcgen05_st_32x32b_x8(sfb_col + Int32(SF_COLS_PER_CHUNK * c0), r8)

        if warp_in_task < Int32(4):
            # Chunks 0-3 and 8-9.
            _store_pair(0)
            _store_pair(2)
            _store_pair(8)
        else:
            # Chunks 4-7.
            _store_pair(4)
            _store_pair(6)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()

    @consumer_work
    @cute.jit
    def sfb_visible(self, stage_info: StageInfo) -> None:
        """Consumer side (MMA warp): order the dequant's TMEM stores before BMM1."""
        del stage_info
        tcgen05_fence_after_thread_sync()
