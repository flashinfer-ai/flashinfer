# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: BSD-3-Clause
#
# Packed conversion/store sequences adapted from hca_v41.py and
# hca_v41_primitives.py.
"""Selected mixed-cache tiles for the D512 Prims-TS Keep-AB graph.

SWA bytes and reconstructed page64/group16 FP4 share E4M3 cache units. The
FP4 conversion forms the exact E2M1 * E4M3 product (F16 on bundled CUDA >= 13.2,
BF16 otherwise) before E4M3 RN/SATFINITE packing, without a request amax,
division or descale. Loader-written D128 stages stay resident through delayed
PV; no full converted KV cache is materialized.

Gathers use cp.async: SWA rows copy straight into the KV stages, and main FP4
rows into thread-private raw staging that is converted from SMEM. Register
loads were rescheduled next to their consumers, serializing L2 round trips.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch import nvvm_wrappers as nvw
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import SignalingThreads, WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    PipelineConfig,
    StageInfo,
    producer_work,
)

from ....attention.prims_ts.kernels.mla_decode.helpers.ops import (
    pack_float4_to_fp8_e4m3,
)
from ....attention.prims_ts.kernels.mla_decode.helpers.layout import smem_array
from ....attention.prims_ts.kernels.mla_decode.helpers.stage import MlaStage
from ....attention.prims_ts.kernels.mla_decode.helpers.tile import cta_idx_kv_for_stage
from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.resources.smem_resources import (
    SmemKvResource,
)
from .. import hca_v41_primitives as hw
from ._sparse_views import bind_combined_sparse_views


@dataclass(frozen=True)
class Fp4KvProducer:
    native_fp4: bool

    def compile_signature(self):
        return ("dsv41_page64_fp4_group16_primary", self.native_fp4)

    def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
        if (
            cfg.kernel_variant != "keeps_mma_ab"
            or cfg.qkv_dtype != "bf16"
            or cfg.rope_dim != 0
            or cfg.head_dim_qk != 512
            or cfg.head_dim_per_stage_kv != 128
            or cfg.tile_size_kv != 64
            or cfg.load_num_warps != 4
            or not cfg.sparse_reuse_kv
            or not cfg.sparse_direct
        ):
            raise ValueError(
                "packed V4.1 KV requires BF16 Keep-AB/BK64 retained stages"
            )
        # Every writing thread arrives after its async-proxy fence. Native TMA
        # completion bytes would never arrive for these ordinary SMEM stores.
        return PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.kv_stages,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=cta_layout_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    def make_resource(self, pools, **kwargs):
        return Fp4SmemKvResource(
            fp4_cache=pools[0],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@dataclass(kw_only=True)
class Fp4SmemKvResource(SmemKvResource):
    fp4_cache: object = None
    native_fp4: cutlass.Constexpr[bool] = True

    @cute.jit
    def _store_fragment(self, fragment, stage_base, row, dim):
        # A D128 stage is two consecutive [BK,64] BF16 planes. SW128 XORs
        # bits 3:5 of the element offset with row bits 0:2. This is the native
        # descriptor layout, not Frost's full-D512 SMEM staging layout.
        element = (
            (dim // 64) * (self.cfg.tile_size_kv * 64) + row * 64 + (dim & 63)
        ) ^ ((row & 7) * 8)
        dst = cute.make_ptr(
            cutlass.BFloat16,
            stage_base.data_ptr().toint(cutlass.Int32) + element * 2,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        hw.store_bf16_fragment(fragment, dst)

    @cute.jit
    def _store_fp4_chunk(self, chunk, scale_word, stage_base, row, dim):
        if cutlass.const_expr(self.native_fp4):
            packed = hw.fp8x2_to_bf16_word(scale_word).to(cutlass.Uint32)
            scale0 = ((packed & 65535) * 65537).to(cutlass.Int32)
            scale1 = ((packed >> 16) * 65537).to(cutlass.Int32)
            for word in cutlass.range_constexpr(4):
                scale = scale0 if word < 2 else scale1
                values = hw.fp4x8_to_bf16_words(chunk[word])
                bits = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
                # E2M1 times finite E4M3 is exact in BF16 (at most six
                # significant bits); avoid FP32 expansion on CUDA >= 13.2.
                for pair in cutlass.range_constexpr(4):
                    bits[pair] = prims.mul_bf16x2(values[pair], scale)
                fragment = cute.make_tensor(
                    cute.recast_ptr(bits.iterator, dtype=cutlass.BFloat16),
                    cute.make_layout(8),
                )
                self._store_fragment(fragment, stage_base, row, dim + word * 8)
        else:
            scale_bytes = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Uint8)
            scale_bytes[0] = (scale_word & 255).to(cutlass.Uint8)
            scale_bytes[1] = ((scale_word >> 8) & 255).to(cutlass.Uint8)
            scales = cute.make_tensor(
                cute.recast_ptr(scale_bytes.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(2),
            )
            for word in cutlass.range_constexpr(4):
                packed = cute.make_tensor(chunk.iterator + word, cute.make_layout(1))
                fp4 = cute.make_tensor(
                    cute.recast_ptr(packed.iterator, dtype=cutlass.Float4E2M1FN),
                    cute.make_layout(8),
                ).load()
                fp16 = nvw.cvt_f4e2m1x8_to_f16x8(fp4)
                values = cute.TensorSSA(fp16, (8,), cutlass.Float16).to(cutlass.Float32)
                scale = scales[word // 2].to(cutlass.Float32)
                fragment = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
                fragment.store((values * scale).to(cutlass.BFloat16))
                self._store_fragment(fragment, stage_base, row, dim + word * 8)

    @cute.jit
    def _load_fp4_stage(
        self,
        pool,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
        chunk_elements = 32
        chunks_per_row = 128 // chunk_elements
        rows_per_group = 32 // chunks_per_row
        rows_per_warp = self.cfg.tile_size_kv // self.cfg.load_num_warps
        row_bytes, scale_bytes = 256, 32
        capacity = cutlass.Int64(cute.size(pool)) // (row_bytes + scale_bytes)
        base = page_id_slot * self.cfg.sparse_cache_words
        for group in cutlass.range_constexpr(rows_per_warp // rows_per_group):
            local_row = group * rows_per_group + lane // chunks_per_row
            row = warp * rows_per_warp + local_row
            chunk_idx = lane % chunks_per_row
            raw = cutlass.Int32(0x7FFFFFFF)
            # Page staging caches row quads in this warp. All writing lanes
            # recover their row from that cache; no index reread per D.
            for entry in cutlass.range_constexpr(4):
                value = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_page_ids[base + entry],
                    offset=local_row // 4,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                if local_row % 4 == entry:
                    raw = cutlass.Int32(value)
            slot = raw & cutlass.Int32(0x7FFFFFFF)
            chunk = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            for word in cutlass.range_constexpr(4):
                chunk[word] = 0
            scale_word = cutlass.Int32(0)
            if slot != cutlass.Int32(0x7FFFFFFF) and cutlass.Int64(slot) < capacity:
                # Page data and scale planes are separate. Widen before every
                # byte-stride multiplication, including pools above 2 GiB.
                page_base = cutlass.Int64(slot // 64) * (64 * (row_bytes + scale_bytes))
                token = cutlass.Int64(slot & 63)
                column = stage_idx * 128 + chunk_idx * chunk_elements
                data_offset = page_base + token * row_bytes + cutlass.Int64(column // 2)
                scale_offset = (
                    page_base
                    + 64 * row_bytes
                    + token * scale_bytes
                    + cutlass.Int64(column // 16)
                )
                data_ptr = cute.make_ptr(
                    cutlass.Int32,
                    pool.iterator.toint() + data_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                chunk.store(hw.load_cache_vector(data_ptr))
                scale_ptr = cute.make_ptr(
                    cutlass.Uint16,
                    pool.iterator.toint() + scale_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=2,
                )
                scale_word = hw.load_cache_scale(scale_ptr, cutlass.Uint16)
            dim = chunk_idx * chunk_elements
            self._store_fp4_chunk(chunk, scale_word, stage_base, row, dim)

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        del inst_id, is_v, section
        stage_base = self._stage_base(stage_info)
        self._load_fp4_stage(
            self.fp4_cache,
            stage_base,
            cached_page_ids,
            stage_idx=stage_idx,
            page_id_slot=page_id_slot,
        )
        # Publication is the task graph's AsyncThread producer commit. Each
        # writer fences its own stores; all 128 arrivals precede UMMA's wait.
        cute.arch.fence_view_async_shared()


@dataclass(frozen=True)
class Fp4Fp8KvProducer(Fp4KvProducer):
    def compile_signature(self):
        return ("dsv41_page64_fp4_group16_to_e4m3_units_v1", self.native_fp4)

    def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
        if (
            cfg.kernel_variant != "keeps_mma_ab"
            or cfg.qkv_dtype != "e4m3"
            or cfg.rope_dim != 0
            or cfg.head_dim_qk != 512
            or cfg.head_dim_per_stage_kv != 128
            or cfg.tile_size_q != 64
            or cfg.tile_size_kv != 128
            or cfg.kv_stages != 8
            or cfg.load_num_warps not in (4, 8)
            or not cfg.sparse_reuse_kv
            or not cfg.sparse_direct
        ):
            raise ValueError(
                "V4.1 FP4-to-E4M3 KV requires Keep-AB/M64/BK128/KV8 retained stages"
            )
        # The inherited loader fences every writer before the same all-writer
        # AsyncThread commit as the BF16 resource; these are not TMA stores.
        return PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.kv_stages,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=cta_layout_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    def make_resource(self, pools, *, runtime_args=None, **kwargs):
        if runtime_args is not None:
            raise ValueError(
                "FP4-to-E4M3 KV uses common cache units, not request scales"
            )
        return Fp4Fp8SmemKvResource(
            fp4_cache=pools[0],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@cute.jit
def _cp_async_16(
    dst_smem: cutlass.Int32, src_gmem: cutlass.Int64, src_bytes: cutlass.Int32
):
    """16B global-to-CTA-SMEM copy that zero-fills when src_bytes is 0."""
    prims.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[dst_smem, src_gmem, src_bytes],
    )


@cute.jit
def _cp_async_8(
    dst_smem: cutlass.Int32, src_gmem: cutlass.Int64, src_bytes: cutlass.Int32
):
    """8B global-to-CTA-SMEM copy that zero-fills when src_bytes is 0."""
    prims.inline_ptx(
        "cp.async.ca.shared.global [{$r0}], [{$r1}], 8, {$r2};",
        read_only_args=[dst_smem, src_gmem, src_bytes],
    )


@cute.jit
def _cp_async_commit():
    prims.inline_ptx("cp.async.commit_group;")


@cute.jit
def _cp_async_wait_group(pending: cutlass.Constexpr[int]):
    prims.inline_ptx(f"cp.async.wait_group {pending};")


@cute.jit
def _fp4x8_scaled_to_e4m3x8(word: cutlass.Int32, scale: cutlass.Int32):
    """Eight E2M1 values times one F16x2-replicated E4M3 scale, as E4M3 RN/SATFINITE.

    E2M1 * finite E4M3 has at most six significant bits and |x| <= 2688, so the
    F16 product is exact and its E4M3 rounding equals the BF16/FP32 route.
    """
    return prims.inline_ptx(
        "{ .reg .b8 b0, b1, b2, b3; .reg .b32 h0, h1, h2, h3; "
        ".reg .b16 e0, e1, e2, e3; "
        "mov.b32 {b0, b1, b2, b3}, {$r0}; "
        "cvt.rn.f16x2.e2m1x2 h0, b0; cvt.rn.f16x2.e2m1x2 h1, b1; "
        "cvt.rn.f16x2.e2m1x2 h2, b2; cvt.rn.f16x2.e2m1x2 h3, b3; "
        "mul.rn.f16x2 h0, h0, {$r1}; mul.rn.f16x2 h1, h1, {$r1}; "
        "mul.rn.f16x2 h2, h2, {$r1}; mul.rn.f16x2 h3, h3, {$r1}; "
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0; cvt.rn.satfinite.e4m3x2.f16x2 e1, h1; "
        "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2; cvt.rn.satfinite.e4m3x2.f16x2 e3, h3; "
        "mov.b32 {$w0}, {e0, e1}; mov.b32 {$w1}, {e2, e3}; }",
        write_only_types=[cutlass.Int32] * 2,
        read_only_args=[word, scale],
    )


def _uses_packed_fp4_mul():
    """Whether the compile target has PTX 9.4's packed E2M1 x E4M3 multiply."""
    from ....attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.tmem_transformed_kv import (
        _has_packed_e4m3_e2m1_mul,
    )

    return _has_packed_e4m3_e2m1_mul()


@cute.jit
def _fp4x8_mul_e4m3x8(word: cutlass.Int32, scale: cutlass.Int32):
    """Eight E2M1 values times one byte-broadcast E4M3 scale, as E4M3 RN/SATFINITE.

    The packed multiply takes one E2M1 value in the low nibble of each byte, so
    the even and odd nibbles are multiplied separately and their bytes
    re-interleaved: 7 instructions instead of the F16 route's 12 (4 cvt, 4 mul,
    4 cvt). The loader warps' conversion is the main-tile critical path.
    """
    return prims.inline_ptx(
        "{ .reg .b32 lo, hi; "
        "and.b32 lo, {$r0}, 0x0F0F0F0F; shr.b32 hi, {$r0}, 4; "
        "and.b32 hi, hi, 0x0F0F0F0F; "
        "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite lo, lo, {$r1}; "
        "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite hi, hi, {$r1}; "
        "prmt.b32 {$w0}, lo, hi, 0x5140; prmt.b32 {$w1}, lo, hi, 0x7362; }",
        write_only_types=[cutlass.Int32] * 2,
        read_only_args=[word, scale],
    )


@cute.jit
def _e4m3_byte_broadcast(scale_word: cutlass.Int32, selector: cutlass.Int32):
    """Replicate the scale byte chosen by a register PRMT selector four times."""
    return prims.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[scale_word, selector],
    )


@cute.jit
def _e4m3x2_to_f16x2_pair(scale_word: cutlass.Int32):
    """Two E4M3 group scales, each replicated across one F16x2 register."""
    packed = prims.inline_ptx(
        "cvt.rn.f16x2.e4m3x2 {$w0}, {$r0};",
        write_only_types=[cutlass.Int32],
        read_only_args=[scale_word.to(cutlass.Uint16)],
    ).to(cutlass.Uint32)
    return (
        ((packed & 65535) * 65537).to(cutlass.Int32),
        ((packed >> 16) * 65537).to(cutlass.Int32),
    )


@dataclass(kw_only=True)
class Fp4Fp8SmemKvResource(Fp4SmemKvResource):
    @cute.jit
    def _store_e4m3_chunk32(self, packed, stage_base, row, dim):
        # Thirty-two E4M3 bytes as two SW128 16-byte vectors; each 16-byte half
        # keeps its own XOR position (see _store_fragment).
        for half in cutlass.range_constexpr(2):
            offset = (row * 128 + dim + half * 16) ^ ((row & 7) * 16)
            dst = cutlass.inttoptr(
                stage_base.data_ptr().toint(cutlass.Int32) + offset,
                mem_space=3,
                dtype=cutlass.Int32,
            )
            dst.store(packed.load(half * 4, 4), alignment=16)

    @cute.jit
    def _store_fragment(self, fragment, stage_base, row, dim):
        # Scaled E2M1 is exact in inherited BF16 registers. Both sources are
        # already in common cache units: apply E4M3 RN/SATFINITE directly.
        values = fragment.load().to(cutlass.Float32)
        packed = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.rmem)
        for word in cutlass.range_constexpr(2):
            column = word * 4
            packed[word] = pack_float4_to_fp8_e4m3(
                values[column],
                values[column + 1],
                values[column + 2],
                values[column + 3],
            )
        # Native FP8 D128 descriptors use one SW128 byte plane. Eight values
        # occupy eight bytes, unlike the BF16 base's split-plane 16-byte store.
        offset = (row * 128 + dim) ^ ((row & 7) * 16)
        dst = cutlass.inttoptr(
            stage_base.data_ptr().toint(cutlass.Int32) + offset,
            mem_space=3,
            dtype=cutlass.Int32,
        )
        dst.store(packed.load(0, 2), alignment=8)


@dataclass(frozen=True)
class MixedFp8KvProducer(Fp4Fp8KvProducer):
    def compile_signature(self):
        return (
            "dsv41_shared_pertensor_joint_swa_fp4_e4m3_inline_metadata_v2",
            self.native_fp4,
        )

    @cute.jit
    def bind_sparse_views(
        self, indices, seq_lens, topk_lens, pools, capacity, request=None
    ):
        return bind_combined_sparse_views(
            indices, seq_lens, topk_lens, pools, capacity, request
        )

    def make_resource(self, pools, *, runtime_args=None, **kwargs):
        if runtime_args is not None:
            raise ValueError("mixed V4.1 KV does not accept request scales")
        return MixedFp8SmemKvResource(
            fp4_cache=pools[0],
            swa_cache=pools[1],
            native_fp4=self.native_fp4,
            **kwargs,
        )


# Raw FP4 staging for one BK128 main tile. Payload: four D128 stages times
# one thread-private 16B chunk per row group and loader thread (four groups of
# 128 threads or two of 256; 512 chunks per stage either way). Scales: each
# row's 32B (8B per stage), copied by the row's lanes.
_FP4_RAW_DATA_BYTES = 4 * 4 * 128 * 16
_FP4_RAW_BYTES = _FP4_RAW_DATA_BYTES + 128 * 32


@dataclass(kw_only=True)
class MixedFp8SmemKvResource(Fp4Fp8SmemKvResource):
    swa_cache: cute.Tensor = None
    _fp4_raw_alloc: object = None

    @property
    def _rows_per_warp(self):
        # 32 BK128 rows per loader warp with four warps, 16 with eight.
        return self.cfg.tile_size_kv // self.cfg.load_num_warps

    @property
    def _fp4_groups(self):
        # Four lanes cover one row's D128 slice, so a group spans eight rows.
        return self._rows_per_warp // 8

    @property
    def _loader_threads(self):
        return 32 * self.cfg.load_num_warps

    def get_smem_requirements(self):
        allocations = super().get_smem_requirements()
        if not self.native_fp4:
            return allocations
        if self._fp4_raw_alloc is None:
            self._fp4_raw_alloc = SmemAllocation(
                name=f"{self.name}_fp4_raw",
                size_bytes=_FP4_RAW_BYTES,
                alignment=128,
            )
        return [*allocations, self._fp4_raw_alloc]

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prefetch_staged_tile(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Fetch a main tile's raw FP4 rows before its first KV slot is free.

        Reused KV slots are released only after the previous tile's PV, which
        the in-order MMA issues after the next QK. The raw staging is private,
        so its copies overlap that wait instead of following it.
        """
        if cutlass.const_expr(not self.native_fp4):
            return
        if cutlass.const_expr(section == MlaStage.Loop):
            self._issue_fp4_tile(stage_info, cached_page_ids, page_id_slot=page_id_slot)
        elif cutlass.const_expr(self.cfg.use_multi_ctas_kv == 1):
            self._issue_fp4_tile(
                stage_info,
                cached_page_ids,
                page_id_slot=page_id_slot,
                enabled=self._head_is_main(stage_info),
            )

    @cute.jit
    def _head_is_main(self, stage_info: StageInfo):
        # KV splits are contiguous tile ranges, each starting with its own
        # HEAD, so only split 0's HEAD is the SWA tile. Warp-uniform.
        return cta_idx_kv_for_stage(self.cta_idx_kv, stage_info) != cutlass.Int32(0)

    @cute.jit
    def _issue_fp4_tile(
        self,
        stage_info: StageInfo,
        cached_page_ids,
        *,
        page_id_slot: cutlass.Constexpr[int],
        enabled=None,
    ):
        # Rows/chunks match the base loader: four lanes cover one row's D128
        # slice in 32-element chunks, eight rows per group, _fp4_groups groups
        # per warp (four with four loader warps, two with eight).
        # Register loads were rescheduled next to their conversions, which
        # serialized one L2 round trip per row group. Instead, cp.async copies
        # all four D128 stages of this tile (one commit group per stage) into
        # SMEM; each stage waits for its own group and converts from SMEM.
        # Invalid rows zero-fill payload and scales (src-size 0), giving the
        # same zero staging as the base loader without reading global memory.
        tid = cute.arch.thread_idx()[0] - self.cfg.load_warp_idx * 32
        lane = tid % 32
        warp = tid // 32
        pool = self.fp4_cache
        row_bytes, scale_bytes = 256, 32
        capacity = cutlass.Int64(cute.size(pool)) // (row_bytes + scale_bytes)
        chunk_idx = lane % 4
        raw = self._fp4_raw_base(stage_info)
        # All four stage calls of the tile reuse this selection. It stays
        # outside any runtime branch so those calls can see it; a disabled
        # (SWA) HEAD only clears any_valid and issues nothing.
        slots, valid, any_valid = self._fp4_select_rows(
            cached_page_ids,
            capacity,
            page_id_slot * self.cfg.sparse_cache_words,
            lane,
        )
        if cutlass.const_expr(enabled is not None):
            any_valid = any_valid & enabled
        self._fp4_tile_rows = slots, valid, any_valid
        # Every stage of a tile sees the same rows, so the issue and all four
        # stage calls take the same warp-uniform branch; group counts agree.
        if cute.arch.vote_any_sync(any_valid):
            pool_base = pool.iterator.toint()
            scale_dst = raw + _FP4_RAW_DATA_BYTES
            for stage in cutlass.range_constexpr(4):
                for group in cutlass.range_constexpr(self._fp4_groups):
                    slot = slots[group]
                    ok = valid[group]
                    # Widen before every byte-stride multiplication.
                    page_base = cutlass.Int64(slot // 64) * (
                        64 * (row_bytes + scale_bytes)
                    )
                    token = cutlass.Int64(slot & 63)
                    slot_idx = (
                        stage * self._fp4_groups + group
                    ) * self._loader_threads + tid
                    _cp_async_16(
                        raw + slot_idx * 16,
                        pool_base
                        + page_base
                        + token * row_bytes
                        + (stage * 64 + chunk_idx * 16),
                        cutlass.Int32(16) if ok else cutlass.Int32(0),
                    )
                    if cutlass.const_expr(stage == 0):
                        # Lane c copies its row's stage-c scale block.
                        row = warp * self._rows_per_warp + group * 8 + lane // 4
                        _cp_async_8(
                            scale_dst + row * scale_bytes + chunk_idx * 8,
                            pool_base
                            + page_base
                            + 64 * row_bytes
                            + token * scale_bytes
                            + chunk_idx * 8,
                            cutlass.Int32(8) if ok else cutlass.Int32(0),
                        )
                _cp_async_commit()

    @cute.jit
    def _fp4_raw_base(self, stage_info: StageInfo):
        return (
            smem_array(
                stage_info.context, self._fp4_raw_alloc, cutlass.Uint8, _FP4_RAW_BYTES
            )
            .data_ptr()
            .toint(cutlass.Int32)
        )

    @cute.jit
    def _load_fp4_tile_stage(
        self,
        stage_info: StageInfo,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        # Converts one D128 stage of the tile issued by prefetch_staged_tile.
        del cached_page_ids, page_id_slot
        tid = cute.arch.thread_idx()[0] - self.cfg.load_warp_idx * 32
        lane = tid % 32
        warp = tid // 32
        scale_bytes = 32
        chunk_idx = lane % 4
        raw = self._fp4_raw_base(stage_info)
        _, _, any_valid = self._fp4_tile_rows
        if cute.arch.vote_any_sync(any_valid):
            _cp_async_wait_group(3 - stage_idx)
            if cutlass.const_expr(stage_idx == 0):
                # Scale blocks were copied by the row's other lanes.
                cute.arch.sync_warp()
            datas = []
            scale_words = []
            for group in cutlass.range_constexpr(self._fp4_groups):
                row = warp * self._rows_per_warp + group * 8 + lane // 4
                slot_idx = (
                    stage_idx * self._fp4_groups + group
                ) * self._loader_threads + tid
                datas.append(
                    cutlass.inttoptr(
                        raw + slot_idx * 16, mem_space=3, dtype=cutlass.Int32
                    ).load(count=4, alignment=16)
                )
                scale_words.append(
                    cutlass.inttoptr(
                        raw
                        + _FP4_RAW_DATA_BYTES
                        + row * scale_bytes
                        + (stage_idx * 8)
                        + (chunk_idx // 2) * 4,
                        mem_space=3,
                        dtype=cutlass.Int32,
                    ).load()
                )
            # A lane's 32 values span two group16 scales: bytes 0/1 of its
            # scale word for even chunks, bytes 2/3 for odd ones.
            selector = (chunk_idx & 1) * cutlass.Int32(0x2222)
            for group in cutlass.range_constexpr(self._fp4_groups):
                row = warp * self._rows_per_warp + group * 8 + lane // 4
                packed = cutlass.Array(
                    cutlass.Int32, 8, space=cutlass.AddressSpace.rmem
                )
                if cutlass.const_expr(_uses_packed_fp4_mul()):
                    scale0 = _e4m3_byte_broadcast(scale_words[group], selector)
                    scale1 = _e4m3_byte_broadcast(
                        scale_words[group], selector + cutlass.Int32(0x1111)
                    )
                    for word in cutlass.range_constexpr(4):
                        lo, hi = _fp4x8_mul_e4m3x8(
                            cutlass.Int32(datas[group][word]),
                            scale0 if word < 2 else scale1,
                        )
                        packed[2 * word] = lo
                        packed[2 * word + 1] = hi
                else:
                    scale0, scale1 = _e4m3x2_to_f16x2_pair(
                        (scale_words[group] >> ((chunk_idx & 1) * 16))
                        & cutlass.Int32(0xFFFF)
                    )
                    for word in cutlass.range_constexpr(4):
                        lo, hi = _fp4x8_scaled_to_e4m3x8(
                            cutlass.Int32(datas[group][word]),
                            scale0 if word < 2 else scale1,
                        )
                        packed[2 * word] = lo
                        packed[2 * word + 1] = hi
                self._store_e4m3_chunk32(packed, stage_base, row, chunk_idx * 32)
        else:
            packed = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.rmem)
            for word in cutlass.range_constexpr(8):
                packed[word] = cutlass.Int32(0)
            for group in cutlass.range_constexpr(self._fp4_groups):
                row = warp * self._rows_per_warp + group * 8 + lane // 4
                self._store_e4m3_chunk32(packed, stage_base, row, chunk_idx * 32)

    @cute.jit
    def _fp4_select_rows(self, cached_page_ids, capacity, base, lane):
        slots = []
        valid = []
        any_valid = cutlass.Boolean(False)
        for group in cutlass.range_constexpr(self._fp4_groups):
            local_row = group * 8 + lane // 4
            raw_id = cutlass.Int32(0x7FFFFFFF)
            for entry in cutlass.range_constexpr(4):
                value = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_page_ids[base + entry],
                    offset=local_row // 4,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                if local_row % 4 == entry:
                    raw_id = cutlass.Int32(value)
            slot = raw_id & cutlass.Int32(0x7FFFFFFF)
            ok = (slot != cutlass.Int32(0x7FFFFFFF)) & (cutlass.Int64(slot) < capacity)
            slots.append(slot if ok else cutlass.Int32(0))
            valid.append(ok)
            any_valid = any_valid | ok
        return slots, valid, any_valid

    @cute.jit
    def _load_swa_stage(
        self,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
        capacity = cutlass.Int64(cute.size(self.swa_cache)) // 512
        base = page_id_slot * self.cfg.sparse_cache_words
        dim = (lane % 8) * 16
        # Eight lanes copy each row's D128 slice in disjoint aligned 16B
        # vectors. Smaller copies respect SW128's separate 16B half-chunks;
        # an unpermuted 32B copy would silently exchange their contents.
        # cp.async has no register destination, so rows stay in flight
        # together: register loads were rescheduled next to their consumers,
        # which serialized one L2 round trip per row. Invalid rows use
        # src-size 0 (zero fill, no global read) at the always-mapped row 0.
        #
        # The SWA HEAD is the first use of the nonpersistent KV ring, so its
        # four D128 stages are physical stages 0-3 and already empty. Stage 0
        # issues all four (one commit group each); stage s then waits only for
        # its own group before the inherited fence and all-writer commit.
        if cutlass.const_expr(self.cfg.use_persistent_scheduler == 1):
            raise ValueError("V4.1 SWA prefetch requires a nonpersistent KV ring")
        if cutlass.const_expr(stage_idx == 0):
            stage_bytes = self.cfg.kv_smem_stage_elements * self.cfg.qkv_dtype_bytes
            stage_dst = stage_base.data_ptr().toint(cutlass.Int32)
            swa_base = self.swa_cache.iterator.toint() + cutlass.Int64(dim)
            rows = []
            # Eight lanes per row: four rows per group.
            for group in cutlass.range_constexpr(self._rows_per_warp // 4):
                local_row = group * 4 + lane // 8
                raw = cutlass.Int32(0x7FFFFFFF)
                for entry in cutlass.range_constexpr(4):
                    value = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=cached_page_ids[base + entry],
                        offset=local_row // 4,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                    )
                    if local_row % 4 == entry:
                        raw = cutlass.Int32(value)
                slot = raw & cutlass.Int32(0x7FFFFFFF)
                ok = (slot != cutlass.Int32(0x7FFFFFFF)) & (
                    cutlass.Int64(slot) < capacity
                )
                row = warp * self._rows_per_warp + local_row
                rows.append(
                    (
                        (row * 128 + dim) ^ ((row & 7) * 16),
                        swa_base
                        + cutlass.Int64(slot if ok else cutlass.Int32(0)) * 512,
                        cutlass.Int32(16) if ok else cutlass.Int32(0),
                    )
                )
            for stage in cutlass.range_constexpr(4):
                for offset, src, size in rows:
                    _cp_async_16(
                        stage_dst + stage * stage_bytes + offset,
                        src + stage * 128,
                        size,
                    )
                _cp_async_commit()
        # Each writer completes its own copies before the inherited
        # async-proxy fence and all-writer commit; UMMA semantics match the
        # previous thread stores.
        _cp_async_wait_group(3 - stage_idx)

    @cute.jit
    def _load_main_stage(
        self,
        stage_info: StageInfo,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.native_fp4):
            self._load_fp4_tile_stage(
                stage_info,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        else:
            self._load_fp4_stage(
                self.fp4_cache,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        if cutlass.const_expr(inst_id != 0 or is_v != 0):
            raise ValueError("mixed V4.1 KV retains K through PV on one stream")
        stage_base = self._stage_base(stage_info)
        # The fixed 128-slot primary span guarantees exactly one SWA tile, the
        # first HEAD. Keep-AB's single KV pipe loads every later tile (LOOP,
        # and the HEADs of later KV splits) from main; counts cannot shorten
        # that span, even for a completely masked SWA.
        if cutlass.const_expr(
            section == MlaStage.Head and self.cfg.use_multi_ctas_kv == 1
        ):
            if self._head_is_main(stage_info):
                self._load_main_stage(
                    stage_info,
                    stage_base,
                    cached_page_ids,
                    stage_idx=stage_idx,
                    page_id_slot=page_id_slot,
                )
            else:
                self._load_swa_stage(
                    stage_base,
                    cached_page_ids,
                    stage_idx=stage_idx,
                    page_id_slot=page_id_slot,
                )
        elif cutlass.const_expr(section == MlaStage.Head):
            self._load_swa_stage(
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        elif cutlass.const_expr(section == MlaStage.Loop):
            self._load_main_stage(
                stage_info,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        else:
            raise ValueError("unsupported mixed V4.1 KV loading phase")
        # Both sources use thread stores and the same all-128-writer commit.
        # This deliberately avoids mixing TMA completion with thread arrivals;
        # retained K is released by the native graph only after delayed PV.
        cute.arch.fence_view_async_shared()
